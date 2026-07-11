import GreenFunc
import SparseIR

"""
    IRCoefficients

SparseIR expansion coefficients for one physical axis of a `GreenFunc.Gf`.
`axis` numbers physical axes only; target axes remain the leading
`target_ndim` dimensions of `coefficients`.
"""
struct IRCoefficients{B,N,MT,L,M}
    basis::B
    coefficients::Array{ComplexF64,N}
    axis::Int
    meshes::MT
    target_ndim::Int
    statistics::Bool
    component::Symbol
    target_labels::L
    metadata::M
end

_ir_isfermionic(basis) = SparseIR.statistics(basis) isa SparseIR.Fermionic

function _ir_axis(gf::GreenFunc.Gf, axis)
    if isnothing(axis)
        candidates = findall(mesh -> mesh isa Union{GreenFunc.ImTime,GreenFunc.ImFreq},
            gf.mesh)
        length(candidates) == 1 || throw(ArgumentError(
            "axis must be specified unless the Green's function has exactly one imaginary-time or imaginary-frequency physical mesh",
        ))
        return only(candidates)
    end

    axis isa Integer || throw(ArgumentError("axis must be an integer physical-axis index"))
    physical_axis = Int(axis)
    1 <= physical_axis <= length(gf.mesh) ||
        throw(ArgumentError("physical axis $physical_axis is out of range"))
    gf.mesh[physical_axis] isa Union{GreenFunc.ImTime,GreenFunc.ImFreq} ||
        throw(ArgumentError("physical axis $physical_axis is not an imaginary-time or imaginary-frequency mesh"))
    return physical_axis
end

function _validate_ir_mesh(basis, statistics::Bool,
                           mesh::Union{GreenFunc.ImTime,GreenFunc.ImFreq})
    mesh.isFermi == statistics ||
        throw(ArgumentError("mesh statistics do not match the IR coefficients"))
    _ir_isfermionic(basis) == statistics ||
        throw(ArgumentError("SparseIR basis statistics do not match the Green's function"))
    isapprox(Float64(SparseIR.beta(basis)), Float64(mesh.β);
             atol=0, rtol=sqrt(eps(Float64))) ||
        throw(ArgumentError("SparseIR basis and mesh have different inverse temperatures"))
    return nothing
end

function _ir_sampling(basis, mesh::GreenFunc.ImTime)
    return SparseIR.TauSampling(basis; sampling_points=Float64.(collect(mesh)))
end

function _ir_sampling(basis, mesh::GreenFunc.ImFreq)
    # GreenFunc indexes Matsubara points by m, while SparseIR uses the parity
    # carrying integer n in ω = nπ/β: n = 2m + ζ.
    m = GreenFunc.matfreq_to_int.(Ref(mesh), GreenFunc.matfreq(mesh))
    n = 2 .* m .+ Int(mesh.isFermi)
    return SparseIR.MatsubaraSampling(basis;
        sampling_points=n, positive_only=false)
end

"""
    fit_ir(gf, basis; axis=nothing) -> IRCoefficients

Fit one imaginary-time or imaginary-frequency physical axis of `gf` to
`basis`. If `axis` is omitted, the axis is inferred only when exactly one
supported physical mesh is present.
"""
function fit_ir(gf::GreenFunc.Gf, basis; axis=nothing)
    physical_axis = _ir_axis(gf, axis)
    mesh = gf.mesh[physical_axis]
    _validate_ir_mesh(basis, gf.statistics, mesh)

    data_dim = gf.target_ndim + physical_axis
    input = ComplexF64.(gf.data)
    output_size = collect(size(input))
    output_size[data_dim] = length(basis)
    coefficients = Array{ComplexF64}(undef, output_size...)
    sampling = _ir_sampling(basis, mesh)
    SparseIR.fit!(coefficients, sampling, input; dim=data_dim)

    return IRCoefficients(
        basis,
        coefficients,
        physical_axis,
        gf.mesh,
        gf.target_ndim,
        gf.statistics,
        gf.component,
        gf.target_labels,
        gf.metadata,
    )
end

function _validate_ir_coefficients(coeffs::IRCoefficients)
    0 <= coeffs.target_ndim < ndims(coeffs.coefficients) ||
        throw(ArgumentError("invalid target_ndim in IRCoefficients"))
    1 <= coeffs.axis <= length(coeffs.meshes) ||
        throw(ArgumentError("invalid physical axis in IRCoefficients"))
    data_dim = coeffs.target_ndim + coeffs.axis
    size(coeffs.coefficients, data_dim) == length(coeffs.basis) ||
        throw(DimensionMismatch("IR coefficient axis does not match the basis size"))
    return data_dim
end

"""
    evaluate_ir(coeffs, mesh::Union{ImTime,ImFreq}) -> GreenFunc.Gf

Evaluate `coeffs` on an imaginary-time or imaginary-frequency mesh, replacing
only the fitted physical axis and preserving all target semantics and metadata.
Real-axis continuation is intentionally not part of this adapter.
"""
function evaluate_ir(coeffs::IRCoefficients,
                     mesh::Union{GreenFunc.ImTime,GreenFunc.ImFreq})
    data_dim = _validate_ir_coefficients(coeffs)
    _validate_ir_mesh(coeffs.basis, coeffs.statistics, mesh)

    sampling = _ir_sampling(coeffs.basis, mesh)
    output_size = collect(size(coeffs.coefficients))
    output_size[data_dim] = length(mesh)
    values = Array{ComplexF64}(undef, output_size...)
    SparseIR.evaluate!(values, sampling, coeffs.coefficients; dim=data_dim)

    meshes = Base.setindex(coeffs.meshes, mesh, coeffs.axis)
    target_shape = ntuple(i -> size(values, i), coeffs.target_ndim)
    return GreenFunc.Gf(meshes...;
        target_shape,
        data=values,
        dtype=ComplexF64,
        statistics=coeffs.statistics,
        component=coeffs.component,
        target_labels=coeffs.target_labels,
        metadata=coeffs.metadata,
    )
end

function _ir_mesh_options(coeffs::IRCoefficients)
    _validate_ir_coefficients(coeffs)
    source = coeffs.meshes[coeffs.axis]
    source isa Union{GreenFunc.ImTime,GreenFunc.ImFreq} ||
        throw(ArgumentError("IRCoefficients source mesh is not an imaginary mesh"))
    return (; beta=Float64(SparseIR.beta(coeffs.basis)),
        isfermi=coeffs.statistics, Euv=source.Euv, rtol=source.rtol,
        symmetry=source.symmetry)
end

"""
    to_imtime_ir(coeffs; grid=nothing) -> GreenFunc.Gf

Evaluate IR coefficients in imaginary time. The default grid is SparseIR's
full default `TauSampling` grid for the stored basis.
"""
function to_imtime_ir(coeffs::IRCoefficients; grid=nothing)
    options = _ir_mesh_options(coeffs)
    points = isnothing(grid) ?
        copy(SparseIR.sampling_points(SparseIR.TauSampling(coeffs.basis))) :
        collect(grid)
    mesh = GreenFunc.ImTime(options.beta, options.isfermi;
        Euv=options.Euv, rtol=options.rtol, symmetry=options.symmetry,
        grid=points)
    return evaluate_ir(coeffs, mesh)
end

"""
    to_imfreq_ir(coeffs; grid=nothing) -> GreenFunc.Gf

Evaluate IR coefficients on a Matsubara mesh. A supplied `grid` uses
GreenFunc's integer `m` convention. The default is converted from SparseIR's
full Matsubara sampling integers `n = 2m + ζ`.
"""
function to_imfreq_ir(coeffs::IRCoefficients; grid=nothing)
    options = _ir_mesh_options(coeffs)
    if isnothing(grid)
        sampling = SparseIR.MatsubaraSampling(coeffs.basis; positive_only=false)
        n = Int.(SparseIR.sampling_points(sampling))
        parity = Int(options.isfermi)
        all(value -> iseven(value - parity), n) ||
            throw(ArgumentError("SparseIR returned Matsubara points with inconsistent statistics"))
        points = div.(n .- parity, 2)
    else
        points = collect(grid)
    end
    mesh = GreenFunc.ImFreq(options.beta, options.isfermi;
        Euv=options.Euv, rtol=options.rtol, symmetry=options.symmetry,
        grid=points)
    return evaluate_ir(coeffs, mesh)
end
