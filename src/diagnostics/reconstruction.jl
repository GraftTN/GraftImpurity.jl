"""
    _bathfit_component(input)

Return the source representation recorded by a `BathFitInput`. Manual
real-axis inputs retain the historical, explicit default `:spectral`; typed
GreenFunc inputs carry their component in source metadata.
"""
function _bathfit_component(input::BathFitInput)
    component = hasproperty(input.metadata, :component) ?
                getproperty(input.metadata, :component) :
                (input.domain in (:matsubara, :imaginary_time) ?
                 :matsubara : :spectral)
    component isa Symbol || throw(ArgumentError("BathFitInput component must be a Symbol"))
    if input.domain in (:matsubara, :imaginary_time)
        component === :matsubara || throw(ArgumentError(
            "an imaginary-axis BathFitInput must use component=:matsubara",
        ))
    else
        component in (:spectral, :retarded) || throw(ArgumentError(
            "a real-axis BathFitInput must use component=:spectral or :retarded",
        ))
    end
    return component
end

function _reconstruction_broadening(input::BathFitInput, broadening)
    if input.domain in (:matsubara, :imaginary_time)
        broadening === nothing && return nothing
        value = Float64(broadening)
        isfinite(value) && iszero(value) || throw(ArgumentError(
            "imaginary-axis reconstruction does not accept nonzero broadening",
        ))
        return nothing
    end
    broadening === nothing && throw(ArgumentError(
        "real-axis reconstruction requires an explicit positive broadening",
    ))
    value = Float64(broadening)
    isfinite(value) && value > 0 || throw(ArgumentError(
        "real-axis reconstruction broadening must be finite and positive",
    ))
    return value
end

_bathfit_resolvent(z::ComplexF64, energy::Float64, ::Val{:fermion}) =
    inv(z - energy)

_bathfit_resolvent(z::ComplexF64, energy::Float64, ::Val{:boson}) =
    iszero(z) ? -1.0 + 0im : energy / (z - energy)

function _bathfit_resolvent(z::ComplexF64, energy::Float64, statistics::Symbol)
    statistics in (:fermion, :boson) || throw(ArgumentError(
        "bath statistics must be :fermion or :boson",
    ))
    return _bathfit_resolvent(z, energy, Val(statistics))
end

function _bathfit_block_value(bath::DiscreteBath, block_index_value::Int,
                              z::ComplexF64)
    dimension = length(block_flavors(
        bath.partition, block_names(bath.partition)[block_index_value],
    ))
    value = zeros(ComplexF64, dimension, dimension)
    for mode_index in eachindex(bath.orbitals.energies)
        bath.orbitals.block_indices[mode_index] == block_index_value || continue
        coupling = bath.orbitals.couplings[mode_index]
        value .+= _bathfit_resolvent(
            z, bath.orbitals.energies[mode_index], bath.statistics,
        ) .* (coupling * coupling')
    end
    return value
end

function _bathfit_imaginary_time_factor(tau::Float64, beta::Float64,
                                         energy::Float64)
    if energy >= 0
        return exp(-tau * energy) / (1 + exp(-beta * energy))
    end
    return exp((beta - tau) * energy) / (1 + exp(beta * energy))
end

function _bathfit_imaginary_time_block_value(bath::DiscreteBath,
                                              block_index_value::Int,
                                              tau::Float64,
                                              beta::Float64)
    bath.statistics === :fermion || throw(ArgumentError(
        "imaginary-time bath reconstruction currently supports only fermionic baths",
    ))
    dimension = length(block_flavors(
        bath.partition, block_names(bath.partition)[block_index_value],
    ))
    value = zeros(ComplexF64, dimension, dimension)
    for mode_index in eachindex(bath.orbitals.energies)
        bath.orbitals.block_indices[mode_index] == block_index_value || continue
        energy = bath.orbitals.energies[mode_index]
        coupling = bath.orbitals.couplings[mode_index]
        factor = _bathfit_imaginary_time_factor(tau, beta, energy)
        value .-= factor .* (coupling * coupling')
    end
    return value
end

function _bathfit_reconstruction_value(bath::DiscreteBath,
                                       block_index_value::Int,
                                       frequency::Float64,
                                       domain::Symbol,
                                       component::Symbol,
                                       broadening::Union{Nothing,Float64},
                                       beta::Union{Nothing,Float64})
    if domain === :imaginary_time
        beta === nothing && throw(ArgumentError(
            "imaginary-time reconstruction requires beta from the tau grid",
        ))
        return _bathfit_imaginary_time_block_value(
            bath, block_index_value, frequency, beta,
        )
    end
    if component === :matsubara
        return _bathfit_block_value(bath, block_index_value, im * frequency)
    end
    broadening === nothing && throw(ArgumentError(
        "real-axis reconstruction requires an explicit positive broadening",
    ))
    retarded = _bathfit_block_value(
        bath, block_index_value, frequency + im * broadening,
    )
    component === :retarded && return retarded
    component === :spectral || throw(ArgumentError(
        "real-axis reconstruction component must be :spectral or :retarded",
    ))
    # This full matrix expression preserves complex off-diagonal spectral
    # weight, unlike elementwise `imag` which would erase it.
    return (adjoint(retarded) - retarded) / (2pi * im)
end

function _reconstructed_greenfunc(template::GreenFunc.Gf,
                                  samples::Vector{Matrix{ComplexF64}})
    output = similar(template, ComplexF64)
    length(only(template.mesh)) == length(samples) || throw(DimensionMismatch(
        "GreenFunc reconstruction sample count does not match its physical mesh",
    ))
    if template.target_ndim == 0
        all(sample -> size(sample) == (1, 1), samples) || throw(DimensionMismatch(
            "scalar GreenFunc reconstruction requires scalar bath block samples",
        ))
        for index in eachindex(samples)
            output.data[index] = samples[index][1, 1]
        end
    elseif template.target_ndim == 2
        size(output.data, 3) == length(samples) || throw(DimensionMismatch(
            "matrix GreenFunc reconstruction requires frequency as the final axis",
        ))
        expected = template.target_shape
        all(sample -> size(sample) == expected, samples) || throw(DimensionMismatch(
            "GreenFunc reconstruction block shape does not match its template",
        ))
        for index in eachindex(samples)
            @views output.data[:, :, index] .= samples[index]
        end
    else
        throw(ArgumentError(
            "bath reconstruction supports scalar or matrix GreenFunc targets",
        ))
    end
    return output
end

function _reconstructed_template(input::BathFitInput,
                                 blocks::NamedTuple)
    template = input.source_template
    template === nothing && return nothing
    names = Tuple(keys(blocks))
    if template isa GreenFunc.Gf
        length(names) == 1 || throw(ArgumentError(
            "a single GreenFunc template requires a one-block BathFitInput",
        ))
        return _reconstructed_greenfunc(template, only(values(blocks)))
    elseif template isa GreenFunc.BlockGf
        Tuple(keys(template)) == names || throw(ArgumentError(
            "BlockGf template names do not match BathFitInput blocks",
        ))
        return GreenFunc.BlockGf(
            (name => _reconstructed_greenfunc(template[name], getproperty(blocks, name))
             for name in names)...,
        )
    end
    throw(ArgumentError(
        "BathFitInput source_template must be nothing, GreenFunc.Gf, or GreenFunc.BlockGf",
    ))
end

function _reconstruct_bathfit_input(bath::DiscreteBath, input::BathFitInput;
                                    broadening=nothing)
    bath.layout == input.layout || throw(ArgumentError(
        "DiscreteBath FlavorLayout does not match BathFitInput",
    ))
    bath.statistics === input.statistics || throw(ArgumentError(
        "DiscreteBath statistics do not match BathFitInput",
    ))
    _validate_fit_input(input, bath.partition)
    component = _bathfit_component(input)
    eta = _reconstruction_broadening(input, broadening)
    beta = input.domain === :imaginary_time ?
           _bathfit_imaginary_time_beta(input.frequencies) : nothing
    names = block_names(bath.partition)
    samples = Tuple(begin
        block_index_value = block_index(bath.partition, block)
        Matrix{ComplexF64}[
            _bathfit_reconstruction_value(
                bath, block_index_value, frequency, input.domain, component,
                eta, beta,
            ) for frequency in input.frequencies
        ]
    end for block in names)
    blocks = NamedTuple{names}(samples)
    template = _reconstructed_template(input, blocks)
    return BathFitInput(
        input.layout, input.domain, input.statistics, copy(input.frequencies),
        blocks, input.target_labels, input.metadata, template, Val(:validated),
    )
end

"""
    reconstruct_hybridization(bath, input::BathFitInput; broadening=nothing)

Reconstruct every named block on the source grid carried by `input`. The
returned `BathFitInput` retains its labels and metadata and, when `input` was
adapted from GreenFunc data, carries a reconstructed copy of the original
`Gf`/`BlockGf` template in `source_template`.
"""
function reconstruct_hybridization(bath::DiscreteBath, input::BathFitInput;
                                   broadening=nothing)
    return _reconstruct_bathfit_input(bath, input; broadening)
end

function _reconstruction_input_for_greenfunc(bath::DiscreteBath,
                                             template::GreenFunc.Gf,
                                             block::Union{Nothing,Symbol})
    resolved_block = if block === nothing
        names = block_names(bath.partition)
        length(names) == 1 || throw(ArgumentError(
            "a multi-block bath needs an explicit block keyword for a single GreenFunc template",
        ))
        only(names)
    else
        block
    end
    return BathFitInput(bath.layout, template, resolved_block)
end

function _reconstruct_greenfunc_block(bath::DiscreteBath,
                                      template::GreenFunc.Gf,
                                      block::Union{Nothing,Symbol};
                                      broadening=nothing)
    input = _reconstruction_input_for_greenfunc(bath, template, block)
    bath.statistics === input.statistics || throw(ArgumentError(
        "DiscreteBath statistics do not match GreenFunc template",
    ))
    resolved_block = only(keys(input.blocks))
    expected_flavors = block_flavors(bath.partition, resolved_block)
    samples = only(values(input.blocks))
    all(sample -> size(sample) == (length(expected_flavors),
                                   length(expected_flavors)), samples) ||
        throw(DimensionMismatch(
            "GreenFunc template target shape does not match bath block $resolved_block",
        ))
    labels = only(values(input.target_labels))
    labels === nothing || labels == (expected_flavors, expected_flavors) ||
        throw(ArgumentError(
            "GreenFunc template target labels do not match bath block $resolved_block",
        ))
    component = _bathfit_component(input)
    eta = _reconstruction_broadening(input, broadening)
    beta = input.domain === :imaginary_time ?
           _bathfit_imaginary_time_beta(input.frequencies) : nothing
    block_index_value = block_index(bath.partition, resolved_block)
    reconstructed = Matrix{ComplexF64}[
        _bathfit_reconstruction_value(
            bath, block_index_value, frequency, input.domain, component,
            eta, beta,
        ) for frequency in input.frequencies
    ]
    return _reconstructed_greenfunc(template, reconstructed)
end

"""
    reconstruct_hybridization(bath, template::GreenFunc.Gf;
                              block=nothing, broadening=nothing)

Reconstruct one named bath block into a `GreenFunc.Gf` that preserves the
template's mesh, target shape, labels, statistics, temperature, and metadata.
For a one-block bath, `block` may be omitted.
"""
function reconstruct_hybridization(bath::DiscreteBath, template::GreenFunc.Gf;
                                   block::Union{Nothing,Symbol}=nothing,
                                   broadening=nothing)
    return _reconstruct_greenfunc_block(bath, template, block; broadening)
end

"""
    reconstruct_hybridization(bath, template::GreenFunc.BlockGf;
                              broadening=nothing)

Reconstruct all named bath blocks into a `GreenFunc.BlockGf`, retaining its
name/order and every per-block GreenFunc semantic field.
"""
function reconstruct_hybridization(bath::DiscreteBath,
                                   template::GreenFunc.BlockGf;
                                   broadening=nothing)
    input = BathFitInput(bath.layout, template)
    reconstructed = _reconstruct_bathfit_input(bath, input; broadening)
    return reconstructed.source_template::GreenFunc.BlockGf
end

function _validate_residual_inputs(source::BathFitInput,
                                   reconstruction::BathFitInput)
    source.layout == reconstruction.layout || throw(ArgumentError(
        "residual hybridization inputs must use the same FlavorLayout",
    ))
    source.domain === reconstruction.domain || throw(ArgumentError(
        "residual hybridization inputs must use the same domain",
    ))
    source.statistics === reconstruction.statistics || throw(ArgumentError(
        "residual hybridization inputs must use the same statistics",
    ))
    source.frequencies == reconstruction.frequencies || throw(ArgumentError(
        "residual hybridization inputs must use the same frequency values and order",
    ))
    names = Tuple(keys(source.blocks))
    names == Tuple(keys(reconstruction.blocks)) || throw(ArgumentError(
        "residual hybridization inputs must use the same block names and order",
    ))
    source.target_labels == reconstruction.target_labels || throw(ArgumentError(
        "residual hybridization inputs must use the same target labels",
    ))
    for name in names
        source_samples = getproperty(source.blocks, name)
        reconstruction_samples = getproperty(reconstruction.blocks, name)
        length(source_samples) == length(source.frequencies) || throw(DimensionMismatch(
            "source block $name does not match its frequency grid",
        ))
        length(reconstruction_samples) == length(source.frequencies) ||
            throw(DimensionMismatch(
                "reconstruction block $name does not match the source frequency grid",
            ))
        for (index, (source_sample, reconstruction_sample)) in enumerate(
                zip(source_samples, reconstruction_samples))
            size(source_sample) == size(reconstruction_sample) ||
                throw(DimensionMismatch(
                    "residual hybridization block $name sample $index has mismatched shapes",
                ))
        end
    end
    return names
end

"""
    residual_hybridization(source::BathFitInput,
                           reconstruction::BathFitInput)

Compute the signed residual `source - reconstruction` without changing the
source layout, domain, statistics, frequency or block order, target labels, or
metadata. A GreenFunc-backed source retains an equally typed residual template
with the source mesh and semantic metadata.
"""
function residual_hybridization(source::BathFitInput,
                                reconstruction::BathFitInput)
    names = _validate_residual_inputs(source, reconstruction)
    blocks = NamedTuple{names}(Tuple(
        Matrix{ComplexF64}[
            source_sample - reconstruction_sample
            for (source_sample, reconstruction_sample) in zip(
                getproperty(source.blocks, name),
                getproperty(reconstruction.blocks, name),
            )
        ] for name in names
    ))
    template = _reconstructed_template(source, blocks)
    return BathFitInput(
        source.layout, source.domain, source.statistics,
        copy(source.frequencies), blocks, source.target_labels,
        source.metadata, template, Val(:validated),
    )
end

"""
    residual_hybridization(source::BathFitInput, bath::DiscreteBath;
                           broadening=nothing)

Reconstruct `bath` on the source grid, then return the signed residual
`source - reconstruction`. Real-axis sources require the same explicit
positive broadening as [`reconstruct_hybridization`](@ref).
"""
function residual_hybridization(source::BathFitInput, bath::DiscreteBath;
                                broadening=nothing)
    reconstruction = reconstruct_hybridization(bath, source; broadening)
    return residual_hybridization(source, reconstruction)
end

"""
    residual_hybridization(report::BathFitReport)

Return the signed source-minus-reconstruction input stored by a bath-fit
report. Reports without a reconstruction fail explicitly.
"""
function residual_hybridization(report::BathFitReport)
    reconstruction = report.reconstruction
    reconstruction === nothing && throw(ArgumentError(
        "BathFitReport has no reconstruction for residual_hybridization",
    ))
    return residual_hybridization(report.source, reconstruction)
end
