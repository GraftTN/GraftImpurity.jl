"""
    _bathfit_component(input)

Return the source representation recorded by a `BathFitInput`. Manual
real-axis inputs retain the historical, explicit default `:spectral`; typed
GreenFunc inputs carry their component in source metadata.
"""
function _bathfit_component(input::BathFitInput)
    component = hasproperty(input.metadata, :component) ?
                getproperty(input.metadata, :component) :
                (input.domain === :matsubara ? :matsubara : :spectral)
    component isa Symbol || throw(ArgumentError("BathFitInput component must be a Symbol"))
    if input.domain === :matsubara
        component === :matsubara || throw(ArgumentError(
            "a Matsubara BathFitInput must use component=:matsubara",
        ))
    else
        component in (:spectral, :retarded) || throw(ArgumentError(
            "a real-axis BathFitInput must use component=:spectral or :retarded",
        ))
    end
    return component
end

function _reconstruction_broadening(input::BathFitInput, broadening)
    if input.domain === :matsubara
        broadening === nothing && return nothing
        value = Float64(broadening)
        isfinite(value) && iszero(value) || throw(ArgumentError(
            "Matsubara reconstruction does not accept nonzero broadening",
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

function _bathfit_reconstruction_value(bath::DiscreteBath,
                                       block_index_value::Int,
                                       frequency::Float64,
                                       component::Symbol,
                                       broadening::Union{Nothing,Float64})
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
    names = block_names(bath.partition)
    samples = Tuple(begin
        block_index_value = block_index(bath.partition, block)
        Matrix{ComplexF64}[
            _bathfit_reconstruction_value(
                bath, block_index_value, frequency, component, eta,
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
    block_index_value = block_index(bath.partition, resolved_block)
    reconstructed = Matrix{ComplexF64}[
        _bathfit_reconstruction_value(
            bath, block_index_value, frequency, component, eta,
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
