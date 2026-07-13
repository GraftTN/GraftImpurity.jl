"""
    BCFFitInput(layout, times, blocks...; channel, metadata=(;))

Validated time-domain bath-correlation-function input for the complex BCF
MiniPole route. Each named block contains scalar or square matrix samples on a
common uniform grid beginning at zero. The type is deliberately distinct from
`BathFitInput`: a BCF exponential sum is not a Hamiltonian hybridization.
"""
struct BCFFitInput{B<:NamedTuple,M<:NamedTuple}
    layout::FlavorLayout
    times::Vector{Float64}
    blocks::B
    channel::Symbol
    metadata::M

    function BCFFitInput(layout::FlavorLayout, times::Vector{Float64},
                         blocks::B, channel::Symbol, metadata::M,
                         ::Val{:validated}) where {B<:NamedTuple,M<:NamedTuple}
        new{B,M}(layout, times, blocks, channel, metadata)
    end
end

function _validate_bcf_times(times::AbstractVector{<:Real})
    length(times) >= 4 || throw(ArgumentError(
        "BCFFitInput needs at least four uniform time samples for matrix ESPRIT",
    ))
    values = Float64.(times)
    all(isfinite, values) ||
        throw(ArgumentError("BCFFitInput times must be finite"))
    all(value -> value >= 0, values) ||
        throw(ArgumentError("BCFFitInput times must be nonnegative"))
    iszero(first(values)) ||
        throw(ArgumentError("BCFFitInput requires its canonical time grid to start at zero"))
    steps = diff(values)
    all(step -> step > 0, steps) ||
        throw(ArgumentError("BCFFitInput times must be strictly increasing"))
    step = first(steps)
    all(candidate -> isapprox(candidate, step; atol=32 * eps(Float64) *
                              max(1.0, abs(step)), rtol=32 * eps(Float64)),
        steps) || throw(ArgumentError(
        "BCFFitInput times must be uniformly spaced for matrix ESPRIT",
    ))
    return values
end

function BCFFitInput(layout::FlavorLayout, times::AbstractVector{<:Real},
                     blocks::Pair...; channel::Symbol,
                     metadata::NamedTuple=(;))
    isempty(String(channel)) &&
        throw(ArgumentError("BCFFitInput channel must be a nonempty Symbol"))
    isempty(blocks) && throw(ArgumentError("BCFFitInput needs named blocks"))
    values = _validate_bcf_times(times)
    names = Symbol[first(block) for block in blocks]
    allunique(names) || throw(ArgumentError("BCFFitInput block names must be unique"))
    samples = Tuple(_fit_sample_matrices(block.second, length(values))
                    for block in blocks)
    canonical = NamedTuple{Tuple(names)}(samples)
    return BCFFitInput(layout, values, canonical, channel, metadata,
                       Val(:validated))
end

function _validate_bcf_input(input::BCFFitInput, partition::Partition)
    validate_partition(partition, input.layout)
    Tuple(keys(input.blocks)) == block_names(partition) ||
        throw(ArgumentError(
            "BCFFitInput block names and order must match the named Partition",
        ))
    for block in block_names(partition)
        dimension = length(block_flavors(partition, block))
        samples = getproperty(input.blocks, block)
        all(sample -> size(sample) == (dimension, dimension), samples) ||
            throw(DimensionMismatch(
                "BCFFitInput block $block does not match its Partition dimension",
            ))
    end
    return input
end

_bcf_block_samples(input::BCFFitInput, block::Symbol) =
    getproperty(input.blocks, block)

_bcf_timestep(input::BCFFitInput) = input.times[2] - input.times[1]
