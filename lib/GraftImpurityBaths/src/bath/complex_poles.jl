"""
    ComplexPoles(layout, partition, poles, weights, block_indices;
                 channel, diagnostics=(;), stability_tolerance=0.0)

Typed complex bath-correlation-function (BCF) exponential sum. Its canonical
convention is `C_b(t) = sum_k W_k * exp(-z_k * t)` for `t >= 0`, with
`z_k = gamma_k + im * omega_k` and nonnegative damping `gamma_k` within the
declared tolerance. These are BCF pseudomode data, not Hamiltonian bath sites:
`ComplexPoles` is deliberately not an `AbstractHamiltonianBath` and has no
Hamiltonian mounting or `realize_bath` method.
"""
struct ComplexPoles{D<:NamedTuple} <: AbstractBCFParametrization
    layout::FlavorLayout
    partition::Partition
    poles::Vector{ComplexF64}
    weights::Vector{Matrix{ComplexF64}}
    block_indices::Vector{Int}
    channel::Symbol
    diagnostics::D

    function ComplexPoles(layout::FlavorLayout, partition::Partition,
                          poles::Vector{ComplexF64},
                          weights::Vector{Matrix{ComplexF64}},
                          block_indices::Vector{Int}, channel::Symbol,
                          diagnostics::D, ::Val{:validated}) where {D<:NamedTuple}
        new{D}(layout, partition, poles, weights, block_indices, channel,
               diagnostics)
    end
end

function ComplexPoles(layout::FlavorLayout, partition::Partition,
                      poles::AbstractVector{<:Number},
                      weights::AbstractVector{<:AbstractMatrix},
                      block_indices::AbstractVector{<:Integer};
                      channel::Symbol,
                      diagnostics::NamedTuple=(;),
                      stability_tolerance::Real=0.0)
    validate_partition(partition, layout)
    length(poles) == length(weights) == length(block_indices) ||
        throw(DimensionMismatch(
            "ComplexPoles needs one matrix weight and block index per complex pole",
        ))
    isempty(String(channel)) &&
        throw(ArgumentError("ComplexPoles channel must be a nonempty Symbol"))
    tolerance = Float64(stability_tolerance)
    isfinite(tolerance) && tolerance >= 0 ||
        throw(ArgumentError("ComplexPoles stability_tolerance must be finite and nonnegative"))
    exponents = ComplexF64.(poles)
    all(value -> isfinite(real(value)) && isfinite(imag(value)), exponents) ||
        throw(ArgumentError("ComplexPoles exponents must be finite"))
    all(value -> real(value) >= -tolerance, exponents) ||
        throw(ArgumentError(
            "ComplexPoles requires nonnegative damping; unstable BCF exponents are invalid",
        ))
    blocks = Int.(block_indices)
    all(index -> 1 <= index <= length(block_names(partition)), blocks) ||
        throw(ArgumentError("ComplexPoles block indices must reference the Partition"))
    matrices = Matrix{ComplexF64}[]
    for (index, (weight, block_index_value)) in enumerate(zip(weights, blocks))
        block = block_names(partition)[block_index_value]
        dimension = length(block_flavors(partition, block))
        size(weight) == (dimension, dimension) ||
            throw(DimensionMismatch(
                "ComplexPoles weight $index must be $dimension by $dimension for block $block",
            ))
        matrix = Matrix{ComplexF64}(weight)
        all(value -> isfinite(real(value)) && isfinite(imag(value)), matrix) ||
            throw(ArgumentError("ComplexPoles weights must be finite"))
        push!(matrices, matrix)
    end
    return ComplexPoles(layout, partition, exponents, matrices, blocks,
                        channel, diagnostics, Val(:validated))
end

Base.length(poles::ComplexPoles) = length(poles.poles)
Base.isempty(poles::ComplexPoles) = isempty(poles.poles)

function _complex_bcf_times(times::AbstractVector{<:Real})
    values = Float64.(times)
    all(isfinite, values) || throw(ArgumentError("BCF evaluation times must be finite"))
    all(value -> value >= 0, values) ||
        throw(ArgumentError("BCF evaluation times must be nonnegative"))
    return values
end

"""
    evaluate_bcf(poles, times, block)

Evaluate full matrix BCF values for one named block under the canonical
`exp(-z*t)` convention. This is a value operation only; it does not create a
Hamiltonian bath or a Lindbladian realization.
"""
function evaluate_bcf(poles::ComplexPoles, times::AbstractVector{<:Real},
                      block::Symbol)
    block_index_value = block_index(poles.partition, block)
    values = _complex_bcf_times(times)
    dimension = length(block_flavors(poles.partition, block))
    relevant = findall(==(block_index_value), poles.block_indices)
    result = Matrix{ComplexF64}[]
    for time in values
        value = zeros(ComplexF64, dimension, dimension)
        for index in relevant
            value .+= exp(-poles.poles[index] * time) .* poles.weights[index]
        end
        push!(result, value)
    end
    return result
end

function evaluate_bcf(poles::ComplexPoles, time::Real, block::Symbol)
    return only(evaluate_bcf(poles, Float64[time], block))
end

function evaluate_bcf(poles::ComplexPoles, times::AbstractVector{<:Real})
    names = block_names(poles.partition)
    length(names) == 1 || throw(ArgumentError(
        "evaluate_bcf without a block is defined only for a single-block ComplexPoles value",
    ))
    return evaluate_bcf(poles, times, only(names))
end

function evaluate_bcf(poles::ComplexPoles, time::Real)
    return only(evaluate_bcf(poles, Float64[time]))
end
