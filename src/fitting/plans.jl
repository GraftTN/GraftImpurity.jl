"""
    SpectralInterval(lower, upper, modes; forced_poles=[])

One closed support interval in a real-axis discretization plan. Gaps are
represented by separate intervals. `modes` counts pole bins, not the expanded
number of physical bath orbitals produced by a matrix residue factorization.
"""
struct SpectralInterval
    lower::Float64
    upper::Float64
    modes::Int
    forced_poles::Tuple{Vararg{Float64}}

    function SpectralInterval(lower::Float64, upper::Float64, modes::Int,
                              forced_poles::Tuple{Vararg{Float64}},
                              ::Val{:validated})
        new(lower, upper, modes, forced_poles)
    end
end

function SpectralInterval(lower::Real, upper::Real, modes::Integer;
                          forced_poles::AbstractVector{<:Real}=Float64[])
    lo, hi = Float64(lower), Float64(upper)
    isfinite(lo) && isfinite(hi) && lo < hi ||
        throw(ArgumentError("SpectralInterval bounds must be finite and ordered"))
    count = Int(modes)
    count > 0 || throw(ArgumentError("SpectralInterval needs at least one mode"))
    points = Tuple(sort(Float64.(forced_poles)))
    all(point -> isfinite(point) && lo <= point <= hi, points) ||
        throw(ArgumentError("forced poles must lie in their SpectralInterval"))
    allunique(points) || throw(ArgumentError("forced poles must be unique"))
    length(points) <= count ||
        throw(ArgumentError("each forced pole consumes one interval mode"))
    return SpectralInterval(lo, hi, count, points, Val(:validated))
end

"""
    BlockDiscretizationPlan(intervals; outer_bounds=nothing, discarded_weight=0)

Stable real-axis allocation data for one named hybridization block. Empty
intervals are valid only for manually supplied pole expansions, where no
quadrature/boundary allocation was used.
"""
struct BlockDiscretizationPlan
    outer_bounds::Union{Nothing,Tuple{Float64,Float64}}
    intervals::Tuple{Vararg{SpectralInterval}}
    discarded_weight::Float64
    weight_measure::Symbol

    function BlockDiscretizationPlan(
        outer_bounds::Union{Nothing,Tuple{Float64,Float64}},
        intervals::Tuple{Vararg{SpectralInterval}}, discarded_weight::Float64,
        weight_measure::Symbol,
        ::Val{:validated},
    )
        new(outer_bounds, intervals, discarded_weight, weight_measure)
    end
end

function BlockDiscretizationPlan(
    intervals::AbstractVector{<:SpectralInterval}=SpectralInterval[];
    outer_bounds::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    discarded_weight::Real=0,
    weight_measure::Symbol=:unspecified,
)
    discarded = Float64(discarded_weight)
    isfinite(discarded) && discarded >= 0 ||
        throw(ArgumentError("discarded spectral weight must be finite and nonnegative"))
    isempty(String(weight_measure)) &&
        throw(ArgumentError("spectral weight measure must be named"))
    canonical = Tuple(SpectralInterval[intervals...])
    issorted(canonical; by=interval -> interval.lower) ||
        throw(ArgumentError("spectral intervals must be ordered"))
    all(canonical[index].upper <= canonical[index + 1].lower
        for index in 1:max(length(canonical) - 1, 0)) ||
        throw(ArgumentError("spectral intervals may not overlap"))
    bounds = if outer_bounds === nothing
        isempty(canonical) ? nothing :
            (first(canonical).lower, last(canonical).upper)
    else
        lower, upper = Float64(outer_bounds[1]), Float64(outer_bounds[2])
        isfinite(lower) && isfinite(upper) && lower <= upper ||
            throw(ArgumentError("outer bounds must be finite and ordered"))
        all(interval -> lower <= interval.lower && interval.upper <= upper,
            canonical) ||
            throw(ArgumentError("spectral intervals must lie within outer bounds"))
        (lower, upper)
    end
    return BlockDiscretizationPlan(
        bounds, canonical, discarded, weight_measure, Val(:validated),
    )
end

"""
    BlockDiscretizationPlan(outer_bounds, supports, modes;
                            forced_poles=[], discarded_weight=0)

Construct a support/gap-aware block plan with an exact total pole-bin budget.
Every support interval receives at least one bin and every forced pole reserves
one; the remaining bins are allocated by interval length using deterministic
largest-remainder rounding. A gap is represented by the absence of a support
interval rather than by a hidden zero-weight bin.
"""
function BlockDiscretizationPlan(
    outer_bounds::Tuple{<:Real,<:Real}, supports::AbstractVector,
    modes::Integer; forced_poles::AbstractVector{<:Real}=Float64[],
    discarded_weight::Real=0,
    weight_measure::Symbol=:unspecified,
)
    isempty(supports) && throw(ArgumentError("support intervals may not be empty"))
    canonical_supports = Tuple(begin
        support isa Tuple && length(support) == 2 ||
            throw(ArgumentError("each support interval must be a two-tuple"))
        lower, upper = Float64(support[1]), Float64(support[2])
        isfinite(lower) && isfinite(upper) && lower < upper ||
            throw(ArgumentError("support bounds must be finite and ordered"))
        (lower, upper)
    end for support in supports)
    issorted(canonical_supports; by=first) ||
        throw(ArgumentError("support intervals must be ordered"))
    all(canonical_supports[index][2] <= canonical_supports[index + 1][1]
        for index in 1:max(length(canonical_supports) - 1, 0)) ||
        throw(ArgumentError("support intervals may not overlap"))

    lower, upper = Float64(outer_bounds[1]), Float64(outer_bounds[2])
    isfinite(lower) && isfinite(upper) && lower <= upper ||
        throw(ArgumentError("outer bounds must be finite and ordered"))
    all(interval -> lower <= interval[1] && interval[2] <= upper,
        canonical_supports) ||
        throw(ArgumentError("support intervals must lie within outer bounds"))

    forced = Tuple(sort(Float64.(forced_poles)))
    allunique(forced) || throw(ArgumentError("forced poles must be unique"))
    all(point -> isfinite(point), forced) ||
        throw(ArgumentError("forced poles must be finite"))
    assignments = [Int[] for _ in canonical_supports]
    for (point_index, point) in enumerate(forced)
        interval_index = findfirst(
            interval -> interval[1] <= point <= interval[2], canonical_supports,
        )
        interval_index === nothing &&
            throw(ArgumentError("forced pole $point lies outside the declared support"))
        push!(assignments[interval_index], point_index)
    end

    budget = Int(modes)
    minimum_counts = [max(1, length(indices)) for indices in assignments]
    sum(minimum_counts) <= budget ||
        throw(ArgumentError(
            "mode budget must provide at least one bin per support and forced pole",
        ))
    remaining = budget - sum(minimum_counts)
    lengths = Float64[interval[2] - interval[1] for interval in canonical_supports]
    total_length = sum(lengths)
    ideal = remaining .* lengths ./ total_length
    extra = floor.(Int, ideal)
    leftover = remaining - sum(extra)
    ordering = sortperm(eachindex(ideal); by=index -> (-rem(ideal[index], 1), index))
    for index in ordering[1:leftover]
        extra[index] += 1
    end
    intervals = SpectralInterval[
        SpectralInterval(
            interval[1], interval[2], minimum_counts[index] + extra[index];
            forced_poles=Float64[forced[point] for point in assignments[index]],
        ) for (index, interval) in enumerate(canonical_supports)
    ]
    return BlockDiscretizationPlan(
        intervals; outer_bounds=(lower, upper), discarded_weight, weight_measure,
    )
end

function _planning_psd_sample(sample::Matrix{ComplexF64})
    tolerance = sqrt(eps(Float64)) * max(opnorm(sample), 1.0)
    hermitian = (sample + adjoint(sample)) / 2
    norm(sample - adjoint(sample)) <= tolerance || return false
    return minimum(real.(eigvals(Hermitian(hermitian)))) >= -tolerance
end

function _spectral_weight_profile(frequencies::AbstractVector{<:Real}, samples)
    length(frequencies) >= 2 ||
        throw(ArgumentError("automatic spectral planning needs at least two frequencies"))
    values = Float64.(frequencies)
    all(isfinite, values) && allunique(values) ||
        throw(ArgumentError("automatic spectral planning needs distinct finite frequencies"))
    matrices = _fit_sample_matrices(samples, length(values))
    permutation = sortperm(values)
    sorted_frequencies = values[permutation]
    sorted_samples = matrices[permutation]
    if all(_planning_psd_sample, sorted_samples)
        trace_weights = Float64[
            max(real(sum(diag((sample + adjoint(sample)) / 2))), 0.0)
            for sample in sorted_samples
        ]
        return sorted_frequencies, trace_weights, :hermitian_trace
    end
    return sorted_frequencies, Float64[norm(sample) for sample in sorted_samples],
           :frobenius_norm
end

function _spectral_support_weight(frequencies::Vector{Float64},
                                  weights::Vector{Float64}, supports)
    total = _integrate_spectral_weights(
        frequencies, weights, first(frequencies), last(frequencies),
    )
    retained = sum(_integrate_spectral_weights(
        frequencies, weights, Float64(support[1]), Float64(support[2]),
    ) for support in supports)
    return total, retained
end

function _spectral_weight_at(frequencies::Vector{Float64}, weights::Vector{Float64},
                             point::Float64)
    frequencies[1] <= point <= frequencies[end] ||
        throw(ArgumentError("spectral boundary lies outside the supplied frequency mesh"))
    right = searchsortedfirst(frequencies, point)
    if right == 1 || frequencies[right] == point
        return weights[right]
    elseif right > length(frequencies)
        return weights[end]
    end
    left = right - 1
    fraction = (point - frequencies[left]) /
               (frequencies[right] - frequencies[left])
    return (1 - fraction) * weights[left] + fraction * weights[right]
end

function _integrate_spectral_weights(frequencies::Vector{Float64},
                                     weights::Vector{Float64},
                                     lower::Float64, upper::Float64)
    lower < upper || return 0.0
    interior = frequencies[(lower .< frequencies) .& (frequencies .< upper)]
    nodes = vcat(lower, interior, upper)
    values = [_spectral_weight_at(frequencies, weights, point) for point in nodes]
    return sum((nodes[index + 1] - nodes[index]) *
               (values[index] + values[index + 1]) / 2
               for index in 1:(length(nodes) - 1))
end

function _automatic_outer_bounds(frequencies::Vector{Float64},
                                 weights::Vector{Float64},
                                 discarded_fraction::Float64)
    total = _integrate_spectral_weights(
        frequencies, weights, first(frequencies), last(frequencies),
    )
    total > 0 || return (first(frequencies), last(frequencies)), 0.0, total
    tail_budget = discarded_fraction * total / 2
    lower_index = firstindex(frequencies)
    lower_discarded = 0.0
    while lower_index < lastindex(frequencies)
        panel = (frequencies[lower_index + 1] - frequencies[lower_index]) *
                (weights[lower_index] + weights[lower_index + 1]) / 2
        lower_discarded + panel <= tail_budget || break
        lower_discarded += panel
        lower_index += 1
    end
    upper_index = lastindex(frequencies)
    upper_discarded = 0.0
    while lower_index < upper_index
        panel = (frequencies[upper_index] - frequencies[upper_index - 1]) *
                (weights[upper_index - 1] + weights[upper_index]) / 2
        upper_discarded + panel <= tail_budget || break
        upper_discarded += panel
        upper_index -= 1
    end
    bounds = (frequencies[lower_index], frequencies[upper_index])
    retained = _integrate_spectral_weights(frequencies, weights, bounds...)
    return bounds, max(total - retained, 0.0), total
end

"""
    BlockDiscretizationPlan(frequencies, spectral_samples, modes;
                            discarded_fraction=0, supports=nothing,
                            forced_poles=[])

Build one real-axis block plan by selecting the smallest deterministic outer
window obtained by removing no more than `discarded_fraction` of a documented
positive spectral measure. Hermitian trace weight is used only when every
sample is numerically Hermitian PSD; otherwise a full-matrix Frobenius measure
keeps off-diagonal, non-Hermitian, and indefinite data visible to the later
Hamiltonian-realizability gate. The resulting `discarded_weight` is the
measured support-union area, not the requested fraction.
"""
function BlockDiscretizationPlan(
    frequencies::AbstractVector{<:Real}, spectral_samples, modes::Integer;
    discarded_fraction::Real=0,
    supports::Union{Nothing,AbstractVector}=nothing,
    forced_poles::AbstractVector{<:Real}=Float64[],
)
    fraction = Float64(discarded_fraction)
    isfinite(fraction) && 0 <= fraction < 1 ||
        throw(ArgumentError("discarded_fraction must be finite and in [0, 1)"))
    values = Float64.(frequencies)
    sorted_frequencies, weights, measure = _spectral_weight_profile(
        values, spectral_samples,
    )
    bounds, _, total = _automatic_outer_bounds(
        sorted_frequencies, weights, fraction,
    )
    declared_supports = supports === nothing ?
        [(bounds[1], bounds[2])] : supports
    isempty(declared_supports) &&
        throw(ArgumentError("automatic spectral planning needs at least one support"))
    all(support -> support isa Tuple && length(support) == 2,
        declared_supports) ||
        throw(ArgumentError("automatic-plan supports must be two-tuples"))
    support_bounds = Tuple((Float64(support[1]), Float64(support[2]))
                           for support in declared_supports)
    all(support -> first(sorted_frequencies) <= support[1] < support[2] <=
                   last(sorted_frequencies),
        support_bounds) ||
        throw(ArgumentError("automatic-plan supports must lie within the supplied mesh"))
    forced = Float64.(forced_poles)
    all(point -> isfinite(point) && first(sorted_frequencies) <= point <=
                 last(sorted_frequencies), forced) ||
        throw(ArgumentError("automatic-plan forced poles must lie within the supplied mesh"))
    lower = minimum(first, support_bounds)
    upper = maximum(last, support_bounds)
    _, retained = _spectral_support_weight(
        sorted_frequencies, weights, support_bounds,
    )
    discarded = max(total - retained, 0.0)
    discarded <= fraction * total + 100 * eps(Float64) * total ||
        throw(ArgumentError(
            "declared supports discard more spectral weight than discarded_fraction allows",
        ))
    return BlockDiscretizationPlan(
        (lower, upper), collect(support_bounds), modes;
        forced_poles=forced, discarded_weight=discarded,
        weight_measure=measure,
    )
end

"""
    DiscretizationPlan(:block => block_plan, ...; shared_grid)

Immutable allocation/ownership evidence emitted by real-axis fitting kernels.
`shared_grid=true` means every matrix component of a named block uses its one
declared bin grid; it never means that unrelated named blocks are merged. A
multi-flavor named block is an explicitly coupled matrix block and therefore
requires this shared-grid contract even when a particular sample happens to be
diagonal. Independent diagonal grids are represented by distinct singleton
named `Partition` blocks, matching the declared `gf_struct` ownership rather
than dynamically inspecting data to change topology semantics.
"""
struct DiscretizationPlan{B<:NamedTuple}
    blocks::B
    shared_grid::Bool

    function DiscretizationPlan(blocks::B, shared_grid::Bool,
                                ::Val{:validated}) where {B<:NamedTuple}
        new{B}(blocks, shared_grid)
    end
end

function DiscretizationPlan(blocks::Pair...; shared_grid::Bool=false)
    isempty(blocks) && throw(ArgumentError("DiscretizationPlan needs named blocks"))
    names = Symbol[first(block) for block in blocks]
    allunique(names) || throw(ArgumentError("DiscretizationPlan block names must be unique"))
    values = Tuple(begin
        block.second isa BlockDiscretizationPlan ||
            throw(ArgumentError("DiscretizationPlan blocks must be BlockDiscretizationPlan values"))
        block.second
    end for block in blocks)
    return DiscretizationPlan(NamedTuple{Tuple(names)}(values), shared_grid,
                              Val(:validated))
end

function DiscretizationPlan(partition::Partition; shared_grid::Bool=false)
    return DiscretizationPlan(
        (name => BlockDiscretizationPlan() for name in block_names(partition))...;
        shared_grid,
    )
end

"""Plan record for one named Partition block."""
function plan_block(plan::DiscretizationPlan, block::Symbol)
    hasproperty(plan.blocks, block) || throw(KeyError(block))
    return getproperty(plan.blocks, block)
end
