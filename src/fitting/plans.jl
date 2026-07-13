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
    points = Tuple(Float64.(forced_poles))
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

    function BlockDiscretizationPlan(
        outer_bounds::Union{Nothing,Tuple{Float64,Float64}},
        intervals::Tuple{Vararg{SpectralInterval}}, discarded_weight::Float64,
        ::Val{:validated},
    )
        new(outer_bounds, intervals, discarded_weight)
    end
end

function BlockDiscretizationPlan(
    intervals::AbstractVector{<:SpectralInterval}=SpectralInterval[];
    outer_bounds::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    discarded_weight::Real=0,
)
    discarded = Float64(discarded_weight)
    isfinite(discarded) && discarded >= 0 ||
        throw(ArgumentError("discarded spectral weight must be finite and nonnegative"))
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
    return BlockDiscretizationPlan(bounds, canonical, discarded, Val(:validated))
end

"""
    DiscretizationPlan(:block => block_plan, ...; shared_grid)

Immutable allocation/ownership evidence emitted by real-axis fitting kernels.
`shared_grid=true` means every matrix component of a named block uses its one
declared bin grid; it never means that unrelated named blocks are merged.
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
