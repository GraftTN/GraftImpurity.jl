"""
    QuadratureKernel(plan; rule=:trapezoid)

Executable real-axis spectral discretization by direct bin integration. The
kernel owns only its immutable allocation plan and quadrature policy; source
data and output expansions are supplied per call.
"""
struct QuadratureKernel{P<:DiscretizationPlan} <: AbstractRealPoleBathFitKernel
    plan::P
    rule::Symbol

    function QuadratureKernel(plan::P, rule::Symbol, ::Val{:validated}) where {P<:DiscretizationPlan}
        new{P}(plan, rule)
    end
end

function QuadratureKernel(plan::DiscretizationPlan; rule::Symbol=:trapezoid)
    rule === :trapezoid ||
        throw(ArgumentError("QuadratureKernel currently supports rule=:trapezoid"))
    return QuadratureKernel(plan, rule, Val(:validated))
end

"""
    BoundaryFitKernel(plan; broadening, scan_scales=(1.0,), residue_solver=:auto,
                      orbital_order=nothing, selection_policy=:prefer_mountable)

Real-axis boundary-scan fit kernel. Each candidate owns a rescaled version of
the declared support plan and keeps all matrix entries on one shared pole grid
per named block. Scales are noncontracting outer-boundary expansion factors so
declared support gaps and exact forced-pole positions remain valid.
`residue_solver=:auto` uses direct bin integration for spectral input and
complex linear least squares for retarded input. An explicit immutable
`orbital_order` is carried into both preflight and default realization.
"""
struct BoundaryFitKernel{P<:DiscretizationPlan,O} <: AbstractRealPoleBathFitKernel
    plan::P
    broadening::Float64
    scan_scales::Tuple{Vararg{Float64}}
    residue_solver::Symbol
    orbital_order::O
    selection_policy::Symbol

    function BoundaryFitKernel(plan::P, broadening::Float64,
                               scan_scales::Tuple{Vararg{Float64}},
                               residue_solver::Symbol, orbital_order::O,
                               selection_policy::Symbol,
                               ::Val{:validated}) where {P<:DiscretizationPlan,O}
        new{P,O}(plan, broadening, scan_scales, residue_solver, orbital_order,
                 selection_policy)
    end
end

function _boundary_orbital_order(order)
    order === nothing && return nothing
    order isa NamedTuple ||
        throw(ArgumentError("BoundaryFitKernel orbital_order must be a NamedTuple"))
    names = keys(order)
    canonical_values = Tuple(begin
        declared isa AbstractVector ||
            throw(ArgumentError("each BoundaryFitKernel orbital order must be a vector"))
        Tuple(Symbol.(declared))
    end for declared in values(order))
    return NamedTuple{names}(canonical_values)
end

function _validate_boundary_plan(plan::DiscretizationPlan)
    for block in keys(plan.blocks)
        block_plan = plan_block(plan, block)
        isempty(block_plan.intervals) &&
            throw(ArgumentError("BoundaryFitKernel needs support intervals for block $block"))
        outer = something(
            block_plan.outer_bounds,
            (first(block_plan.intervals).lower, last(block_plan.intervals).upper),
        )
        outer == (first(block_plan.intervals).lower,
                  last(block_plan.intervals).upper) ||
            throw(ArgumentError(
                "BoundaryFitKernel requires outer bounds to equal the declared support hull for block $block",
            ))
    end
    return plan
end

function BoundaryFitKernel(plan::DiscretizationPlan;
                           broadening::Real,
                           scan_scales=[1.0],
                           residue_solver::Symbol=:auto,
                           orbital_order=nothing,
                           selection_policy::Symbol=:prefer_mountable)
    eta = Float64(broadening)
    isfinite(eta) && eta > 0 ||
        throw(ArgumentError("BoundaryFitKernel broadening must be finite and positive"))
    scales = Tuple(Float64.(collect(scan_scales)))
    isempty(scales) && throw(ArgumentError("BoundaryFitKernel needs scan scales"))
    all(scale -> isfinite(scale) && scale >= 1, scales) ||
        throw(ArgumentError(
            "BoundaryFitKernel scan scales must be finite noncontracting factors at least one",
        ))
    allunique(scales) ||
        throw(ArgumentError("BoundaryFitKernel scan scales must be unique"))
    residue_solver in (:auto, :bin_integral, :least_squares) ||
        throw(ArgumentError(
            "BoundaryFitKernel residue_solver must be :auto, :bin_integral, or :least_squares",
        ))
    selection_policy === :prefer_mountable ||
        throw(ArgumentError(
            "BoundaryFitKernel currently supports selection_policy=:prefer_mountable",
        ))
    _validate_boundary_plan(plan)
    return BoundaryFitKernel(
        plan, eta, scales, residue_solver, _boundary_orbital_order(orbital_order),
        selection_policy, Val(:validated),
    )
end

"""
    PESKernel(; tolerance=nothing, n_poles=nothing, solver=:sdp, ...)

Kernel-owned adapter to the independent PES/AAA real-pole algorithm. Exactly
one stopping policy is required; the kernel retains only numerical options and
never stores a Green function or fitted expansion.
"""
struct PESKernel <: AbstractRealPoleBathFitKernel
    tolerance::Union{Nothing,Float64}
    n_poles::Union{Nothing,Int}
    solver::Symbol
    maxiter::Int
    min_support::Int
    max_support::Int
    aaa_tolerance::Float64
    residue_tolerance::Float64
    conic_diagnostic::Symbol

    function PESKernel(tolerance::Union{Nothing,Float64},
                       n_poles::Union{Nothing,Int}, solver::Symbol,
                       maxiter::Int, min_support::Int, max_support::Int,
                       aaa_tolerance::Float64, residue_tolerance::Float64,
                       conic_diagnostic::Symbol, ::Val{:validated})
        new(tolerance, n_poles, solver, maxiter, min_support, max_support,
            aaa_tolerance, residue_tolerance, conic_diagnostic)
    end
end

function PESKernel(; tolerance::Union{Nothing,Real}=nothing,
                   n_poles::Union{Nothing,Integer}=nothing,
                   solver::Symbol=:sdp,
                   maxiter::Integer=0,
                   min_support::Integer=4,
                   max_support::Integer=50,
                   aaa_tolerance::Real=1e-13,
                   residue_tolerance::Real=1e-5,
                   conic_diagnostic::Symbol=:none)
    xor(tolerance === nothing, n_poles === nothing) ||
        throw(ArgumentError("PESKernel needs exactly one of tolerance or n_poles"))
    resolved_tolerance = tolerance === nothing ? nothing : Float64(tolerance)
    resolved_poles = n_poles === nothing ? nothing : Int(n_poles)
    resolved_tolerance === nothing ||
        (isfinite(resolved_tolerance) && resolved_tolerance > 0) ||
        throw(ArgumentError("PESKernel tolerance must be finite and positive"))
    resolved_poles === nothing || resolved_poles > 0 ||
        throw(ArgumentError("PESKernel n_poles must be positive"))
    solver in (:sdp, :least_squares, :lstsq) ||
        throw(ArgumentError("PESKernel solver must be :sdp or :least_squares"))
    maxiter >= 0 || throw(ArgumentError("PESKernel maxiter must be nonnegative"))
    min_support >= 2 ||
        throw(ArgumentError("PESKernel min_support must be at least two"))
    max_support >= min_support ||
        throw(ArgumentError("PESKernel max_support must cover min_support"))
    aaa = Float64(aaa_tolerance)
    residue = Float64(residue_tolerance)
    isfinite(aaa) && aaa > 0 ||
        throw(ArgumentError("PESKernel aaa_tolerance must be finite and positive"))
    isfinite(residue) && residue >= 0 ||
        throw(ArgumentError("PESKernel residue_tolerance must be finite and nonnegative"))
    conic_diagnostic in (:none, :distance) ||
        throw(ArgumentError("PESKernel conic_diagnostic must be :none or :distance"))
    return PESKernel(resolved_tolerance, resolved_poles, solver, Int(maxiter),
                     Int(min_support), Int(max_support), aaa, residue,
                     conic_diagnostic, Val(:validated))
end
