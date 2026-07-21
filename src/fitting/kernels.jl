"""
    QuadratureKernel(plan; rule=:trapezoid)

Executable real-axis spectral discretization by direct bin integration. The
kernel owns only its immutable allocation plan and quadrature policy; source
data and output expansions are supplied per call. Its current bin-residue
convention is fermionic only; bosonic real-axis fitting requires a distinct
kernel and is rejected rather than being reinterpreted.
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
This real-axis residue convention is fermionic only; bosonic data is rejected
until a separately defined bosonic real-axis kernel is implemented.
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

"""
    MiniPoleKernel(; n_poles, rank_tolerance=sqrt(eps()),
                   conformal_scale=nothing, holdout_count=0,
                   fit_tolerance=nothing)

Clean-room MiniPole/matrix-ESPRIT configuration shared by two typed routes.
`real_pole_bath_fit` uses `conformal_scale` as a preferred real-Matsubara
mapping scale and accepts only finite real Hamiltonian energies. `fit_complex_bcf`
uses the same stacked exponential engine on a uniform time grid and preserves
stable complex BCF exponents. `fit_tolerance=nothing` records quality without a
hard fit gate; a positive value enforces a relative training-error limit.
"""
struct MiniPoleKernel <: AbstractRealPoleBathFitKernel
    n_poles::Int
    rank_tolerance::Float64
    conformal_scale::Union{Nothing,Float64}
    holdout_count::Int
    fit_tolerance::Union{Nothing,Float64}

    function MiniPoleKernel(n_poles::Int, rank_tolerance::Float64,
                            conformal_scale::Union{Nothing,Float64},
                            holdout_count::Int,
                            fit_tolerance::Union{Nothing,Float64},
                            ::Val{:validated})
        new(n_poles, rank_tolerance, conformal_scale, holdout_count,
            fit_tolerance)
    end
end

function MiniPoleKernel(; n_poles::Integer,
                        rank_tolerance::Real=sqrt(eps(Float64)),
                        conformal_scale::Union{Nothing,Real}=nothing,
                        holdout_count::Integer=0,
                        fit_tolerance::Union{Nothing,Real}=nothing)
    count = Int(n_poles)
    count > 0 || throw(ArgumentError("MiniPoleKernel n_poles must be positive"))
    tolerance = Float64(rank_tolerance)
    isfinite(tolerance) && tolerance > 0 ||
        throw(ArgumentError("MiniPoleKernel rank_tolerance must be finite and positive"))
    scale = conformal_scale === nothing ? nothing : Float64(conformal_scale)
    scale === nothing || (isfinite(scale) && scale > 0) ||
        throw(ArgumentError("MiniPoleKernel conformal_scale must be finite and positive"))
    retained = Int(holdout_count)
    retained >= 0 || throw(ArgumentError("MiniPoleKernel holdout_count must be nonnegative"))
    fit = fit_tolerance === nothing ? nothing : Float64(fit_tolerance)
    fit === nothing || (isfinite(fit) && fit > 0) ||
        throw(ArgumentError("MiniPoleKernel fit_tolerance must be finite and positive"))
    return MiniPoleKernel(count, tolerance, scale, retained, fit, Val(:validated))
end

"""
    ESPRITTauKernel(; n_poles, pole_tolerance=sqrt(eps()),
                    projection_tolerance=1e-12,
                    fit_tolerance=nothing)

Imaginary-time matrix-ESPRIT finite-bath fitter. It independently fits each
named fermionic hybridization block and returns an ordinary `PoleExpansion`.
This imaginary-time ESPRIT route is not an extension point for the
`green-jl-counterterms` reference BFGS branch; that branch belongs to the
direct coupling-fit family represented by
`CouplingFitKernel`. `pole_tolerance` is the relative fail-closed tolerance for
the imaginary part of an ESPRIT pole energy. `projection_tolerance` controls
the explicit per-pole PSD positive-part projection, and `fit_tolerance`, when
provided, is a hard relative-L2 gate applied after that physical projection.
"""
struct ESPRITTauKernel <: AbstractRealPoleBathFitKernel
    n_poles::Int
    pole_tolerance::Float64
    projection_tolerance::Float64
    fit_tolerance::Union{Nothing,Float64}

    function ESPRITTauKernel(n_poles::Int, pole_tolerance::Float64,
                             projection_tolerance::Float64,
                             fit_tolerance::Union{Nothing,Float64},
                             ::Val{:validated})
        new(n_poles, pole_tolerance, projection_tolerance, fit_tolerance)
    end
end

function ESPRITTauKernel(; n_poles::Integer,
                         pole_tolerance::Real=sqrt(eps(Float64)),
                         projection_tolerance::Real=1e-12,
                         fit_tolerance::Union{Nothing,Real}=nothing)
    count = Int(n_poles)
    count > 0 || throw(ArgumentError(
        "ESPRITTauKernel n_poles must be positive",
    ))
    pole = Float64(pole_tolerance)
    isfinite(pole) && pole > 0 || throw(ArgumentError(
        "ESPRITTauKernel pole_tolerance must be finite and positive",
    ))
    projection = Float64(projection_tolerance)
    isfinite(projection) && projection > 0 || throw(ArgumentError(
        "ESPRITTauKernel projection_tolerance must be finite and positive",
    ))
    fit = fit_tolerance === nothing ? nothing : Float64(fit_tolerance)
    fit === nothing || (isfinite(fit) && fit > 0) || throw(ArgumentError(
        "ESPRITTauKernel fit_tolerance must be finite and positive",
    ))
    return ESPRITTauKernel(count, pole, projection, fit, Val(:validated))
end

"""Allocation policy for the signs of direct-fit bath energies."""
abstract type AbstractCouplingModeAllocation end

"""Allow every direct-fit bath energy to vary across its declared bounds."""
struct FreeModeAllocation <: AbstractCouplingModeAllocation end

"""Require exactly `n_negative` direct-fit bath energies to remain below zero."""
struct SignedModeAllocation <: AbstractCouplingModeAllocation
    n_negative::Int

    function SignedModeAllocation(n_negative::Int, ::Val{:validated})
        new(n_negative)
    end
end

function SignedModeAllocation(n_negative::Integer)
    count = Int(n_negative)
    count >= 0 || throw(ArgumentError(
        "SignedModeAllocation n_negative must be nonnegative",
    ))
    return SignedModeAllocation(count, Val(:validated))
end

"""Constraint family for direct-fit complex coupling components."""
abstract type AbstractCouplingComponents end

"""Permit independently optimized real and imaginary coupling components."""
struct ComplexComponents <: AbstractCouplingComponents end

"""Restrict every direct-fit coupling component to the real subspace."""
struct RealComponents <: AbstractCouplingComponents end

"""Relation family for explicitly tied named coupling blocks."""
abstract type AbstractCouplingTieRelation end

"""Derive target couplings by exact equality with the source couplings."""
struct EqualTie <: AbstractCouplingTieRelation end

"""Derive target couplings by exact complex conjugation of source couplings."""
struct ConjugateTie <: AbstractCouplingTieRelation end

const _CouplingTieRelation = Union{EqualTie,ConjugateTie}

"""
    CouplingBlockTie(source, target, relation=EqualTie())

Explicit named-block relation for `CouplingFitKernel`. `target` shares source
mode parameters during fitting and is then derived exactly. The two supported
relation values preserve the direct `V * V'` PSD residue parameterization.
"""
struct CouplingBlockTie{R<:_CouplingTieRelation}
    source::Symbol
    target::Symbol
    relation::R

    function CouplingBlockTie(source::Symbol, target::Symbol, relation::R,
                              ::Val{:validated}) where {R<:_CouplingTieRelation}
        new{R}(source, target, relation)
    end
end

function CouplingBlockTie(source::Symbol, target::Symbol,
                          relation::R) where {R<:_CouplingTieRelation}
    source != target ||
        throw(ArgumentError("CouplingBlockTie source and target must differ"))
    return CouplingBlockTie(source, target, relation, Val(:validated))
end

CouplingBlockTie(source::Symbol, target::Symbol) =
    CouplingBlockTie(source, target, EqualTie())

function CouplingBlockTie(source::Symbol, target::Symbol,
                          relation::AbstractCouplingTieRelation)
    throw(ArgumentError(
        "CouplingBlockTie relation must be EqualTie() or ConjugateTie()",
    ))
end

function _coupling_bounds(value, name::AbstractString)
    value === nothing && return nothing
    value isa Tuple && length(value) == 2 ||
        throw(ArgumentError("CouplingFitKernel $name must be a two-element tuple"))
    lower, upper = Float64.(value)
    span = upper - lower
    isfinite(lower) && isfinite(upper) && isfinite(span) && span > 0 ||
        throw(ArgumentError("CouplingFitKernel $name must be finite and ordered"))
    return (lower, upper)
end

function _coupling_frequency_window(value)
    bounds = _coupling_bounds(value, "frequency_window")
    bounds === nothing && return nothing
    bounds[1] > 0 || throw(ArgumentError(
        "CouplingFitKernel frequency_window must be strictly positive",
    ))
    return bounds
end

"""
    CouplingFitKernel(; n_modes, alpha=1.0, frequency_window=nothing,
                      energy_bounds=nothing, maxiter=1_000,
                      optimizer_tolerance=sqrt(eps()), fit_tolerance=nothing,
                      allocation=FreeModeAllocation(),
                      components=ComplexComponents(), block_ties=())

Direct fermionic Matsubara coupling-space fit configuration. Every
independently fitted named block receives `n_modes` real bath energies and
complex coupling vectors; its raw residues are exactly `V * V'`. `alpha`
weights the squared Frobenius objective by `abs(omega)^(-alpha)`. An optional
`SignedModeAllocation` keeps its declared number of modes below zero per
independent block; `CouplingBlockTie` values derive compatible named blocks
explicitly. `RealComponents()` restricts coupling components to the real
subspace, while `ComplexComponents()` retains fully complex off-diagonal
couplings. Only `EqualTie()` and `ConjugateTie()` are accepted. The cited paper
prints a weighted Frobenius norm; this kernel uses the corresponding smooth
weighted least-squares objective so its nonlinear optimizer has one
deterministic scalar objective to minimize. A finite nonconverged optimization
remains explicitly labelled `:nonconverged` in the trace; complex per-mode
global phases are parameterization redundancies, while the returned `V * V'`
residues are phase-invariant. Set `fit_tolerance` to make each independently
fitted or tied named block's relative reconstruction threshold a hard
input-domain acceptance gate. Declared energy bounds are closed: each feasible
declared endpoint is refined in its coupling components and retained only when
it strictly improves the weighted objective; the trace records every snap.

The `green-jl-counterterms` reference BFGS branch maps to the same direct
coupling model and objective family through, for example,
`CouplingFitKernel(n_modes=N, alpha=0.0, components=RealComponents(),
energy_bounds=(emin, emax))`; it is not a variant of `ESPRITTauKernel`.
This is not a literal port: the reference implementation initializes randomly
and uses `lambda_range` only to initialize energies, which are unconstrained
thereafter. This kernel instead uses deterministic moment initialization and
treats `energy_bounds` as a true closed feasible interval throughout the fit.
"""
struct CouplingFitKernel{A<:AbstractCouplingModeAllocation,
                         C<:AbstractCouplingComponents,T<:Tuple} <:
        AbstractRealPoleBathFitKernel
    n_modes::Int
    alpha::Float64
    frequency_window::Union{Nothing,Tuple{Float64,Float64}}
    energy_bounds::Union{Nothing,Tuple{Float64,Float64}}
    maxiter::Int
    optimizer_tolerance::Float64
    fit_tolerance::Union{Nothing,Float64}
    allocation::A
    components::C
    block_ties::T

    function CouplingFitKernel(n_modes::Int, alpha::Float64,
                               frequency_window::Union{Nothing,Tuple{Float64,Float64}},
                               energy_bounds::Union{Nothing,Tuple{Float64,Float64}},
                               maxiter::Int, optimizer_tolerance::Float64,
                               fit_tolerance::Union{Nothing,Float64}, allocation::A,
                               components::C, block_ties::T,
                               ::Val{:validated}) where {A<:AbstractCouplingModeAllocation,
                                                         C<:AbstractCouplingComponents,T<:Tuple}
        new{A,C,T}(n_modes, alpha, frequency_window, energy_bounds, maxiter,
                   optimizer_tolerance, fit_tolerance, allocation, components,
                   block_ties)
    end
end

function CouplingFitKernel(; n_modes::Integer,
                           alpha::Real=1.0,
                           frequency_window=nothing,
                           energy_bounds=nothing,
                           maxiter::Integer=1_000,
                           optimizer_tolerance::Real=sqrt(eps(Float64)),
                           fit_tolerance::Union{Nothing,Real}=nothing,
                           allocation::AbstractCouplingModeAllocation=FreeModeAllocation(),
                           components::AbstractCouplingComponents=ComplexComponents(),
                           block_ties=())
    modes = Int(n_modes)
    modes > 0 || throw(ArgumentError("CouplingFitKernel n_modes must be positive"))
    exponent = Float64(alpha)
    isfinite(exponent) && 0 <= exponent <= 1 ||
        throw(ArgumentError("CouplingFitKernel alpha must be finite and lie in [0, 1]"))
    iterations = Int(maxiter)
    iterations > 0 || throw(ArgumentError("CouplingFitKernel maxiter must be positive"))
    tolerance = Float64(optimizer_tolerance)
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError(
        "CouplingFitKernel optimizer_tolerance must be finite and positive",
    ))
    quality = fit_tolerance === nothing ? nothing : Float64(fit_tolerance)
    quality === nothing || (isfinite(quality) && quality > 0) ||
        throw(ArgumentError("CouplingFitKernel fit_tolerance must be finite and positive"))
    if allocation isa SignedModeAllocation
        allocation.n_negative <= modes || throw(ArgumentError(
            "SignedModeAllocation n_negative must not exceed CouplingFitKernel n_modes",
        ))
    elseif !(allocation isa FreeModeAllocation)
        throw(ArgumentError("CouplingFitKernel allocation is unsupported"))
    end
    components isa Union{ComplexComponents,RealComponents} || throw(ArgumentError(
        "CouplingFitKernel component constraint is unsupported",
    ))
    ties = Tuple(block_ties)
    all(tie -> tie isa CouplingBlockTie, ties) || throw(ArgumentError(
        "CouplingFitKernel block_ties must contain CouplingBlockTie values",
    ))
    all(tie -> tie.relation isa _CouplingTieRelation, ties) || throw(ArgumentError(
        "CouplingFitKernel block_ties use an unsupported relation",
    ))
    targets = Symbol[tie.target for tie in ties]
    allunique(targets) ||
        throw(ArgumentError("CouplingFitKernel block-tie targets must be unique"))
    sources = Set(tie.source for tie in ties)
    all(target -> !(target in sources), targets) || throw(ArgumentError(
        "CouplingFitKernel does not permit chained block ties",
    ))
    return CouplingFitKernel(
        modes, exponent, _coupling_frequency_window(frequency_window),
        _coupling_bounds(energy_bounds, "energy_bounds"), iterations, tolerance,
        quality, allocation, components, ties, Val(:validated),
    )
end
