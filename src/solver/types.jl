"""
    ZeroTemperature()

Typed temperature marker for ground-state real-time and complex-time requests.
It deliberately carries no `beta_eff`: finite-temperature data belongs only to
[`FiniteTemperature`](@ref) imaginary-time requests.
"""
abstract type AbstractSolverTemperature end

struct ZeroTemperature <: AbstractSolverTemperature end

"""
    FiniteTemperature(beta_eff)

Immutable finite-temperature context for an imaginary-time request. `beta_eff`
is the physical inverse temperature used by Graft's purification driver.
"""
struct FiniteTemperature <: AbstractSolverTemperature
    beta_eff::Float64

    function FiniteTemperature(beta_eff::Real)
        value = Float64(beta_eff)
        isfinite(value) && value > 0 || throw(ArgumentError(
            "FiniteTemperature beta_eff must be finite and positive",
        ))
        new(value)
    end
end

"""
    GroundStateRequest(; trunc=TruncationScheme(), nsweeps=10,
                       tolerance=1e-10, krylovdim=20, verbose=false)

Concrete ground-state stage for [`SolveRequest`](@ref). The current production
path uses Graft's adaptive two-site DMRG, so its truncation policy is an
explicit request value rather than hidden solver state.
"""
struct GroundStateRequest{T<:TruncationScheme}
    trunc::T
    nsweeps::Int
    tolerance::Float64
    krylovdim::Int
    verbose::Bool

    function GroundStateRequest(trunc::T, nsweeps::Int, tolerance::Float64,
                                krylovdim::Int, verbose::Bool,
                                ::Val{:validated}) where {T<:TruncationScheme}
        new{T}(trunc, nsweeps, tolerance, krylovdim, verbose)
    end
end

function GroundStateRequest(; trunc::T=TruncationScheme(),
                            nsweeps::Integer=10,
                            tolerance::Real=1e-10,
                            krylovdim::Integer=20,
                            verbose::Bool=false) where {T<:TruncationScheme}
    sweeps = Int(nsweeps)
    dimension = Int(krylovdim)
    tol = Float64(tolerance)
    sweeps > 0 || throw(ArgumentError("GroundStateRequest nsweeps must be positive"))
    dimension >= 2 || throw(ArgumentError(
        "GroundStateRequest krylovdim must be at least two",
    ))
    isfinite(tol) && tol >= 0 || throw(ArgumentError(
        "GroundStateRequest tolerance must be finite and nonnegative",
    ))
    return GroundStateRequest(trunc, sweeps, tol, dimension, verbose,
                              Val(:validated))
end

function _solver_time_grid(values::AbstractVector{<:Real}, name::AbstractString)
    isempty(values) && throw(ArgumentError("$name needs at least one time"))
    grid = Float64.(values)
    all(isfinite, grid) || throw(ArgumentError("$name values must be finite"))
    all(value -> value >= 0, grid) || throw(ArgumentError(
        "$name values must be nonnegative",
    ))
    issorted(grid) || throw(ArgumentError("$name values must be nondecreasing"))
    return grid
end

"""
    RealTimeRequest(times; evolver, temperature=ZeroTemperature())

Raw zero-temperature real-time correlator request. `times` are physical
nonnegative times; the solver delegates their `-im*dt` steps to Graft's
real-time correlator driver.
"""
struct RealTimeRequest{E<:Evolver}
    temperature::ZeroTemperature
    times::Vector{Float64}
    evolver::E

    function RealTimeRequest(temperature::ZeroTemperature, times::Vector{Float64},
                             evolver::E, ::Val{:validated}) where {E<:Evolver}
        new{E}(temperature, times, evolver)
    end
end

function RealTimeRequest(times::AbstractVector{<:Real}; evolver::E,
                         temperature::ZeroTemperature=ZeroTemperature()) where {
                             E<:Evolver}
    return RealTimeRequest(temperature, _solver_time_grid(times, "RealTimeRequest"),
                           evolver, Val(:validated))
end

"""
    ImaginaryTimeRequest(taus, temperature; evolver, thermal_nsteps=40,
                         propagation_nsteps=thermal_nsteps)

Finite-temperature imaginary-time correlator request. Its `beta_eff` lives in
the typed `FiniteTemperature` value and is never reused by real- or
complex-time routes.
"""
struct ImaginaryTimeRequest{E<:Evolver}
    temperature::FiniteTemperature
    taus::Vector{Float64}
    evolver::E
    thermal_nsteps::Int
    propagation_nsteps::Int

    function ImaginaryTimeRequest(temperature::FiniteTemperature,
                                  taus::Vector{Float64}, evolver::E,
                                  thermal_nsteps::Int,
                                  propagation_nsteps::Int,
                                  ::Val{:validated}) where {E<:Evolver}
        new{E}(temperature, taus, evolver, thermal_nsteps, propagation_nsteps)
    end
end

function ImaginaryTimeRequest(taus::AbstractVector{<:Real},
                              temperature::FiniteTemperature;
                              evolver::E,
                              thermal_nsteps::Integer=40,
                              propagation_nsteps::Integer=thermal_nsteps) where {
                                  E<:Evolver}
    grid = _solver_time_grid(taus, "ImaginaryTimeRequest")
    all(tau -> tau <= temperature.beta_eff, grid) || throw(ArgumentError(
        "ImaginaryTimeRequest taus must lie in [0, beta_eff]",
    ))
    prep_steps = Int(thermal_nsteps)
    prop_steps = Int(propagation_nsteps)
    prep_steps > 0 || throw(ArgumentError(
        "ImaginaryTimeRequest thermal_nsteps must be positive",
    ))
    prop_steps > 0 || throw(ArgumentError(
        "ImaginaryTimeRequest propagation_nsteps must be positive",
    ))
    return ImaginaryTimeRequest(temperature, grid, evolver, prep_steps, prop_steps,
                                Val(:validated))
end

"""
    ComplexTimeSegment(dz, steps; label=:segment)

One explicit, repeated core-evolution step in a piecewise-linear complex
contour. `dz` has Graft's `step!` convention (`exp(dz * H)`), not an inferred
physical-time sign convention. A request stores these values verbatim so
parallel, tilted, and kink contours remain reproducible offline.
"""
struct ComplexTimeSegment
    label::Symbol
    dz::ComplexF64
    steps::Int

    function ComplexTimeSegment(label::Symbol, dz::ComplexF64, steps::Int,
                                ::Val{:validated})
        new(label, dz, steps)
    end
end

function ComplexTimeSegment(dz::Number, steps::Integer; label::Symbol=:segment)
    isempty(String(label)) && throw(ArgumentError(
        "ComplexTimeSegment label must be nonempty",
    ))
    value = ComplexF64(dz)
    isfinite(real(value)) && isfinite(imag(value)) && !iszero(value) ||
        throw(ArgumentError("ComplexTimeSegment dz must be finite and nonzero"))
    count = Int(steps)
    count > 0 || throw(ArgumentError("ComplexTimeSegment steps must be positive"))
    return ComplexTimeSegment(label, value, count, Val(:validated))
end

"""
    ComplexTimeRequest(segments; evolver, temperature=ZeroTemperature())

Zero-temperature arbitrary-contour request. The solver expands the explicit
segment list into a labelled complex `z_grid`, calls core `step!` once per
`dz`, and returns raw correlators only.
"""
struct ComplexTimeRequest{E<:Evolver,S<:Tuple}
    temperature::ZeroTemperature
    segments::S
    evolver::E

    function ComplexTimeRequest(temperature::ZeroTemperature, segments::S,
                                evolver::E, ::Val{:validated}) where {
                                    E<:Evolver,S<:Tuple}
        new{E,S}(temperature, segments, evolver)
    end
end

function ComplexTimeRequest(segments::Union{Tuple,AbstractVector}; evolver::E,
                            temperature::ZeroTemperature=ZeroTemperature()) where {
                                E<:Evolver}
    canonical = Tuple(segments)
    isempty(canonical) && throw(ArgumentError(
        "ComplexTimeRequest needs at least one contour segment",
    ))
    all(segment -> segment isa ComplexTimeSegment, canonical) || throw(ArgumentError(
        "ComplexTimeRequest segments must be ComplexTimeSegment values",
    ))
    return ComplexTimeRequest(temperature, canonical, evolver, Val(:validated))
end

ComplexTimeRequest(segment::ComplexTimeSegment; kwargs...) =
    ComplexTimeRequest((segment,); kwargs...)

function _complex_contour_grid(request::ComplexTimeRequest)
    grid = ComplexF64[0.0 + 0.0im]
    labels = Symbol[:initial]
    z = 0.0 + 0.0im
    for segment in request.segments
        for _ in 1:segment.steps
            z += segment.dz
            push!(grid, z)
            push!(labels, segment.label)
        end
    end
    return grid, labels
end

function _complex_request_needs_general_steps(request::ComplexTimeRequest)
    return any(segment -> !isreal(segment.dz) || real(segment.dz) > 0,
               request.segments)
end

function _validated_local_insertion(value, name::AbstractString)
    value isa Pair || throw(ArgumentError(
        "$name must be a `site::Symbol => op::AbstractTensorMap` pair",
    ))
    site = value.first
    op = value.second
    site isa Symbol || throw(ArgumentError("$name site must be a Symbol"))
    op isa AbstractTensorMap || throw(ArgumentError(
        "$name operator must be an AbstractTensorMap",
    ))
    return site, op
end

"""
    LocalObservable(name, insertion)

One named local expectation value in a [`SolveRequest`](@ref). It is separate
from a two-insertion [`LocalCorrelator`](@ref), so results cannot mislabel a
raw correlator as an assembled Green function.
"""
struct LocalObservable{O<:AbstractTensorMap}
    name::Symbol
    site::Symbol
    op::O

    function LocalObservable(name::Symbol, site::Symbol, op::O,
                             ::Val{:validated}) where {O<:AbstractTensorMap}
        new{O}(name, site, op)
    end
end

function LocalObservable(name::Symbol, insertion)
    isempty(String(name)) && throw(ArgumentError("LocalObservable name must be nonempty"))
    site, op = _validated_local_insertion(insertion, "LocalObservable insertion")
    return LocalObservable(name, site, op, Val(:validated))
end

"""
    LocalCorrelator(name, left, right)

One named ordered local two-point channel. `left` and `right` are explicit
`site => operator` insertions; no bath arm, flavor, or operator ordering is
inferred from the channel name.
"""
struct LocalCorrelator{A<:AbstractTensorMap,B<:AbstractTensorMap}
    name::Symbol
    left_site::Symbol
    left::A
    right_site::Symbol
    right::B

    function LocalCorrelator(name::Symbol, left_site::Symbol, left::A,
                             right_site::Symbol, right::B,
                             ::Val{:validated}) where {
                                 A<:AbstractTensorMap,B<:AbstractTensorMap}
        new{A,B}(name, left_site, left, right_site, right)
    end
end

function LocalCorrelator(name::Symbol, left, right)
    isempty(String(name)) && throw(ArgumentError("LocalCorrelator name must be nonempty"))
    left_site, left_op = _validated_local_insertion(left, "LocalCorrelator left")
    right_site, right_op = _validated_local_insertion(right, "LocalCorrelator right")
    return LocalCorrelator(name, left_site, left_op, right_site, right_op,
                           Val(:validated))
end

"""
    RawCorrelator(name, contour, z_grid, values; metadata=(;))

Unassembled correlator data. The fixed `convention=:raw_correlator` prevents
this value from being presented as a retarded/Matsubara Green function or a
self-energy before a separate post-processing stage.
"""
struct RawCorrelator{M<:NamedTuple}
    name::Symbol
    contour::Symbol
    z_grid::Vector{ComplexF64}
    values::Vector{ComplexF64}
    convention::Symbol
    metadata::M

    function RawCorrelator(name::Symbol, contour::Symbol,
                           z_grid::Vector{ComplexF64}, values::Vector{ComplexF64},
                           convention::Symbol, metadata::M,
                           ::Val{:validated}) where {M<:NamedTuple}
        convention === :raw_correlator || throw(ArgumentError(
            "RawCorrelator convention is fixed to :raw_correlator",
        ))
        new{M}(name, contour, z_grid, values, convention, metadata)
    end
end

function RawCorrelator(name::Symbol, contour::Symbol,
                       z_grid::AbstractVector{<:Number},
                       values::AbstractVector{<:Number};
                       metadata::NamedTuple=(;))
    isempty(String(name)) && throw(ArgumentError("RawCorrelator name must be nonempty"))
    isempty(String(contour)) && throw(ArgumentError(
        "RawCorrelator contour label must be nonempty",
    ))
    # Comprehensions retain the declared element type for empty inputs; ordinary
    # broadcast can otherwise preserve an abstract empty-vector eltype and miss
    # the validated inner constructor below.
    grid = ComplexF64[ComplexF64(value) for value in z_grid]
    samples = ComplexF64[ComplexF64(value) for value in values]
    length(grid) == length(samples) || throw(DimensionMismatch(
        "RawCorrelator needs one value per z-grid point",
    ))
    all(value -> isfinite(real(value)) && isfinite(imag(value)), grid) ||
        throw(ArgumentError("RawCorrelator z-grid values must be finite"))
    all(value -> isfinite(real(value)) && isfinite(imag(value)), samples) ||
        throw(ArgumentError("RawCorrelator values must be finite"))
    return RawCorrelator(name, contour, grid, samples, :raw_correlator, metadata,
                         Val(:validated))
end

"""Ground-state result retained independently from time-domain outputs."""
struct GroundStateResult{S<:TTNS}
    state::S
    energy::Float64
    energies::Vector{Float64}

    function GroundStateResult(state::S, energy::Float64,
                               energies::Vector{Float64},
                               ::Val{:validated}) where {S<:TTNS}
        isfinite(energy) || throw(ArgumentError(
            "GroundStateResult energy must be finite",
        ))
        all(isfinite, energies) || throw(ArgumentError(
            "GroundStateResult sweep energies must be finite",
        ))
        new{S}(state, energy, energies)
    end
end

function GroundStateResult(state::S, energy::Real,
                           energies::AbstractVector{<:Real}) where {S<:TTNS}
    return GroundStateResult(state, Float64(energy), Float64.(energies),
                             Val(:validated))
end

function _require_raw_correlator_values(correlators::NamedTuple,
                                        name::AbstractString)
    all(value -> value isa RawCorrelator, values(correlators)) ||
        throw(ArgumentError("$name must contain only RawCorrelator values"))
    return correlators
end

function _require_observable_values(observables::NamedTuple)
    all(value -> value isa Number && isfinite(real(value)) && isfinite(imag(value)),
        values(observables)) || throw(ArgumentError(
            "ImpurityResult observables must be finite numeric values",
        ))
    return observables
end

"""Finite-temperature purification output and its raw imaginary-time channels."""
struct ImaginaryTimeResult{T<:PurificationTrajectory,C<:NamedTuple}
    temperature::FiniteTemperature
    trajectory::T
    correlators::C

    function ImaginaryTimeResult(temperature::FiniteTemperature, trajectory::T,
                                 correlators::C, ::Val{:validated}) where {
                                     T<:PurificationTrajectory,C<:NamedTuple}
        new{T,C}(temperature, trajectory, correlators)
    end
end

function ImaginaryTimeResult(temperature::FiniteTemperature,
                             trajectory::T, correlators::C) where {
                                 T<:PurificationTrajectory,C<:NamedTuple}
    _require_raw_correlator_values(correlators, "ImaginaryTimeResult correlators")
    return ImaginaryTimeResult(temperature, trajectory, correlators, Val(:validated))
end

abstract type AbstractSolveRequest end

"""
    SolveRequest(; ground_state=GroundStateRequest(), real_time=nothing,
                 imaginary_time=nothing, complex_time=nothing,
                 observables=(), correlators=())

Typed execution request. Every time-domain route consumes the same explicit
local correlator channels; a time route without a channel is rejected rather
than producing an ambiguous empty data product.
"""
struct SolveRequest{G<:GroundStateRequest,R<:Union{Nothing,RealTimeRequest},
                    I<:Union{Nothing,ImaginaryTimeRequest},
                    C<:Union{Nothing,ComplexTimeRequest},O<:Tuple,K<:Tuple} <:
        AbstractSolveRequest
    ground_state::G
    real_time::R
    imaginary_time::I
    complex_time::C
    observables::O
    correlators::K

    function SolveRequest(ground_state::G, real_time::R, imaginary_time::I,
                          complex_time::C, observables::O, correlators::K,
                          ::Val{:validated}) where {
                              G<:GroundStateRequest,
                              R<:Union{Nothing,RealTimeRequest},
                              I<:Union{Nothing,ImaginaryTimeRequest},
                              C<:Union{Nothing,ComplexTimeRequest},O<:Tuple,K<:Tuple}
        new{G,R,I,C,O,K}(ground_state, real_time, imaginary_time, complex_time,
                         observables, correlators)
    end
end

function _validated_named_values(values, expected, name::AbstractString)
    canonical = Tuple(values)
    all(value -> value isa expected, canonical) || throw(ArgumentError(
        "$name values must have type $expected",
    ))
    names = Tuple(value.name for value in canonical)
    allunique(names) || throw(ArgumentError("$name names must be unique"))
    return canonical
end

function SolveRequest(; ground_state::GroundStateRequest=GroundStateRequest(),
                      real_time=nothing, imaginary_time=nothing,
                      complex_time=nothing, observables=(), correlators=())
    real_time === nothing || real_time isa RealTimeRequest || throw(ArgumentError(
        "SolveRequest real_time must be nothing or RealTimeRequest",
    ))
    imaginary_time === nothing || imaginary_time isa ImaginaryTimeRequest ||
        throw(ArgumentError(
            "SolveRequest imaginary_time must be nothing or ImaginaryTimeRequest",
        ))
    complex_time === nothing || complex_time isa ComplexTimeRequest ||
        throw(ArgumentError(
            "SolveRequest complex_time must be nothing or ComplexTimeRequest",
        ))
    observable_values = _validated_named_values(observables, LocalObservable,
                                                 "SolveRequest observables")
    channel_values = _validated_named_values(correlators, LocalCorrelator,
                                              "SolveRequest correlators")
    any_time = real_time !== nothing || imaginary_time !== nothing ||
               complex_time !== nothing
    !any_time || !isempty(channel_values) || throw(ArgumentError(
        "SolveRequest time-domain requests need at least one LocalCorrelator",
    ))
    return SolveRequest(ground_state, real_time, imaginary_time, complex_time,
                        observable_values, channel_values, Val(:validated))
end

abstract type AbstractImpurityResult end

"""
    NonMountableImpurityResult

Typed terminal solver result for a fit that cannot become a Hamiltonian bath.
The original expansion and realization diagnostics remain available; no
diagonal-only fallback or partially mounted Hamiltonian is produced.
"""
struct NonMountableImpurityResult{SI<:BathFitInput,I<:BathFitInput,E<:PoleExpansion,
                                  N<:NonMountablePoleFit,A<:BathFitAudit,
                                  Q<:SolveRequest} <: AbstractImpurityResult
    source_input::SI
    input_kind::Symbol
    h_loc0::ImpurityOneBody
    input::I
    expansion::E
    discretization::N
    bathfit_audit::A
    request::Q

    function NonMountableImpurityResult(source_input::SI, input_kind::Symbol,
                                        h_loc0::ImpurityOneBody, input::I,
                                        expansion::E, discretization::N,
                                        bathfit_audit::A, request::Q,
                                        ::Val{:validated}) where {
                                            SI<:BathFitInput,I<:BathFitInput,
                                            E<:PoleExpansion,N<:NonMountablePoleFit,
                                            A<:BathFitAudit,Q<:SolveRequest}
        new{SI,I,E,N,A,Q}(source_input, input_kind, h_loc0, input, expansion,
                          discretization, bathfit_audit, request)
    end
end

function NonMountableImpurityResult(source_input::BathFitInput, input_kind::Symbol,
                                    h_loc0::ImpurityOneBody, input::BathFitInput,
                                    expansion::PoleExpansion,
                                    discretization::NonMountablePoleFit,
                                    bathfit_audit::BathFitAudit,
                                    request::SolveRequest)
    input_kind in (:weiss, :hybridization) || throw(ArgumentError(
        "NonMountableImpurityResult input_kind must be :weiss or :hybridization",
    ))
    return NonMountableImpurityResult(
        source_input, input_kind, h_loc0, input, expansion, discretization,
        bathfit_audit, request, Val(:validated),
    )
end

"""
    ImpurityResult

Complete successful solver lifecycle output. It keeps input, fit/realization
evidence, mounted bath, lowered TTNO/compression report, ground state, static
observables, and each raw contour-labelled correlator category distinct.
"""
struct ImpurityResult{SI<:BathFitInput,I<:BathFitInput,E<:PoleExpansion,
                      D<:DiscretizationResult,
                      M<:AbstractMountedBath,H<:LoweredImpurityHamiltonian,
                      G<:GroundStateResult,O<:NamedTuple,R<:NamedTuple,
                      IT<:Union{Nothing,ImaginaryTimeResult},C<:NamedTuple,
                      A<:BathFitAudit,Q<:SolveRequest} <:
        AbstractImpurityResult
    source_input::SI
    input_kind::Symbol
    h_loc0::ImpurityOneBody
    input::I
    expansion::E
    discretization::D
    bathfit_audit::A
    mounted::M
    lowered::H
    ground_state::G
    energy::Float64
    observables::O
    real_time::R
    imaginary_time::IT
    complex_time::C
    request::Q
    warm_identity::UInt

    function ImpurityResult(source_input::SI, input_kind::Symbol,
                            h_loc0::ImpurityOneBody, input::I, expansion::E,
                            discretization::D, bathfit_audit::A, mounted::M,
                            lowered::H, ground_state::G, energy::Float64,
                            observables::O, real_time::R, imaginary_time::IT,
                            complex_time::C, request::Q, warm_identity::UInt,
                            ::Val{:validated}) where {
                                SI<:BathFitInput,I<:BathFitInput,E<:PoleExpansion,
                                D<:DiscretizationResult,M<:AbstractMountedBath,
                                H<:LoweredImpurityHamiltonian,G<:GroundStateResult,
                                O<:NamedTuple,R<:NamedTuple,
                                IT<:Union{Nothing,ImaginaryTimeResult},C<:NamedTuple,
                                A<:BathFitAudit,Q<:SolveRequest}
        new{SI,I,E,D,M,H,G,O,R,IT,C,A,Q}(
            source_input, input_kind, h_loc0, input, expansion, discretization,
            bathfit_audit, mounted, lowered, ground_state, energy, observables,
            real_time, imaginary_time, complex_time, request, warm_identity,
        )
    end
end

function ImpurityResult(source_input::BathFitInput, input_kind::Symbol,
                        h_loc0::ImpurityOneBody, input::BathFitInput,
                        expansion::PoleExpansion, discretization::DiscretizationResult,
                        bathfit_audit::BathFitAudit, mounted::AbstractMountedBath,
                        lowered::LoweredImpurityHamiltonian,
                        ground_state::GroundStateResult, energy::Real,
                        observables::NamedTuple, real_time::NamedTuple,
                        imaginary_time::Union{Nothing,ImaginaryTimeResult},
                        complex_time::NamedTuple, request::SolveRequest,
                        warm_identity::UInt)
    input_kind in (:weiss, :hybridization) || throw(ArgumentError(
        "ImpurityResult input_kind must be :weiss or :hybridization",
    ))
    resolved_energy = Float64(energy)
    isfinite(resolved_energy) || throw(ArgumentError(
        "ImpurityResult energy must be finite",
    ))
    _require_observable_values(observables)
    _require_raw_correlator_values(real_time, "ImpurityResult real_time")
    _require_raw_correlator_values(complex_time, "ImpurityResult complex_time")
    return ImpurityResult(
        source_input, input_kind, h_loc0, input, expansion, discretization,
        bathfit_audit, mounted, lowered, ground_state, resolved_energy,
        observables, real_time, imaginary_time, complex_time, request,
        warm_identity, Val(:validated),
    )
end

"""
    Solver(; gf_struct, layout, topology_plan, bath_mapping=nothing, phys=nothing,
           bath_fit_kernel, ops=ImpurityOperators(layout), symmetry=SymmetrySpec(layout),
           soc=nothing, compression_atol=0, scheme=TruncationScheme())

Stateful breaking impurity-solver owner. `gf_struct` is the exact named
[`Partition`](@ref), never an inferred collection of arms. Staged mutable
fields record fit, realization, mounting, lowering, warm-start, and result
state so every input replacement can invalidate them atomically.
"""
mutable struct Solver{L<:FlavorLayout,P<:Partition,K<:AbstractRealPoleBathFitKernel,
                      O<:ImpurityOperators,PH<:NamedTuple,S<:SymmetrySpec,
                      T<:TruncationScheme} <: AbstractImpuritySolver
    gf_struct::P
    layout::L
    topology_plan::Union{Nothing,AbstractImpurityTopologyPlan,TreeTopology}
    bath_mapping::Union{Nothing,AbstractBathMappingKernel}
    phys::PH
    bath_fit_kernel::K
    ops::O
    symmetry::S
    soc::Union{Nothing,ImpurityOneBody}
    compression_atol::Float64
    scheme::T
    input_kind::Symbol
    source_input::Union{Nothing,BathFitInput}
    input::Union{Nothing,BathFitInput}
    h_loc0::Union{Nothing,ImpurityOneBody}
    expansion::Union{Nothing,PoleExpansion}
    discretization::Union{Nothing,DiscretizationResult,NonMountablePoleFit}
    mapping_result::Union{Nothing,CayleyMappingResult}
    mounted::Union{Nothing,AbstractMountedBath}
    interaction::Union{Nothing,AbstractImpurityInteraction}
    lowered::Union{Nothing,LoweredImpurityHamiltonian}
    bathfit_audit::Union{Nothing,BathFitAudit}
    warm_start::Union{Nothing,TTNS}
    warm_identity::Union{Nothing,UInt}
    last_request::Union{Nothing,AbstractSolveRequest}
    last_result::Union{Nothing,AbstractImpurityResult}
end
