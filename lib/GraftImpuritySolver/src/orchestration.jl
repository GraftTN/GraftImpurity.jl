function _solver_physical_manifest(operators::ImpurityOperators)
    names = layout_sites(operators.layout)
    return NamedTuple{names}(Tuple(
        site_operators(operators, site).P for site in names
    ))
end

function _validate_solver_topology(topology_plan, bath_mapping)
    if bath_mapping === nothing
        topology_plan === nothing && throw(ArgumentError(
            "TTNSSolver needs an explicit T3NS, FTPS, or custom TreeTopology when " *
            "bath_mapping is nothing",
        ))
        topology_plan isa Union{T3NS,FTPS,TreeTopology} || throw(ArgumentError(
            "TTNSSolver topology_plan must be T3NS, FTPS, or TreeTopology",
        ))
    else
        bath_mapping isa CayleyTreeKernel || throw(ArgumentError(
            "TTNSSolver currently accepts only CayleyTreeKernel bath mappings",
        ))
        topology_plan === nothing || throw(ArgumentError(
            "TTNSSolver bath_mapping and topology_plan are mutually exclusive",
        ))
    end
    return nothing
end

function _validate_solver_problem_topology(solver::TTNSSolver,
                                           problem::ImpurityProblem)
    layout = bath_layout(problem.bath)
    plan = solver.topology_plan
    if plan isa AbstractImpurityTopologyPlan
        plan.layout == layout || throw(ArgumentError(
            "TTNSSolver topology plan FlavorLayout must match ImpurityProblem",
        ))
    elseif plan isa TreeTopology
        _validate_impurity_nodes(plan, layout)
    end
    return nothing
end

"""
    TTNSSolver(; topology_plan=nothing, bath_mapping=nothing,
               carrier=ParticleNumberSector(),
               ttno_builder=LegacyTTNOBuilder(), compression_atol=0,
               scheme=TruncationScheme())

Construct immutable TTNS backend policy. The physical problem and every mutable
execution product are supplied separately to `solve!` through an
[`ImpurityProblem`](@ref) and [`TTNSWorkspace`](@ref).
"""
function TTNSSolver(; topology_plan=nothing, bath_mapping=nothing,
                    carrier::C=ParticleNumberSector(),
                    ttno_builder::B=LegacyTTNOBuilder(),
                    compression_atol::Real=0.0,
                    scheme::T=TruncationScheme()) where {
                        C<:AbstractFermionSector,B<:AbstractTTNOBuilder,
                        T<:TruncationScheme}
    _validate_solver_topology(topology_plan, bath_mapping)
    tolerance = Float64(compression_atol)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError(
        "TTNSSolver compression_atol must be finite and nonnegative",
    ))
    return TTNSSolver(topology_plan, bath_mapping, carrier, ttno_builder,
                      tolerance, scheme)
end

function _invalidate_workspace_build!(workspace::TTNSWorkspace)
    workspace.ops = nothing
    workspace.phys = nothing
    workspace.mapping_result = nothing
    workspace.mounted = nothing
    workspace.lowered = nothing
    workspace.warm_start = nothing
    workspace.warm_identity = nothing
    workspace.last_request = nothing
    workspace.last_result = nothing
    return workspace
end

function _solver_mount_bath!(workspace::TTNSWorkspace, solver::TTNSSolver,
                             bath::DiscreteBath)
    if solver.bath_mapping !== nothing
        mapped = map_bath(solver.bath_mapping, bath)
        workspace.mapping_result = mapped
        return mount_bath(mapped; sector=solver.carrier)
    end
    plan = solver.topology_plan
    if plan isa Union{T3NS,FTPS}
        return mount_bath(
            impurity_topology(plan, bath_partition(bath), bath), bath;
            sector=solver.carrier,
        )
    end
    plan isa TreeTopology || throw(ArgumentError(
        "TTNSSolver has no mountable topology plan",
    ))
    return mount_bath(plan, bath; sector=solver.carrier)
end

function _solver_materialize_build!(workspace::TTNSWorkspace,
                                    solver::TTNSSolver,
                                    problem::ImpurityProblem,
                                    symmetry::SymmetrySpec)
    operators = ImpurityOperators(problem_layout(problem); sector=solver.carrier)
    workspace.ops = operators
    workspace.phys = _solver_physical_manifest(operators)
    mounted = _solver_mount_bath!(workspace, solver, problem.bath)
    lowered = lower_hamiltonian(
        mounted, problem.interaction, operators;
        h_loc=problem.h_loc, soc=nothing, symmetry,
        ttno_builder=solver.ttno_builder,
        compression_atol=solver.compression_atol, scheme=solver.scheme,
    )
    workspace.mounted = mounted
    workspace.lowered = lowered
    return mounted, lowered
end

function _solver_problem_identity(problem::ImpurityProblem)
    return problem_identity(problem)
end

function _solver_policy_identity(solver::TTNSSolver)
    return hash((:GraftImpurityTTNSPolicyIdentity, solver.topology_plan,
                 solver.bath_mapping, solver.carrier, solver.ttno_builder,
                 solver.compression_atol, solver.scheme))
end

_solver_manifold_identity(request::TTNSSolveRequest) =
    manifold_identity(request.manifold)

function _solver_request_identity(request::TTNSSolveRequest)
    return hash((:GraftImpurityTTNSRequestIdentity, request.manifold,
                 request.ground_state, request.real_time, request.imaginary_time,
                 request.complex_time, request.observables, request.correlators))
end

function _solver_warm_identity(problem_identity::UInt, policy_identity::UInt,
                               manifold_identity::UInt, request_identity::UInt,
                               mounted::AbstractMountedBath)
    bath_hash = _mounted_bath_integrity_hash(mounted)
    return hash((:GraftImpuritySolverWarmStart, problem_identity, policy_identity,
                 manifold_identity, request_identity,
                 mounted.topology, mounted.diagnostics.ownership_hash, bath_hash,
                 ))
end

function _solver_disposition_actions(problem::ImpurityProblem,
                                     solver::TTNSSolver)
    actions = symmetry_actions(problem.symmetry)
    canonical = action_identity(ChargeU1(problem_layout(problem)))
    unsupported = Tuple(
        action_identity(action) for action in actions
        if !(action isa ChargeU1 && action_identity(action) == canonical)
    )
    if length(actions) != 1 || !isempty(unsupported)
        throw(TTNSCapabilityError(
            :symmetry_declaration,
            "TTNSSolver v2 requires exactly the certified ChargeU1 physical " *
            "action; every additional, differently parameterized, audit-only, " *
            "or non-Abelian declaration is unsupported",
        ))
    end
    solver.carrier isa ParticleNumberSector || throw(TTNSCapabilityError(
        :charge_u1_carrier,
        "TTNSSolver ChargeU1 execution requires ParticleNumberSector() carrier policy",
    ))
    return only(actions), canonical
end

function _solver_validate_target(problem::ImpurityProblem,
                                 target::TargetIrrep,
                                 canonical_action)
    target.action_identity == canonical_action || throw(TTNSCapabilityError(
        :target_action,
        "TTNSSolver target must use the exact certified ChargeU1 action, basis, " *
        "category product, and physical generator semantics",
    ))
    target.target isa U1Irrep || throw(TTNSCapabilityError(
        :target_irrep,
        "TTNSSolver certified ChargeU1 targets must use Graft U1Irrep values",
    ))
    particle_number = try
        Int(target.target.charge)
    catch
        throw(TTNSCapabilityError(
            :target_irrep,
            "TTNSSolver ChargeU1 target must carry an integer particle number",
        ))
    end
    capacity = length(flavors(problem_layout(problem))) + length(problem.bath)
    0 <= particle_number <= capacity || throw(TTNSCapabilityError(
        :target_bounds,
        "TTNSSolver ChargeU1 target particle number must lie in 0:$capacity",
    ))
    return target
end

struct _TTNSChargeU1ResponseBackend{P}
    physical::P
end

function _response_failure(reason::Symbol, message::String)
    throw(ResponseReachabilityError(:ttns_charge_u1, reason, message))
end

function _response_siteop(backend::_TTNSChargeU1ResponseBackend,
                          site::Symbol, operator::AbstractTensorMap,
                          side::Symbol)
    site in propertynames(backend.physical) || _response_failure(
        :unknown_site,
        "response $side insertion site $site is absent from the impurity basis",
    )
    expected = getproperty(backend.physical, site)
    codomain(operator)[1] == expected || _response_failure(
        :physical_space,
        "response $side operator physical output does not match site $site",
    )
    try
        return SiteOp(site, Symbol(:response_, side), operator)
    catch error
        _response_failure(
            :operator_flux,
            "response $side operator has no certified scalar tensor flux: " *
            sprint(showerror, error),
        )
    end
end

function validate_response_reachability(
        backend::_TTNSChargeU1ResponseBackend,
        source::TargetIrrep, response::TargetIrrep,
        right::Pair{Symbol,<:AbstractTensorMap},
        left::Pair{Symbol,<:AbstractTensorMap})
    validate_response_target(source, response)
    source.target isa U1Irrep || _response_failure(
        :source_target, "TTNS ChargeU1 response source must be a U1Irrep",
    )
    response.target isa U1Irrep || _response_failure(
        :response_target, "TTNS ChargeU1 response target must be a U1Irrep",
    )
    right_flux = charge(_response_siteop(
        backend, right.first, right.second, :right,
    ))
    left_flux = charge(_response_siteop(
        backend, left.first, left.second, :left,
    ))
    typeof(left_flux) == typeof(right_flux) && left_flux == dual(right_flux) ||
        _response_failure(
            :closure_flux,
            "left and right response operators must carry opposite full tensor flux",
        )
    right_charge = try
        right_flux[2]
    catch
        _response_failure(
            :category_product,
            "TTNS ChargeU1 response operators must carry parity × U1 tensor flux",
        )
    end
    left_charge = left_flux[2]
    right_charge isa U1Irrep && left_charge isa U1Irrep || _response_failure(
        :category_product,
        "TTNS ChargeU1 response operators must carry U1Irrep as their second flux factor",
    )
    expected_response = U1Irrep(
        Int(source.target.charge) + Int(right_charge.charge),
    )
    expected_source = U1Irrep(
        Int(response.target.charge) + Int(left_charge.charge),
    )
    response.target == expected_response || _response_failure(
        :right_reachability,
        "right operator tensor flux does not move the source into the explicit response target",
    )
    source.target == expected_source || _response_failure(
        :left_reachability,
        "left operator tensor flux does not close the response target back to the source",
    )
    return response
end

function _solver_response_backend(problem::ImpurityProblem,
                                  solver::TTNSSolver)
    operators = ImpurityOperators(problem_layout(problem); sector=solver.carrier)
    return _TTNSChargeU1ResponseBackend(_solver_physical_manifest(operators))
end

function _solver_validate_observables(backend::_TTNSChargeU1ResponseBackend,
                                      request::TTNSSolveRequest)
    for observable in request.observables
        flux = charge(_response_siteop(
            backend, observable.site, observable.op, :observable,
        ))
        flux == one(typeof(flux)) || throw(TTNSCapabilityError(
            :charged_observable,
            "TTNSSolver observables must have neutral full tensor flux",
        ))
    end
    return request
end

function _solver_preflight(problem::ImpurityProblem, solver::TTNSSolver,
                           request::TTNSSolveRequest)
    action, canonical = _solver_disposition_actions(problem, solver)
    manifold = request.manifold
    if manifold isa TargetIrrep
        _solver_validate_target(problem, manifold, canonical)
    elseif manifold isa IrrepScan
        for outer_target in manifold.targets
            _solver_validate_target(
                problem, TargetIrrep(manifold.action_identity, outer_target),
                canonical,
            )
        end
        any_time = request.real_time !== nothing ||
                   request.imaginary_time !== nothing ||
                   request.complex_time !== nothing
        (!any_time && isempty(request.correlators)) || throw(TTNSCapabilityError(
            :scan_response,
            "TTNSSolver IrrepScan currently supports ground-state and neutral " *
            "observable results only; multi-sector response/time results have " *
            "no certified contract",
        ))
    else
        throw(TTNSCapabilityError(
            :manifold, "TTNSSolver does not support this impurity manifold type",
        ))
    end

    if !isempty(request.observables) || !isempty(request.correlators)
        backend = _solver_response_backend(problem, solver)
        _solver_validate_observables(backend, request)
        if manifold isa TargetIrrep
            for channel in request.correlators
                _solver_validate_target(
                    problem, channel.response_target, canonical,
                )
                validate_response_reachability(
                    backend, manifold, channel.response_target,
                    channel.right_site => channel.right,
                    channel.left_site => channel.left,
                )
            end
        end
    end
    return SymmetrySpec(problem_layout(problem); abelian=(action,))
end

function _solver_target_sector(target::TargetIrrep)
    particle_number = Int(target.target.charge)
    return FermionParity(mod(particle_number, 2)) ⊠ target.target
end

function _state_requires_complex_eltype(request::TTNSSolveRequest)
    return request.real_time !== nothing || request.complex_time !== nothing
end

function _validate_solver_state(state::TTNS, lowered::LoweredImpurityHamiltonian,
                                request::TTNSSolveRequest)
    topology(state) == topology(lowered.operator) || throw(ArgumentError(
        "initial_state topology does not match the mounted impurity Hamiltonian",
    ))
    state.hasphys == lowered.operator.hasphys || throw(ArgumentError(
        "initial_state physical layout does not match the mounted impurity Hamiltonian",
    ))
    spacetype(state) == spacetype(lowered.operator) ||
        throw(ArgumentError(
            "initial_state symmetry space does not match the mounted impurity Hamiltonian",
        ))
    eltype(state) == eltype(lowered.operator) || throw(ArgumentError(
        "initial_state scalar type $(eltype(state)) does not match Hamiltonian " *
        "scalar type $(eltype(lowered.operator))",
    ))
    for site in propertynames(lowered.mounted.phys)
        node = nodeindex(state.topo, site)
        state.hasphys[node] || throw(ArgumentError(
            "initial_state is missing the physical leg at mounted site $site",
        ))
        physspace(state, node) == getproperty(lowered.mounted.phys, site) ||
            throw(ArgumentError(
                "initial_state physical space at $site does not match the mounted " *
                "impurity Hamiltonian",
            ))
    end
    root_space = domain(state.tensors[state.topo.root])[1]
    request.manifold isa TargetIrrep || throw(ArgumentError(
        "internal TTNS state validation requires a fixed TargetIrrep request",
    ))
    collect(sectors(root_space)) == [_solver_target_sector(request.manifold)] ||
        throw(ArgumentError(
            "initial_state root sector does not match TTNSSolveRequest manifold target",
        ))
    !_state_requires_complex_eltype(request) || eltype(state) <: Complex ||
        throw(ArgumentError(
            "real/complex-time TTNSSolver requests require a complex-eltype initial_state",
        ))
    return state
end

function _solver_initial_state(workspace::TTNSWorkspace,
                               lowered::LoweredImpurityHamiltonian,
                               identity::UInt, request::TTNSSolveRequest,
                               initial_state;
                               warm_start=workspace.warm_start,
                               warm_identity=workspace.warm_identity)
    if initial_state !== nothing
        initial_state isa TTNS || throw(ArgumentError(
            "solve! initial_state must be a Graft.TTNS",
        ))
        return copy(_validate_solver_state(initial_state, lowered, request))
    end
    warm_start === nothing && throw(ArgumentError(
        "solve! needs initial_state because no warm start is available",
    ))
    warm_identity == identity || throw(ArgumentError(
        "solve! warm start is invalid for the current layout, bath ownership, " *
        "one-body, interaction, or topology identity; supply initial_state",
    ))
    return copy(_validate_solver_state(warm_start, lowered, request))
end

function _solver_namedtuple(channels::Tuple, values)
    names = Tuple(channel.name for channel in channels)
    return NamedTuple{names}(Tuple(values))
end

function _solver_observables(state::TTNS, observables::Tuple)
    values = ComplexF64[
        ComplexF64(expect(state, observable.op, observable.site))
        for observable in observables
    ]
    return _solver_namedtuple(observables, values)
end

function _solver_real_time(state::TTNS, energy::Float64,
                           lowered::LoweredImpurityHamiltonian,
                           request::RealTimeRequest, channels::Tuple)
    values = RawCorrelator[]
    for channel in channels
        series = correlator_series(
            state, energy, channel.left_site => channel.left,
            channel.right_site => channel.right, request.times;
            H=lowered.operator, evolver=request.evolver,
            metadata=(; temperature=:zero, contour=:real_time,
                       channel=channel.name),
        )
        push!(values, RawCorrelator(
            channel.name, :real_time, -im .* ComplexF64.(request.times), series.values;
            metadata=merge(series.metadata, (; coordinate=:core_step,
                                             physical_times=copy(request.times))),
        ))
    end
    return _solver_namedtuple(channels, values)
end

function _fresh_solver_evolver(evolver::Evolver)
    fresh = deepcopy(evolver)
    if hasproperty(fresh, :cache)
        setproperty!(fresh, :cache, nothing)
    end
    return fresh
end

function _solver_complex_time(state::TTNS, energy::Float64,
                              lowered::LoweredImpurityHamiltonian,
                              request::ComplexTimeRequest, channels::Tuple)
    _require_complex_contour_evolver(request)
    grid, labels = _complex_contour_grid(request)
    values = RawCorrelator[]
    for channel in channels
        bra = apply_local(state, adjoint(channel.left), channel.left_site)
        ket = apply_local(state, channel.right, channel.right_site)
        samples = Vector{ComplexF64}(undef, length(grid))
        samples[1] = ComplexF64(inner(bra, ket))
        evolver = _fresh_solver_evolver(request.evolver)
        position = 1
        z = grid[1]
        for segment in request.segments
            for _ in 1:segment.steps
                z += segment.dz
                position += 1
                step!(evolver, ket, lowered.operator, segment.dz)
                samples[position] = ComplexF64(exp(-energy * z) * inner(bra, ket))
            end
        end
        push!(values, RawCorrelator(
            channel.name, :complex_time, grid, samples;
            metadata=(; temperature=:zero, contour_segments=request.segments,
                       segment_labels=Tuple(labels), channel=channel.name,
                       evolver_type=typeof(request.evolver),
                       step_convention=:exp_dz_H),
        ))
    end
    return _solver_namedtuple(channels, values)
end

function _solver_imaginary_time(lowered::LoweredImpurityHamiltonian,
                                request::ImaginaryTimeRequest, channels::Tuple)
    beta = request.temperature.beta_eff
    physical = _mounted_physical_spaces(lowered.mounted)
    problem = purification_problem(lowered.opsum, lowered.mounted.topology,
                                  physical; hermitian=true)
    save_betas = sort!(unique!(vcat(beta .- request.taus, [beta])))
    trajectory = thermalize(
        Purified(), problem, beta;
        evolver=request.evolver, nsteps=request.thermal_nsteps,
        save_betas,
    )
    values = RawCorrelator[]
    for channel in channels
        series = thermal_correlator(
            Purified(), problem, channel.left_site => channel.left,
            channel.right_site => channel.right, beta, request.taus;
            evolver=request.evolver, trajectory,
            prop_nsteps=request.propagation_nsteps,
            metadata=(; contour=:imaginary_time, channel=channel.name),
        )
        push!(values, RawCorrelator(
            channel.name, :imaginary_time, ComplexF64.(request.taus), series.values;
            metadata=merge(series.metadata, (; coordinate=:tau)),
        ))
    end
    return ImaginaryTimeResult(request.temperature, trajectory,
                               _solver_namedtuple(channels, values))
end

"""
    solve!(workspace, solver, problem, request; initial_state=nothing)

Run the TTNS backend lifecycle `topology/mount -> lower -> DMRG -> requested
correlators` for an already prepared finite [`ImpurityProblem`](@ref). Fitting,
source conversion, and non-mountable preparation failures terminate upstream
and never enter this method.
"""
function _solve_fixed!(workspace::TTNSWorkspace, solver::TTNSSolver,
                       problem::ImpurityProblem, request::TTNSSolveRequest;
                       initial_state=nothing)
    _validate_solve_request_contract(request)
    problem_statistics(problem) === :fermion || throw(ArgumentError(
        "TTNSSolver currently supports fermionic ImpurityProblem values only",
    ))
    request.manifold isa TargetIrrep || throw(ArgumentError(
        "internal fixed TTNS execution requires TargetIrrep",
    ))
    symmetry = _solver_preflight(problem, solver, request)
    _validate_solver_problem_topology(solver, problem)

    current_problem_identity = _solver_problem_identity(problem)
    current_policy_identity = _solver_policy_identity(solver)
    current_manifold_identity = _solver_manifold_identity(request)
    current_request_identity = _solver_request_identity(request)

    build_matches = workspace.problem_identity == current_problem_identity &&
                    workspace.policy_identity == current_policy_identity &&
                    workspace.mounted !== nothing && workspace.lowered !== nothing
    if !build_matches
        _invalidate_workspace_build!(workspace)
    elseif workspace.manifold_identity != current_manifold_identity ||
           workspace.request_identity != current_request_identity
        workspace.warm_start = nothing
        workspace.warm_identity = nothing
        workspace.last_request = nothing
        workspace.last_result = nothing
    end

    workspace.problem_identity = current_problem_identity
    workspace.policy_identity = current_policy_identity
    workspace.manifold_identity = current_manifold_identity
    workspace.request_identity = current_request_identity

    if !build_matches
        _solver_materialize_build!(workspace, solver, problem, symmetry)
    end

    mounted = workspace.mounted::AbstractMountedBath
    lowered = workspace.lowered::LoweredImpurityHamiltonian
    identity = _solver_warm_identity(
        current_problem_identity, current_policy_identity,
        current_manifold_identity, current_request_identity, mounted,
    )
    state = _solver_initial_state(
        workspace, lowered, identity, request, initial_state,
    )
    state, energies = dmrg2!(
        state, lowered.operator;
        trunc=request.ground_state.trunc,
        nsweeps=request.ground_state.nsweeps,
        tol=request.ground_state.tolerance,
        krylovdim=request.ground_state.krylovdim,
        verbose=request.ground_state.verbose,
    )
    normalize!(state)
    energy = Float64(real(expect(state, lowered.operator)))
    ground_state = GroundStateResult(state, energy, Float64.(energies))
    observables = _solver_observables(state, request.observables)
    real_time = request.real_time === nothing ? NamedTuple() :
        _solver_real_time(state, energy, lowered, request.real_time,
                          request.correlators)
    imaginary_time = request.imaginary_time === nothing ? nothing :
        _solver_imaginary_time(lowered, request.imaginary_time, request.correlators)
    complex_time = request.complex_time === nothing ? NamedTuple() :
        _solver_complex_time(state, energy, lowered, request.complex_time,
                             request.correlators)

    result = TTNSSolveResult(
        current_problem_identity, current_policy_identity,
        current_manifold_identity, current_request_identity,
        mounted, lowered, ground_state, energy, observables, real_time,
        imaginary_time, complex_time, request, identity,
    )
    workspace.warm_start = copy(state)
    workspace.warm_identity = identity
    workspace.last_request = request
    workspace.last_result = result
    return result
end

function _scan_fixed_request(request::TTNSSolveRequest,
                             target::TargetIrrep)
    return TTNSSolveRequest(
        target; ground_state=request.ground_state,
        observables=request.observables,
    )
end

function _scan_initial_states(workspace::TTNSWorkspace,
                              targets::Tuple, initial_states)
    if initial_states === nothing
        missing = TargetIrrep[
            target for target in targets
            if !haskey(workspace.scan_lanes, target) ||
               workspace.scan_lanes[target].warm_start === nothing
        ]
        isempty(missing) || throw(ArgumentError(
            "IrrepScan needs one initial state per target because at least one " *
            "target lane has no independent warm state",
        ))
        return Tuple(workspace.scan_lanes[target].warm_start for target in targets),
            false
    end
    states = Tuple(initial_states)
    length(states) == length(targets) || throw(DimensionMismatch(
        "IrrepScan initial_states must follow the target order exactly",
    ))
    all(state -> state isa TTNS, states) || throw(ArgumentError(
        "IrrepScan initial_states must contain only Graft TTNS values",
    ))
    return states, true
end

function _solve_scan!(workspace::TTNSWorkspace, solver::TTNSSolver,
                      problem::ImpurityProblem, request::TTNSSolveRequest;
                      initial_states=nothing)
    _validate_solve_request_contract(request)
    problem_statistics(problem) === :fermion || throw(ArgumentError(
        "TTNSSolver currently supports fermionic ImpurityProblem values only",
    ))
    request.manifold isa IrrepScan || throw(ArgumentError(
        "internal TTNS scan execution requires IrrepScan",
    ))
    symmetry = _solver_preflight(problem, solver, request)
    _validate_solver_problem_topology(solver, problem)

    scan = request.manifold
    targets = Tuple(TargetIrrep(scan.action_identity, outer)
                    for outer in scan.targets)
    states, supplied = _scan_initial_states(workspace, targets, initial_states)

    current_problem_identity = _solver_problem_identity(problem)
    current_policy_identity = _solver_policy_identity(solver)
    current_manifold_identity = _solver_manifold_identity(request)
    current_request_identity = _solver_request_identity(request)

    # Validate every ordered state against one isolated materialization before
    # mutating either the parent workspace or any persistent target lane.
    validation_workspace = TTNSWorkspace()
    validation_mounted, validation_lowered = _solver_materialize_build!(
        validation_workspace, solver, problem, symmetry,
    )
    for (target, state) in zip(targets, states)
        fixed_request = _scan_fixed_request(request, target)
        _validate_solver_state(state, validation_lowered, fixed_request)
        if !supplied
            lane = workspace.scan_lanes[target]
            fixed_manifold_identity = manifold_identity(target)
            fixed_request_identity = _solver_request_identity(fixed_request)
            expected_warm_identity = _solver_warm_identity(
                current_problem_identity, current_policy_identity,
                fixed_manifold_identity, fixed_request_identity,
                validation_mounted,
            )
            lane.warm_identity == expected_warm_identity || throw(ArgumentError(
                "IrrepScan target warm state is invalid for the current problem, " *
                "policy, target, or request; supply ordered initial_states",
            ))
        end
    end

    _invalidate_workspace_build!(workspace)
    workspace.problem_identity = current_problem_identity
    workspace.policy_identity = current_policy_identity
    workspace.manifold_identity = current_manifold_identity
    workspace.request_identity = current_request_identity

    ordered_results = TTNSSolveResult[]
    for (target, state) in zip(targets, states)
        lane = get!(workspace.scan_lanes, target) do
            TTNSWorkspace()
        end
        fixed_request = _scan_fixed_request(request, target)
        result = solve!(
            lane, solver, problem, fixed_request;
            initial_state=supplied ? state : nothing,
        )
        push!(ordered_results, result)
    end
    result = TTNSScanResult(
        current_problem_identity, current_policy_identity,
        current_manifold_identity, current_request_identity,
        scan.targets, Tuple(ordered_results), request,
    )
    workspace.last_request = request
    workspace.last_result = result
    return result
end

function _solve_manifold!(workspace::TTNSWorkspace, solver::TTNSSolver,
                          problem::ImpurityProblem,
                          request::TTNSSolveRequest,
                          ::TargetIrrep;
                          initial_state=nothing, initial_states=nothing)
    initial_states === nothing || throw(ArgumentError(
        "fixed TargetIrrep execution accepts initial_state, not initial_states",
    ))
    return _solve_fixed!(
        workspace, solver, problem, request; initial_state,
    )
end

function _solve_manifold!(workspace::TTNSWorkspace, solver::TTNSSolver,
                          problem::ImpurityProblem,
                          request::TTNSSolveRequest,
                          ::IrrepScan;
                          initial_state=nothing, initial_states=nothing)
    initial_state === nothing || throw(ArgumentError(
        "IrrepScan execution accepts ordered initial_states, not initial_state",
    ))
    return _solve_scan!(
        workspace, solver, problem, request; initial_states,
    )
end

function solve!(workspace::TTNSWorkspace, solver::TTNSSolver,
                problem::ImpurityProblem, request::TTNSSolveRequest;
                initial_state=nothing, initial_states=nothing)
    return _solve_manifold!(
        workspace, solver, problem, request, request.manifold;
        initial_state, initial_states,
    )
end
