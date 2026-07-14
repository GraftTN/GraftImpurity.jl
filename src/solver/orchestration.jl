function _solver_physical_manifest(operators::ImpurityOperators, phys)
    names = layout_sites(operators.layout)
    expected = NamedTuple{names}(Tuple(
        site_operators(operators, site).P for site in names
    ))
    phys === nothing && return expected

    raw = Dict{Symbol,ElementarySpace}()
    for (site, space) in pairs(phys)
        site isa Symbol || throw(ArgumentError(
            "Solver phys keys must be Symbols",
        ))
        space isa ElementarySpace || throw(ArgumentError(
            "Solver phys values must be Graft ElementarySpaces",
        ))
        raw[site] = space
    end
    Set(keys(raw)) == Set(names) || throw(ArgumentError(
        "Solver phys must declare exactly the FlavorLayout physical sites; " *
        "bath spaces are produced by mount_bath",
    ))
    for site in names
        raw[site] == getproperty(expected, site) || throw(ArgumentError(
            "Solver phys space at $site disagrees with ImpurityOperators",
        ))
    end
    return expected
end

function _validate_solver_topology(layout::FlavorLayout, topology_plan,
                                   bath_mapping)
    if bath_mapping === nothing
        topology_plan === nothing && throw(ArgumentError(
            "Solver needs an explicit T3NS, FTPS, or custom TreeTopology when " *
            "bath_mapping is nothing",
        ))
        topology_plan isa Union{T3NS,FTPS,TreeTopology} || throw(ArgumentError(
            "Solver topology_plan must be T3NS, FTPS, or TreeTopology",
        ))
        if topology_plan isa AbstractImpurityTopologyPlan
            topology_plan.layout == layout || throw(ArgumentError(
                "Solver topology plan FlavorLayout must match layout",
            ))
        else
            _validate_impurity_nodes(topology_plan, layout)
        end
    else
        bath_mapping isa CayleyTreeKernel || throw(ArgumentError(
            "Solver currently accepts only CayleyTreeKernel bath mappings",
        ))
        topology_plan === nothing || throw(ArgumentError(
            "Solver bath_mapping and topology_plan are mutually exclusive",
        ))
    end
    return nothing
end

function _validate_solver_onebody(onebody::Union{Nothing,ImpurityOneBody},
                                  layout::FlavorLayout, name::AbstractString)
    onebody === nothing && return nothing
    onebody.layout == layout || throw(ArgumentError(
        "$name FlavorLayout must match Solver.layout",
    ))
    return onebody
end

"""
    Solver(; gf_struct, layout, topology_plan, bath_mapping=nothing,
           phys=nothing, bath_fit_kernel, ops=ImpurityOperators(layout),
           symmetry=SymmetrySpec(layout), soc=nothing, compression_atol=0,
           scheme=TruncationScheme())

Construct an empty, stateful impurity solver. A fitting input is installed only
through one of the mutually exclusive `set_weiss!` or `set_hybridization!`
methods; construction never stores a mutable GreenFunc source by alias.
"""
function Solver(; gf_struct::Partition, layout::FlavorLayout,
                topology_plan=nothing,
                bath_mapping=nothing,
                phys=nothing,
                bath_fit_kernel::K,
                ops::O=ImpurityOperators(layout),
                symmetry::S=SymmetrySpec(layout),
                soc::Union{Nothing,ImpurityOneBody}=nothing,
                compression_atol::Real=0.0,
                scheme::T=TruncationScheme()) where {
                    K<:AbstractRealPoleBathFitKernel,O<:ImpurityOperators,
                    S<:SymmetrySpec,T<:TruncationScheme}
    validate_partition(gf_struct, layout)
    ops.layout == layout || throw(ArgumentError(
        "Solver ImpurityOperators FlavorLayout must match layout",
    ))
    symmetry.layout == layout || throw(ArgumentError(
        "Solver SymmetrySpec FlavorLayout must match layout",
    ))
    _validate_solver_onebody(soc, layout, "Solver soc")
    _validate_solver_topology(layout, topology_plan, bath_mapping)
    tolerance = Float64(compression_atol)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError(
        "Solver compression_atol must be finite and nonnegative",
    ))
    physical = _solver_physical_manifest(ops, phys)
    return Solver(
        gf_struct, layout, topology_plan, bath_mapping, physical,
        bath_fit_kernel, ops, symmetry, soc, tolerance, scheme,
        :unset,
        nothing, # source_input
        nothing, # input
        nothing, # h_loc0
        nothing, # expansion
        nothing, # discretization
        nothing, # mapping_result
        nothing, # mounted
        nothing, # interaction
        nothing, # lowered
        nothing, # bathfit_audit
        nothing, # warm_start
        nothing, # warm_identity
        nothing, # last_request
        nothing, # last_result
    )
end

function _invalidate_solver!(solver::Solver)
    solver.expansion = nothing
    solver.discretization = nothing
    solver.mapping_result = nothing
    solver.mounted = nothing
    solver.interaction = nothing
    solver.lowered = nothing
    solver.bathfit_audit = nothing
    solver.warm_start = nothing
    solver.warm_identity = nothing
    solver.last_request = nothing
    solver.last_result = nothing
    return solver
end

function _validate_solver_fit_input(solver::Solver, input::BathFitInput)
    input.layout == solver.layout || throw(ArgumentError(
        "Solver bath-fit input FlavorLayout must match Solver.layout",
    ))
    _validate_fit_input(input, solver.gf_struct)
    input.statistics === :fermion || throw(ArgumentError(
        "Solver currently supports fermionic GreenFunc fitting inputs only",
    ))
    hasproperty(input.metadata, :temperature) || throw(ArgumentError(
        "Solver GreenFunc input metadata must retain a temperature context",
    ))
    return input
end

function _solver_input_from_gf(solver::Solver, gf::GreenFunc.Gf;
                               block::Union{Nothing,Symbol}=nothing)
    names = block_names(solver.gf_struct)
    selected = if block === nothing
        length(names) == 1 || throw(ArgumentError(
            "a single GreenFunc.Gf needs an explicit block for a multi-block Solver",
        ))
        only(names)
    else
        block in names || throw(ArgumentError(
            "GreenFunc.Gf block $block is absent from Solver.gf_struct",
        ))
        block
    end
    return _validate_solver_fit_input(solver,
                                     BathFitInput(solver.layout, gf, selected))
end

function _solver_input_from_gf(solver::Solver, blocks::GreenFunc.BlockGf)
    return _validate_solver_fit_input(solver, BathFitInput(solver.layout, blocks))
end

function _replace_solver_input!(solver::Solver, kind::Symbol,
                                source_input::BathFitInput,
                                input::BathFitInput, h_loc0::ImpurityOneBody)
    kind in (:weiss, :hybridization) || throw(ArgumentError(
        "internal Solver input kind must be :weiss or :hybridization",
    ))
    _validate_solver_fit_input(solver, source_input)
    _validate_solver_fit_input(solver, input)
    _validate_solver_onebody(h_loc0, solver.layout, "Solver h_loc0")
    _invalidate_solver!(solver)
    solver.input_kind = kind
    solver.source_input = source_input
    solver.input = input
    solver.h_loc0 = h_loc0
    return solver
end

"""
    set_hybridization!(solver, Delta; h_loc0, block=nothing)

Install a direct GreenFunc hybridization input. `Delta` and a prior Weiss input
are mutually exclusive; replacing either invalidates fit, bath, Hamiltonian,
result, and warm-start state.
"""
function set_hybridization!(solver::Solver, delta::GreenFunc.Gf;
                             h_loc0::ImpurityOneBody,
                             block::Union{Nothing,Symbol}=nothing)
    input = _solver_input_from_gf(solver, delta; block)
    return _replace_solver_input!(solver, :hybridization, input, input, h_loc0)
end

function set_hybridization!(solver::Solver, delta::GreenFunc.BlockGf;
                             h_loc0::ImpurityOneBody)
    input = _solver_input_from_gf(solver, delta)
    return _replace_solver_input!(solver, :hybridization, input, input, h_loc0)
end

function _require_block_local_weiss_onebody(onebody::ImpurityOneBody,
                                             partition::Partition)
    matrix = onebody.matrix
    tolerance = _interaction_tolerance(matrix)
    names = block_names(partition)
    for left in eachindex(names), right in eachindex(names)
        left == right && continue
        left_indices = [flavor_index(onebody.layout, flavor)
                        for flavor in block_flavors(partition, names[left])]
        right_indices = [flavor_index(onebody.layout, flavor)
                         for flavor in block_flavors(partition, names[right])]
        maximum(abs, @view matrix[left_indices, right_indices]; init=0.0) <= tolerance ||
            throw(ArgumentError(
                "set_weiss! requires block-local h_loc0 because a BlockGf Weiss " *
                "input cannot represent cross-block hybridization",
            ))
    end
    return onebody
end

function _weiss_hybridization_input(input::BathFitInput,
                                    h_loc0::ImpurityOneBody,
                                    partition::Partition)
    input.domain === :matsubara || throw(ArgumentError(
        "set_weiss! requires a Matsubara GreenFunc input",
    ))
    input.statistics === :fermion || throw(ArgumentError(
        "set_weiss! requires fermionic GreenFunc statistics",
    ))
    _require_block_local_weiss_onebody(h_loc0, partition)
    any(name -> hasproperty(input.metadata, name), (:weiss_conversion, :h_loc_label)) &&
        throw(ArgumentError(
            "GreenFunc metadata keys :weiss_conversion and :h_loc_label are reserved " *
            "by set_weiss!",
        ))
    names = block_names(partition)
    converted = Tuple(map(names) do name
        indices = [flavor_index(input.layout, flavor)
                   for flavor in block_flavors(partition, name)]
        local_h = Matrix{ComplexF64}(h_loc0.matrix[indices, indices])
        dimension = length(indices)
        samples = Matrix{ComplexF64}[]
        for (index, sample) in enumerate(getproperty(input.blocks, name))
            inverse = try
                LinearAlgebra.inv(sample)
            catch err
                err isa LinearAlgebra.SingularException || rethrow()
                throw(ArgumentError(
                    "set_weiss! G0 block $name is singular at Matsubara index $index",
                ))
            end
            candidate = ComplexF64(im * input.frequencies[index]) *
                Matrix{ComplexF64}(I, dimension, dimension) - local_h - inverse
            all(value -> isfinite(real(value)) && isfinite(imag(value)), candidate) ||
                throw(ArgumentError(
                    "set_weiss! produced a nonfinite hybridization block $name " *
                    "at Matsubara index $index",
                ))
            push!(samples, candidate)
        end
        samples
    end)
    blocks = NamedTuple{names}(converted)
    metadata = merge(input.metadata, (; weiss_conversion=:explicit_inverse,
                                      h_loc_label=h_loc0.label))
    template = _reconstructed_template(input, blocks)
    result = BathFitInput(input.layout, input.domain, input.statistics,
                          copy(input.frequencies), blocks, input.target_labels,
                          metadata, template, Val(:validated))
    _validate_fit_input(result, partition)
    return result
end

"""
    set_weiss!(solver, G0_iw; h_loc0, block=nothing)

Install a Matsubara Weiss Green function by explicitly forming
`Delta(iω) = iω*I - h_loc0 - inv(G0(iω))`. The typed `h_loc0` is mandatory:
the Solver never guesses a chemical-potential or local one-body convention.
"""
function set_weiss!(solver::Solver, weiss::GreenFunc.Gf;
                    h_loc0::ImpurityOneBody,
                    block::Union{Nothing,Symbol}=nothing)
    input = _solver_input_from_gf(solver, weiss; block)
    converted = _weiss_hybridization_input(input, h_loc0, solver.gf_struct)
    return _replace_solver_input!(solver, :weiss, input, converted, h_loc0)
end

function set_weiss!(solver::Solver, weiss::GreenFunc.BlockGf;
                    h_loc0::ImpurityOneBody)
    input = _solver_input_from_gf(solver, weiss)
    converted = _weiss_hybridization_input(input, h_loc0, solver.gf_struct)
    return _replace_solver_input!(solver, :weiss, input, converted, h_loc0)
end

function _require_solver_input(solver::Solver)
    solver.source_input === nothing && throw(ArgumentError(
        "Solver has no source GreenFunc provenance for its fitting input",
    ))
    solver.input === nothing && throw(ArgumentError(
        "Solver needs set_weiss! or set_hybridization! before solve!",
    ))
    solver.h_loc0 === nothing && throw(ArgumentError(
        "Solver has no typed h_loc0 for Hamiltonian lowering",
    ))
    return solver.source_input, solver.input, solver.h_loc0
end

function _solver_bathfit_audit(report::BathFitReport, request::SolveRequest)
    return audit_bathfit(
        report,
        BathFitCriteria(
            beta=_request_beta(request),
            request_horizon=_request_time_horizon(request),
            require_mountable=true,
        ),
    )
end

function _solver_mount_bath!(solver::Solver, bath::DiscreteBath)
    if solver.bath_mapping !== nothing
        mapped = map_bath(solver.bath_mapping, bath)
        solver.mapping_result = mapped
        return mount_bath(mapped; sector=solver.ops.sector)
    end
    plan = solver.topology_plan
    if plan isa Union{T3NS,FTPS}
        return mount_bath(
            impurity_topology(plan, solver.gf_struct, bath), bath;
            sector=solver.ops.sector,
        )
    end
    plan isa TreeTopology || throw(ArgumentError(
        "Solver has no mountable topology plan",
    ))
    return mount_bath(plan, bath; sector=solver.ops.sector)
end

function _solver_warm_identity(solver::Solver,
                               interaction::AbstractImpurityInteraction,
                               mounted::AbstractMountedBath)
    bath_hash = _mounted_bath_integrity_hash(mounted)
    return hash((:GraftImpuritySolverWarmStart, solver.layout, solver.gf_struct,
                 mounted.topology, mounted.diagnostics.ownership_hash, bath_hash,
                 interaction, solver.h_loc0, solver.soc, solver.symmetry,
                 solver.ops.sector, solver.compression_atol))
end

function _state_requires_complex_eltype(request::SolveRequest)
    return request.real_time !== nothing || request.complex_time !== nothing
end

function _validate_solver_state(state::TTNS, lowered::LoweredImpurityHamiltonian,
                                request::SolveRequest)
    topology(state) == topology(lowered.operator) || throw(ArgumentError(
        "initial_state topology does not match the mounted impurity Hamiltonian",
    ))
    state.hasphys == lowered.operator.hasphys || throw(ArgumentError(
        "initial_state physical layout does not match the mounted impurity Hamiltonian",
    ))
    Graft.Backend.spacetype(state) == Graft.Backend.spacetype(lowered.operator) ||
        throw(ArgumentError(
            "initial_state symmetry space does not match the mounted impurity Hamiltonian",
        ))
    eltype(state) == eltype(lowered.operator) || throw(ArgumentError(
        "initial_state scalar type $(eltype(state)) does not match Hamiltonian " *
        "scalar type $(eltype(lowered.operator))",
    ))
    for site in propertynames(lowered.mounted.phys)
        node = Graft.Trees.nodeindex(state.topo, site)
        state.hasphys[node] || throw(ArgumentError(
            "initial_state is missing the physical leg at mounted site $site",
        ))
        Graft.physspace(state, node) == getproperty(lowered.mounted.phys, site) ||
            throw(ArgumentError(
                "initial_state physical space at $site does not match the mounted " *
                "impurity Hamiltonian",
            ))
    end
    !_state_requires_complex_eltype(request) || eltype(state) <: Complex ||
        throw(ArgumentError(
            "real/complex-time Solver requests require a complex-eltype initial_state",
        ))
    return state
end

function _solver_initial_state(solver::Solver, lowered::LoweredImpurityHamiltonian,
                               identity::UInt, request::SolveRequest,
                               initial_state;
                               warm_start=solver.warm_start,
                               warm_identity=solver.warm_identity)
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
        ComplexF64(Graft.expect(state, observable.op, observable.site))
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
    solve!(solver, interaction, request; initial_state=nothing)

Run the explicit lifecycle `fit -> realize -> topology/mount -> lower -> DMRG
-> requested correlators`. A non-mountable real-pole fit returns a typed
`NonMountableImpurityResult`. An explicit Cayley mapping mounts the retained
transformed bath Hamiltonian and compatible mapping topology when every
physical site carries one fermionic mode; multi-mode physical carriers reject
explicitly at mounting. The Solver never falls back to a canonical diagonal
star.
"""
function solve!(solver::Solver, interaction::AbstractImpurityInteraction,
                request::SolveRequest; initial_state=nothing)
    _validate_solve_request_contract(request)
    interaction.layout == solver.layout || throw(ArgumentError(
        "solve! interaction FlavorLayout must match Solver.layout",
    ))
    source_input, input, h_loc0 = _require_solver_input(solver)
    prior_warm_start = solver.warm_start
    prior_warm_identity = solver.warm_identity
    _invalidate_solver!(solver)

    expansion = real_pole_bath_fit(input, solver.bath_fit_kernel, solver.gf_struct)
    discretization = realize_bath(input, expansion, solver.gf_struct)

    if discretization isa NonMountablePoleFit
        audit = _solver_bathfit_audit(discretization.report, request)
        solver.expansion = expansion
        solver.discretization = discretization
        solver.interaction = interaction
        solver.bathfit_audit = audit
        result = NonMountableImpurityResult(
            source_input, solver.input_kind, h_loc0, input, expansion,
            discretization, audit, request,
        )
        solver.last_request = request
        solver.last_result = result
        return result
    end

    audit = _solver_bathfit_audit(discretization.report, request)
    mounted = _solver_mount_bath!(solver, discretization.bath)
    lowered = lower_hamiltonian(
        mounted, interaction, solver.ops;
        h_loc=h_loc0, soc=solver.soc, symmetry=solver.symmetry,
        compression_atol=solver.compression_atol, scheme=solver.scheme,
    )
    identity = _solver_warm_identity(solver, interaction, mounted)
    state = _solver_initial_state(
        solver, lowered, identity, request, initial_state;
        warm_start=prior_warm_start, warm_identity=prior_warm_identity,
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
    energy = Float64(real(Graft.expect(state, lowered.operator)))
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

    result = ImpurityResult(
        source_input, solver.input_kind, h_loc0, input, expansion, discretization,
        audit, mounted, lowered, ground_state, energy, observables, real_time,
        imaginary_time, complex_time, request, identity,
    )
    solver.expansion = expansion
    solver.discretization = discretization
    solver.mounted = mounted
    solver.interaction = interaction
    solver.lowered = lowered
    solver.bathfit_audit = audit
    solver.warm_start = copy(state)
    solver.warm_identity = identity
    solver.last_request = request
    solver.last_result = result
    return result
end
