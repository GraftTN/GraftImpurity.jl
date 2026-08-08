using Test
using GraftFoundation
using GraftNetworks
using GraftEvolution
using GraftThermal
using GraftImpurityFoundations
using GraftImpurityInteractions
using GraftImpurityBaths
using GraftImpurityProblems
using GraftTTNSSolver
using GraftTestUtils: product_ttns, to_dense
using LinearAlgebra: dot, exp

function _solver_layout()
    return FlavorLayout(
        [:d], Dict(:d => :imp), Dict(:imp => [:d]); basis=:solver_fixture,
    )
end

_solver_partition() = Partition(:d => [:d])

function _solver_problem(; energy=0.2, coupling=0.4,
                         h_loc=zeros(ComplexF64, 1, 1),
                         interaction=zeros(ComplexF64, 1, 1))
    layout = _solver_layout()
    partition = _solver_partition()
    orbitals = BathOrbitals(
        [energy], [[coupling + 0im]], [1], [1], [:d]; layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    problem = ImpurityProblem(
        bath, ImpurityOneBody(h_loc, layout),
        DensityDensityInteraction(interaction, layout),
    )
    return problem
end

function _solver_block_problem()
    layout = FlavorLayout(
        [:up, :down], Dict(:up => :up_site, :down => :down_site),
        Dict(:up_site => [:up], :down_site => [:down]);
        basis=:solver_block_fixture,
    )
    partition = Partition(:spin => [:up, :down])
    orbitals = BathOrbitals(
        [0.2, 0.3],
        [ComplexF64[0.4, 0.0], ComplexF64[0.0, 0.3]],
        [1, 2], [1, 1], [:up, :down]; layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    return ImpurityProblem(
        bath, ImpurityOneBody(zeros(ComplexF64, 2, 2), layout),
        DensityDensityInteraction(zeros(ComplexF64, 2, 2), layout),
    )
end

function _charge_target(problem::ImpurityProblem, charge::Integer)
    action = only(symmetry_actions(problem.symmetry))
    return TargetIrrep(action, U1Irrep(charge))
end

function _solver_initial_state(problem::ImpurityProblem, solver::TTNSSolver;
                               occupied_site::Union{Nothing,Symbol}=nothing,
                               T::Type{<:Number}=ComplexF64)
    workspace = TTNSWorkspace()
    mounted = GraftTTNSSolver._solver_mount_bath!(
        workspace, solver, problem.bath,
    )
    names = propertynames(mounted.phys)
    first_space = getproperty(mounted.phys, first(names))
    physical = Dict{Symbol,typeof(first_space)}(
        site => getproperty(mounted.phys, site) for site in names
    )
    vacuum = FermionParity(0) ⊠ U1Irrep(0)
    local_sectors = Dict(site => vacuum for site in keys(physical))
    if occupied_site !== nothing
        local_sectors[occupied_site] = FermionParity(1) ⊠ U1Irrep(1)
    end
    return product_ttns(T, mounted.topology, physical, local_sectors), mounted
end

function _solver_exact_complex_correlator(state::TTNS, energy::Real,
                                           lowered::LoweredImpurityHamiltonian,
                                           channel::LocalCorrelator, z_grid)
    hamiltonian = to_dense(lowered.operator)
    bra = to_dense(apply_local(state, adjoint(channel.left), channel.left_site))
    ket = to_dense(apply_local(state, channel.right, channel.right_site))
    return ComplexF64[
        exp(-energy * z) * dot(bra, exp(z * hamiltonian) * ket)
        for z in z_grid
    ]
end

struct _OpaqueAction
    layout::FlavorLayout
end

struct _OpaqueActionSemantics <: AbstractPhysicalActionSemantics
    generator::Tuple
end

struct _MimicChargeAction
    layout::FlavorLayout
end

function GraftImpurityProblems.action_identity(action::_OpaqueAction)
    return SymmetryActionIdentity(
        :opaque, action.layout, (Val(:opaque),),
        _OpaqueActionSemantics((1,)),
    )
end


function GraftImpurityProblems.action_identity(action::_MimicChargeAction)
    return SymmetryActionIdentity(
        :charge, action.layout, (U1Irrep,),
        _OpaqueActionSemantics((2,)),
    )
end

mutable struct _FieldlessMutableInteraction{L<:FlavorLayout} <:
        AbstractImpurityInteraction
    basis::L
    coefficient::Float64
end

GraftImpurityFoundations.interaction_layout(
    interaction::_FieldlessMutableInteraction,
) = interaction.basis
GraftImpurityFoundations.interaction_identity(
    interaction::_FieldlessMutableInteraction,
) = (
    family=:fieldless_mutable_test,
    layout=interaction.basis,
    coefficient=interaction.coefficient,
)

function GraftImpurityFoundations.lower_interaction(
        interaction::_FieldlessMutableInteraction,
        operators::ImpurityOperators, symmetry::SymmetrySpec)
    reference = DensityDensityInteraction(
        reshape(ComplexF64[interaction.coefficient], 1, 1),
        interaction.basis,
    )
    return lower_interaction(reference, operators, symmetry)
end

@testset "TTNS immutable policy and workspace boundary" begin
    problem = _solver_problem()
    layout = problem_layout(problem)
    solver = TTNSSolver(
        ; topology_plan=T3NS(layout), carrier=ParticleNumberSector(),
        compression_atol=1e-12,
    )
    @test !ismutabletype(typeof(solver))
    @test fieldnames(typeof(solver)) ==
        (:topology_plan, :bath_mapping, :carrier, :ttno_builder,
         :compression_atol, :scheme)
    @test solver.carrier isa ParticleNumberSector
    @test_throws ArgumentError TTNSSolver()
    group = CayleyOwnershipGroup(:d, [1], [:d])
    mapping = CayleyTreeKernel(ScalarCayley(), (group,))
    @test_throws ArgumentError TTNSSolver(
        ; topology_plan=T3NS(layout), bath_mapping=mapping,
    )

    workspace = TTNSWorkspace()
    other = TTNSWorkspace()
    @test workspace isa AbstractImpurityWorkspace
    @test workspace.mounted === workspace.lowered === workspace.warm_start === nothing
    @test other.mounted === other.last_result === nothing
    @test isempty(workspace.scan_lanes)

    target = _charge_target(problem, 0)
    request = TTNSSolveRequest(target)
    @test request.manifold === target
    @test_throws ArgumentError TTNSSolveRequest(
        target; real_time=:invalid,
    )

    invalid_scan_workspace = TTNSWorkspace()
    invalid_scan = IrrepScan(only(symmetry_actions(problem.symmetry)),
                             (U1Irrep(0), :opaque))
    @test_throws TTNSCapabilityError solve!(
        invalid_scan_workspace, solver, problem,
        TTNSSolveRequest(invalid_scan),
    )
    @test invalid_scan_workspace.mounted === nothing
    @test invalid_scan_workspace.problem_identity === nothing
    @test isempty(invalid_scan_workspace.scan_lanes)

    opaque_action = _OpaqueAction(layout)
    opaque_problem = ImpurityProblem(
        problem.bath, problem.h_loc, problem.interaction,
        ImpuritySymmetryDeclaration((opaque_action,)),
    )
    opaque_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        opaque_workspace, solver, opaque_problem,
        TTNSSolveRequest(TargetIrrep(opaque_action, :q)),
    )
    @test opaque_workspace.mounted === nothing
    @test opaque_workspace.problem_identity === nothing

    flavor_action = FlavorU1(:spin, [1.0], layout)
    multi_action_problem = ImpurityProblem(
        problem.bath, problem.h_loc, problem.interaction,
        ImpuritySymmetryDeclaration((ChargeU1(layout), flavor_action)),
    )
    multi_action_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        multi_action_workspace, solver, multi_action_problem,
        TTNSSolveRequest(TargetIrrep(ChargeU1(layout), U1Irrep(0))),
    )
    @test multi_action_workspace.mounted === nothing
    @test multi_action_workspace.problem_identity === nothing

    axial = FlavorU1(:sz, [1.0], layout)
    su2 = SU2Reduce(layout; axial_generator=axial)
    su2_problem = ImpurityProblem(
        problem.bath, problem.h_loc, problem.interaction,
        ImpuritySymmetryDeclaration((su2,)),
    )
    su2_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        su2_workspace, solver, su2_problem,
        TTNSSolveRequest(TargetIrrep(su2, SU2Irrep(0))),
    )
    @test su2_workspace.mounted === nothing
    @test su2_workspace.problem_identity === nothing
end

@testset "extension interaction identity invalidates TTNS build" begin
    baseline = _solver_problem()
    layout = problem_layout(baseline)
    interaction = _FieldlessMutableInteraction(layout, 0.0)
    problem = ImpurityProblem(
        baseline.bath, baseline.h_loc, interaction,
    )
    solver = TTNSSolver(
        ; topology_plan=T3NS(layout), compression_atol=1e-12,
    )
    target = _charge_target(problem, 0)
    request = TTNSSolveRequest(
        target; ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=1, krylovdim=4,
        ),
    )
    initial, _ = _solver_initial_state(problem, solver)
    workspace = TTNSWorkspace()
    first_result = solve!(
        workspace, solver, problem, request; initial_state=initial,
    )
    @test first_result.lowered.interaction === interaction
    first_problem_identity = problem_identity(problem)
    first_lowered = workspace.lowered

    interaction.coefficient = 0.25
    @test problem_identity(problem) != first_problem_identity
    @test_throws ArgumentError solve!(workspace, solver, problem, request)
    @test workspace.problem_identity == problem_identity(problem)
    @test workspace.lowered !== first_lowered
    @test workspace.warm_start === nothing
    rebuilt = solve!(
        workspace, solver, problem, request; initial_state=initial,
    )
    @test rebuilt.lowered.interaction === interaction
    @test rebuilt.problem_identity != first_result.problem_identity
end

@testset "TTNS bounded ChargeU1 scan execution" begin
    problem = _solver_problem()
    layout = problem_layout(problem)
    action = only(symmetry_actions(problem.symmetry))
    solver = TTNSSolver(
        ; topology_plan=T3NS(layout), compression_atol=1e-12,
    )
    target0 = TargetIrrep(action, U1Irrep(0))
    target1 = TargetIrrep(action, U1Irrep(1))
    scan = IrrepScan(action, (target0.target, target1.target))
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    observable = LocalObservable(
        :occupation, :imp => site_operators(operators, :imp).N[1],
    )
    ground_state = GroundStateRequest(
        trunc=TruncationScheme(maxdim=4), nsweeps=2,
        tolerance=1e-11, krylovdim=4,
    )

    density = LocalCorrelator(
        :density, target0,
        :imp => site_operators(operators, :imp).N[1],
        :imp => site_operators(operators, :imp).N[1],
    )
    unsupported_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        unsupported_workspace, solver, problem,
        TTNSSolveRequest(scan; correlators=(density,)),
    )
    @test unsupported_workspace.mounted === nothing
    @test unsupported_workspace.problem_identity === nothing
    @test isempty(unsupported_workspace.scan_lanes)

    initial0, _ = _solver_initial_state(problem, solver)
    initial1, _ = _solver_initial_state(problem, solver; occupied_site=:imp)
    workspace = TTNSWorkspace()
    request = TTNSSolveRequest(
        scan; ground_state, observables=(observable,),
    )
    result = solve!(
        workspace, solver, problem, request;
        initial_states=(initial0, initial1),
    )
    @test result isa TTNSScanResult
    @test result.targets == (U1Irrep(0), U1Irrep(1))
    @test result.results[1].request.manifold == target0
    @test result.results[2].request.manifold == target1
    @test result[U1Irrep(0)] === result.results[1]
    @test result[U1Irrep(1)] === result.results[2]
    @test result.results[1].energy ≈ 0.0 atol=1e-10
    @test result.results[2].energy ≈
        0.1 - sqrt(0.17) atol=1e-8
    @test result.results[1].observables.occupation ≈ 0.0 atol=1e-10
    @test length(workspace.scan_lanes) == 2

    lane0 = workspace.scan_lanes[target0]
    lane1 = workspace.scan_lanes[target1]
    @test lane0 !== lane1
    @test lane0.lowered !== lane1.lowered
    @test lane0.warm_start !== lane1.warm_start
    @test lane0.manifold_identity == manifold_identity(target0)
    @test lane1.manifold_identity == manifold_identity(target1)
    @test lane0.warm_identity != lane1.warm_identity
    cached0 = lane0.lowered
    cached1 = lane1.lowered

    warm = solve!(workspace, solver, problem, request)
    @test warm.targets == result.targets
    @test workspace.scan_lanes[target0] === lane0
    @test workspace.scan_lanes[target1] === lane1
    @test lane0.lowered === cached0
    @test lane1.lowered === cached1

    parent_last_result = workspace.last_result
    lane0_warm = lane0.warm_start
    lane1_warm = lane1.warm_start
    lane0_last_result = lane0.last_result
    lane1_last_result = lane1.last_result
    @test_throws ArgumentError solve!(
        workspace, solver, problem, request;
        initial_states=(initial0, initial0),
    )
    @test workspace.last_result === parent_last_result
    @test workspace.scan_lanes[target0] === lane0
    @test workspace.scan_lanes[target1] === lane1
    @test lane0.lowered === cached0
    @test lane1.lowered === cached1
    @test lane0.warm_start === lane0_warm
    @test lane1.warm_start === lane1_warm
    @test lane0.last_result === lane0_last_result
    @test lane1.last_result === lane1_last_result

    wrong_eltype, _ = _solver_initial_state(
        problem, solver; occupied_site=:imp, T=ComplexF32,
    )
    @test_throws ArgumentError solve!(
        workspace, solver, problem, request;
        initial_states=(initial0, wrong_eltype),
    )
    @test workspace.last_result === parent_last_result
    @test lane0.lowered === cached0
    @test lane1.lowered === cached1
    @test lane0.warm_start === lane0_warm
    @test lane1.warm_start === lane1_warm

    reversed_scan = IrrepScan(action, (U1Irrep(1), U1Irrep(0)))
    reversed = solve!(
        workspace, solver, problem,
        TTNSSolveRequest(reversed_scan; ground_state, observables=(observable,)),
    )
    @test reversed.targets == (U1Irrep(1), U1Irrep(0))
    @test reversed.results[1].request.manifold == target1
    @test reversed.results[2].request.manifold == target0
    @test workspace.scan_lanes[target0] === lane0
    @test workspace.scan_lanes[target1] === lane1
end

@testset "TTNS problem execution and cache identities" begin
    problem = _solver_problem()
    layout = problem_layout(problem)
    solver = TTNSSolver(
        ; topology_plan=T3NS(layout), compression_atol=1e-12,
    )
    target = _charge_target(problem, 0)
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    observable = LocalObservable(
        :occupation, :imp => site_operators(operators, :imp).N[1],
    )
    correlator = LocalCorrelator(
        :particle,
        _charge_target(problem, 1),
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    @test correlator.response_target == _charge_target(problem, 1)
    @test_throws MethodError LocalCorrelator(
        :missing_response,
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )

    wrong_reachability = LocalCorrelator(
        :wrong_reachability, target,
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    wrong_reachability_workspace = TTNSWorkspace()
    @test_throws ResponseReachabilityError solve!(
        wrong_reachability_workspace, solver, problem,
        TTNSSolveRequest(target; correlators=(wrong_reachability,)),
    )
    @test wrong_reachability_workspace.mounted === nothing
    @test wrong_reachability_workspace.problem_identity === nothing

    wrong_closure = LocalCorrelator(
        :wrong_closure, _charge_target(problem, 1),
        :imp => local_creator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    wrong_closure_workspace = TTNSWorkspace()
    @test_throws ResponseReachabilityError solve!(
        wrong_closure_workspace, solver, problem,
        TTNSSolveRequest(target; correlators=(wrong_closure,)),
    )
    @test wrong_closure_workspace.mounted === nothing
    @test wrong_closure_workspace.problem_identity === nothing

    mimic = _MimicChargeAction(layout)
    wrong_action_response = LocalCorrelator(
        :wrong_action,
        TargetIrrep(mimic, U1Irrep(1)),
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    wrong_action_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        wrong_action_workspace, solver, problem,
        TTNSSolveRequest(target; correlators=(wrong_action_response,)),
    )
    @test wrong_action_workspace.mounted === nothing
    @test wrong_action_workspace.problem_identity === nothing

    wrong_basis_layout = FlavorLayout(
        [:d], Dict(:d => :imp), Dict(:imp => [:d]);
        basis=:wrong_response_basis,
    )
    wrong_basis_identity = SymmetryActionIdentity(
        :charge, wrong_basis_layout, (U1Irrep,),
        ChargeU1ActionSemantics((1,)),
    )
    wrong_basis_response = LocalCorrelator(
        :wrong_basis, TargetIrrep(wrong_basis_identity, U1Irrep(1)),
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    wrong_basis_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        wrong_basis_workspace, solver, problem,
        TTNSSolveRequest(target; correlators=(wrong_basis_response,)),
    )
    @test wrong_basis_workspace.mounted === nothing
    @test wrong_basis_workspace.problem_identity === nothing

    wrong_product_identity = SymmetryActionIdentity(
        :charge, layout, (U1Irrep, Val(:extra)),
        ChargeU1ActionSemantics((1,)),
    )
    wrong_product_response = LocalCorrelator(
        :wrong_product, TargetIrrep(wrong_product_identity, U1Irrep(1)),
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    wrong_product_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        wrong_product_workspace, solver, problem,
        TTNSSolveRequest(target; correlators=(wrong_product_response,)),
    )
    @test wrong_product_workspace.mounted === nothing
    @test wrong_product_workspace.problem_identity === nothing

    opaque_response = LocalCorrelator(
        :opaque_response,
        TargetIrrep(only(symmetry_actions(problem.symmetry)), :opaque),
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    opaque_response_workspace = TTNSWorkspace()
    @test_throws TTNSCapabilityError solve!(
        opaque_response_workspace, solver, problem,
        TTNSSolveRequest(target; correlators=(opaque_response,)),
    )
    @test opaque_response_workspace.mounted === nothing
    @test opaque_response_workspace.problem_identity === nothing

    half_target_workspace = TTNSWorkspace()
    half_target = TargetIrrep(
        only(symmetry_actions(problem.symmetry)), U1Irrep(1 // 2),
    )
    @test_throws TTNSCapabilityError solve!(
        half_target_workspace, solver, problem,
        TTNSSolveRequest(half_target),
    )
    @test half_target_workspace.mounted === nothing
    @test half_target_workspace.problem_identity === nothing
    real = RealTimeRequest(
        [0.0, 0.05];
        evolver=GlobalKrylov(krylovdim=4, maxiter=10,
                             fit_nsweeps=1, fit_tol=1e-10),
    )
    contour = ComplexTimeRequest(
        ComplexTimeSegment(-0.05im, 1; label=:real_axis);
        evolver=GlobalKrylov(krylovdim=4, maxiter=10,
                             fit_nsweeps=1, fit_tol=1e-10),
    )
    request = TTNSSolveRequest(
        target;
        ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=2,
            tolerance=1e-10, krylovdim=4,
        ),
        real_time=real, complex_time=contour,
        observables=(observable,), correlators=(correlator,),
    )
    initial, _ = _solver_initial_state(problem, solver)
    workspace = TTNSWorkspace()
    result = solve!(workspace, solver, problem, request; initial_state=initial)

    @test result isa TTNSSolveResult
    @test workspace.last_result === result
    @test workspace.last_request === request
    @test workspace.ops isa ImpurityOperators
    @test result.problem_identity == problem_identity(problem)
    @test result.manifold_identity == manifold_identity(target)
    @test result.request_identity == workspace.request_identity
    @test result.policy_identity == workspace.policy_identity
    @test result.energy == result.ground_state.energy
    @test result.observables.occupation isa ComplexF64
    @test result.real_time.particle.convention === :raw_correlator
    @test result.complex_time.particle.metadata.segment_labels ==
        (:initial, :real_axis)
    @test result.real_time.particle.z_grid == result.complex_time.particle.z_grid
    @test result.complex_time.particle.values ≈
        result.real_time.particle.values atol=1e-8
    @test !hasfield(typeof(result), :source_input)
    @test !hasfield(typeof(result), :discretization)
    @test !hasfield(typeof(result), :bathfit_audit)

    tilted = ComplexTimeRequest(
        ComplexTimeSegment(-0.02 - 0.05im, 1; label=:tilted);
        evolver=DirectKrylovBootstrap(krylovdim=4, max_basis=4),
    )
    raw = GraftTTNSSolver._solver_complex_time(
        result.ground_state.state, result.energy, result.lowered,
        tilted, (correlator,),
    )
    @test raw.particle.values ≈ _solver_exact_complex_correlator(
        result.ground_state.state, result.energy, result.lowered,
        correlator, raw.particle.z_grid,
    ) atol=1e-7 rtol=1e-7

    cached_lowered = workspace.lowered
    warm = solve!(workspace, solver, problem, request)
    @test warm.warm_identity == result.warm_identity
    @test workspace.lowered === cached_lowered

    changed_request = TTNSSolveRequest(
        target;
        ground_state=request.ground_state,
        observables=(observable,),
    )
    @test_throws ArgumentError solve!(
        workspace, solver, problem, changed_request,
    )
    @test workspace.lowered === cached_lowered
    @test workspace.warm_start === nothing
    refreshed = solve!(
        workspace, solver, problem, changed_request; initial_state=initial,
    )
    @test refreshed.request_identity != result.request_identity
    @test workspace.lowered === cached_lowered

    changed_problem = _solver_problem(h_loc=reshape(ComplexF64[0.1], 1, 1))
    @test_throws ArgumentError solve!(
        workspace, solver, changed_problem,
        TTNSSolveRequest(_charge_target(changed_problem, 0);
                         ground_state=request.ground_state),
    )
    @test workspace.problem_identity == problem_identity(changed_problem)
    @test workspace.warm_start === nothing

    isolated = TTNSWorkspace()
    @test isolated.mounted === isolated.lowered === isolated.last_result === nothing
end

@testset "TTNS finite-temperature and topology routes" begin
    problem = _solver_problem()
    layout = problem_layout(problem)
    target = _charge_target(problem, 0)
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    density = LocalCorrelator(
        :density,
        target,
        :imp => site_operators(operators, :imp).N[1],
        :imp => site_operators(operators, :imp).N[1],
    )

    t3ns = TTNSSolver(; topology_plan=T3NS(layout), compression_atol=1e-12)
    initial, _ = _solver_initial_state(problem, t3ns)
    finite_request = TTNSSolveRequest(
        target;
        ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=1, krylovdim=4,
        ),
        imaginary_time=ImaginaryTimeRequest(
            [0.0, 0.5], GraftTTNSSolver.FiniteTemperature(1.0);
            evolver=TDVP1(krylovdim=4, verbose=false),
            thermal_nsteps=1, propagation_nsteps=1,
        ),
        correlators=(density,),
    )
    finite = solve!(
        TTNSWorkspace(), t3ns, problem, finite_request; initial_state=initial,
    )
    @test finite.imaginary_time isa ImaginaryTimeResult
    @test finite.imaginary_time.temperature.beta_eff == 1.0
    @test finite.imaginary_time.correlators.density.metadata.coordinate === :tau

    ftps = TTNSSolver(; topology_plan=FTPS(layout), compression_atol=1e-12)
    ftps_initial, _ = _solver_initial_state(problem, ftps)
    ftps_result = solve!(
        TTNSWorkspace(), ftps, problem,
        TTNSSolveRequest(target; ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=1, krylovdim=4,
        )); initial_state=ftps_initial,
    )
    @test ftps_result.mounted.topology == impurity_topology(
        FTPS(layout), problem_partition(problem), problem.bath,
    )

    group = CayleyOwnershipGroup(:d, [1], [:d])
    mapping = CayleyTreeKernel(ScalarCayley(), (group,))
    cayley = TTNSSolver(; bath_mapping=mapping, compression_atol=1e-12)
    cayley_initial, cayley_mounted = _solver_initial_state(
        problem, cayley; occupied_site=:imp,
    )
    one_particle_request = TTNSSolveRequest(
        _charge_target(problem, 1);
        ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=2, krylovdim=4,
        ),
    )
    cayley_workspace = TTNSWorkspace()
    cayley_result = solve!(
        cayley_workspace, cayley, problem, one_particle_request;
        initial_state=cayley_initial,
    )
    @test cayley_result.mounted isa CayleyAndersonBath
    @test cayley_result.mounted.topology == cayley_mounted.topology
    @test cayley_workspace.mapping_result === cayley_result.mounted.mapping

    block_problem = _solver_block_problem()
    block_layout = problem_layout(block_problem)
    block_group = CayleyOwnershipGroup(:spin, [1, 2], [:up, :down])
    block_mapping = CayleyTreeKernel(BlockCayley(), (block_group,))
    block_solver = TTNSSolver(
        ; bath_mapping=block_mapping, compression_atol=1e-12,
    )
    block_initial, block_mounted = _solver_initial_state(
        block_problem, block_solver; occupied_site=:up_site,
    )
    block_workspace = TTNSWorkspace()
    block_result = solve!(
        block_workspace, block_solver, block_problem,
        TTNSSolveRequest(_charge_target(block_problem, 1);
            ground_state=GroundStateRequest(
                trunc=TruncationScheme(maxdim=8), nsweeps=2, krylovdim=8,
            ));
        initial_state=block_initial,
    )
    @test block_result.mounted isa CayleyAndersonBath
    @test block_result.mounted.mapping.mapped isa BlockCayleyBath
    @test block_result.mounted.topology == block_mounted.topology
end
