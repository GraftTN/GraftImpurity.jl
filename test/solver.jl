using Test
using Graft
using GraftImpurity
using GreenFunc
using Graft.TestUtils: product_ttns, to_dense
using Graft.Backend: FermionParity, U1Irrep, ⊠
using LinearAlgebra: dot, exp, inv

struct _SolverSyntheticKernel <: AbstractRealPoleBathFitKernel
    energy::Float64
    residue::ComplexF64
end

function GraftImpurity.real_pole_bath_fit(input::BathFitInput,
                                          kernel::_SolverSyntheticKernel,
                                          partition::Partition)
    interval = SpectralInterval(-2.0, 2.0, 1)
    plan = DiscretizationPlan(
        :d => BlockDiscretizationPlan([interval]); shared_grid=true,
    )
    poles = BlockRealPoles(
        input.layout, partition, [kernel.energy], [kernel.residue], [1];
        statistics=:fermion,
    )
    return PoleExpansion(poles; kernel=:solver_synthetic, trace=(; plan))
end

struct _SolverNonMountableKernel <: AbstractRealPoleBathFitKernel end

function GraftImpurity.real_pole_bath_fit(input::BathFitInput,
                                          ::_SolverNonMountableKernel,
                                          partition::Partition)
    interval = SpectralInterval(-2.0, 2.0, 1)
    plan = DiscretizationPlan(
        :d => BlockDiscretizationPlan([interval]); shared_grid=true,
    )
    poles = BlockRealPoles(
        input.layout, partition, [0.2], ComplexF64[1.0 + 0.1im], [1];
        statistics=:fermion,
    )
    return PoleExpansion(poles; kernel=:solver_nonmountable, trace=(; plan))
end

function _solver_layout()
    return FlavorLayout(
        [:d], Dict(:d => :imp), Dict(:imp => [:d]); basis=:solver_fixture,
    )
end

function _solver_partition()
    return Partition(:d => [:d])
end

function _solver_hybridization_gf(; beta=10.0, energy=0.2, residue=0.16)
    mesh = ImFreq(beta, true; grid=[-2, -1, 0, 1, 2])
    data = ComplexF64[
        residue / (im * mesh[index] - energy) for index in eachindex(mesh)
    ]
    return Gf(mesh; data, statistics=true, component=:matsubara)
end

function _solver_weiss_gf(; beta=10.0, energy=0.2, residue=0.16)
    mesh = ImFreq(beta, true; grid=[-2, -1, 0, 1, 2])
    data = ComplexF64[]
    for index in eachindex(mesh)
        omega = mesh[index]
        delta = residue / (im * omega - energy)
        push!(data, inv(im * omega - delta))
    end
    return Gf(mesh; data, statistics=true, component=:matsubara)
end

function _solver_initial_state(layout; T::Type{<:Number}=ComplexF64)
    topology = TreeTopology(:imp, [:imp => :bath_d_1])
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    impurity = site_operators(operators, :imp)
    bath = FermionSiteOperators([:bath_mode]; sector=ParticleNumberSector())
    physical = Dict(:imp => impurity.P, :bath_d_1 => bath.P)
    vacuum = FermionParity(0) ⊠ U1Irrep(0)
    return product_ttns(
        T, topology, physical,
        Dict(:imp => vacuum, :bath_d_1 => vacuum),
    )
end

function _solver_value(; kernel=_SolverSyntheticKernel(0.2, 0.16 + 0im),
                       topology_plan=nothing, bath_mapping=nothing)
    layout = _solver_layout()
    partition = _solver_partition()
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    plan = topology_plan === nothing && bath_mapping === nothing ? T3NS(layout) :
           topology_plan
    solver = Solver(
        ; gf_struct=partition, layout, topology_plan=plan, bath_mapping,
        bath_fit_kernel=kernel, ops=operators, compression_atol=1e-12,
    )
    return solver, layout, operators
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

@testset "M6 stateful Solver" begin
    empty_raw = RawCorrelator(:empty, :synthetic, ComplexF64[], ComplexF64[])
    @test isempty(empty_raw.z_grid)
    @test isempty(empty_raw.values)
    @test_throws DimensionMismatch RawCorrelator(
        :mismatched, :synthetic, ComplexF64[0], ComplexF64[],
    )

    solver, layout, operators = _solver_value()
    h_loc = ImpurityOneBody(zeros(ComplexF64, 1, 1), layout)
    delta = _solver_hybridization_gf()
    set_hybridization!(solver, delta; h_loc0=h_loc)
    @test solver.input_kind === :hybridization
    @test solver.input isa BathFitInput
    @test solver.input.statistics === :fermion
    @test solver.input.domain === :matsubara
    @test solver.h_loc0 === h_loc
    set_weiss!(solver, _solver_weiss_gf(); h_loc0=h_loc)
    @test solver.input_kind === :weiss
    @test solver.source_input !== solver.input
    @test solver.expansion === nothing
    set_hybridization!(solver, delta; h_loc0=h_loc)
    @test solver.input_kind === :hybridization
    block_solver, _, _ = _solver_value()
    set_hybridization!(block_solver, BlockGf(:d => delta); h_loc0=h_loc)
    @test Tuple(keys(block_solver.input.blocks)) == (:d,)
    @test block_solver.source_input.source_template isa GreenFunc.BlockGf
    @test_throws ArgumentError set_hybridization!(
        block_solver, BlockGf(:wrong => delta); h_loc0=h_loc,
    )
    set_weiss!(block_solver, BlockGf(:d => _solver_weiss_gf()); h_loc0=h_loc)
    @test block_solver.input_kind === :weiss
    @test block_solver.source_input.source_template isa GreenFunc.BlockGf
    @test block_solver.input.source_template isa GreenFunc.BlockGf

    multi_layout = FlavorLayout(
        [:a, :b], Dict(:a => :imp_a, :b => :imp_b),
        Dict(:imp_a => [:a], :imp_b => [:b]); basis=:solver_multiblock,
    )
    multi_partition = Partition(:a => [:a], :b => [:b])
    multi_ops = ImpurityOperators(multi_layout; sector=ParticleNumberSector())
    multi_solver = Solver(
        ; gf_struct=multi_partition, layout=multi_layout,
        topology_plan=T3NS(multi_layout), bath_fit_kernel=_SolverSyntheticKernel(
            0.2, 0.16 + 0im,
        ), ops=multi_ops,
    )
    multi_hloc = ImpurityOneBody(zeros(ComplexF64, 2, 2), multi_layout)
    multi_source = BlockGf(:a => delta, :b => delta)
    set_hybridization!(multi_solver, multi_source; h_loc0=multi_hloc)
    @test Tuple(keys(multi_solver.input.blocks)) == (:a, :b)
    @test Tuple(keys(multi_solver.source_input.blocks)) == (:a, :b)
    cross_block_hloc = ImpurityOneBody(
        ComplexF64[0 0.1; 0.1 0], multi_layout,
    )
    @test_throws ArgumentError set_weiss!(
        multi_solver, multi_source; h_loc0=cross_block_hloc,
    )

    weiss_solver, _, _ = _solver_value()
    set_weiss!(weiss_solver, _solver_weiss_gf(); h_loc0=h_loc)
    @test weiss_solver.input_kind === :weiss
    @test weiss_solver.source_input !== weiss_solver.input
    @test weiss_solver.source_input.source_template isa GreenFunc.Gf
    @test weiss_solver.input.source_template isa GreenFunc.Gf
    expected_delta = ComplexF64[
        0.16 / (im * frequency - 0.2)
        for frequency in weiss_solver.input.frequencies
    ]
    @test [only(sample) for sample in weiss_solver.input.blocks.d] ≈ expected_delta
    near_singular = Gf(
        ImFreq(10.0, true; grid=[0]); data=ComplexF64[1e-320],
        statistics=true, component=:matsubara,
    )
    @test_throws ArgumentError set_weiss!(weiss_solver, near_singular; h_loc0=h_loc)
    @test_throws ArgumentError set_weiss!(
        weiss_solver, Gf(ImFreq(10.0, false; grid=[0, 1]);
                          data=ComplexF64[1, 1], statistics=false,
                          component=:matsubara); h_loc0=h_loc,
    )

    observable = LocalObservable(:occupation,
                                 :imp => site_operators(operators, :imp).N[1])
    correlator = LocalCorrelator(
        :particle,
        :imp => local_annihilator(site_operators(operators, :imp), :d),
        :imp => local_creator(site_operators(operators, :imp), :d),
    )
    real = RealTimeRequest([0.0, 0.05];
                            evolver=GlobalKrylov(krylovdim=4, maxiter=10,
                                                 fit_nsweeps=1, fit_tol=1e-10))
    contour = ComplexTimeRequest(
        ComplexTimeSegment(-0.05im, 1; label=:real_axis);
        evolver=GlobalKrylov(krylovdim=4, maxiter=10,
                             fit_nsweeps=1, fit_tol=1e-10),
    )
    request = SolveRequest(
        ; ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=2, tolerance=1e-10,
            krylovdim=4,
        ),
        real_time=real, complex_time=contour,
        observables=(observable,), correlators=(correlator,),
    )
    initial = _solver_initial_state(layout)
    @test_throws ArgumentError GroundStateResult(initial, Inf, [0.0])
    @test_throws ArgumentError GroundStateResult(initial, 0.0, [Inf])
    block_weiss_result = solve!(
        block_solver, DensityDensityInteraction(zeros(1, 1), layout),
        SolveRequest(; ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=1, krylovdim=4,
        )); initial_state=_solver_initial_state(layout),
    )
    @test block_weiss_result isa ImpurityResult
    @test block_weiss_result.input_kind === :weiss
    @test block_weiss_result.source_input === block_solver.source_input
    @test block_weiss_result.input === block_solver.input
    @test block_weiss_result.source_input !== block_weiss_result.input
    @test block_weiss_result.source_input.source_template isa GreenFunc.BlockGf
    @test block_weiss_result.input.source_template isa GreenFunc.BlockGf

    result = solve!(solver, DensityDensityInteraction(zeros(1, 1), layout),
                    request; initial_state=initial)
    @test result isa ImpurityResult
    @test solver.last_result === result
    @test solver.last_request === request
    @test result.ground_state.state isa TTNS
    @test result.energy == result.ground_state.energy
    @test result.observables.occupation isa ComplexF64
    @test result.real_time.particle.convention === :raw_correlator
    @test result.real_time.particle.contour === :real_time
    @test result.complex_time.particle.convention === :raw_correlator
    @test result.complex_time.particle.metadata.segment_labels ==
          (:initial, :real_axis)
    @test result.complex_time.particle.z_grid == ComplexF64[0, -0.05im]
    @test result.real_time.particle.z_grid == result.complex_time.particle.z_grid
    @test result.real_time.particle.metadata.coordinate === :core_step
    @test result.complex_time.particle.values ≈ result.real_time.particle.values atol=1e-8
    @test !result.bathfit_audit.passed
    @test any(item -> item.criterion === :request_horizon_to_revival,
              result.bathfit_audit.violations)

    tilted = ComplexTimeRequest(
        ComplexTimeSegment(-0.02 - 0.05im, 1; label=:tilted);
        evolver=GlobalKrylov(krylovdim=4, maxiter=10,
                             fit_nsweeps=1, fit_tol=1e-10),
    )
    parallel = ComplexTimeRequest(
        (ComplexTimeSegment(-0.02, 1; label=:parallel_offset),
         ComplexTimeSegment(-0.05im, 1; label=:parallel_real));
        evolver=GlobalKrylov(krylovdim=4, maxiter=10,
                             fit_nsweeps=1, fit_tol=1e-10),
    )
    kink = ComplexTimeRequest(
        (ComplexTimeSegment(-0.02, 1; label=:kink_imaginary),
         ComplexTimeSegment(-0.05im, 1; label=:kink_real),
         ComplexTimeSegment(-0.01, 1; label=:kink_return));
        evolver=GlobalKrylov(krylovdim=4, maxiter=10,
                             fit_nsweeps=1, fit_tol=1e-10),
    )
    for (request_value, labels, expected_grid) in (
        (tilted, (:initial, :tilted), ComplexF64[0, -0.02 - 0.05im]),
        (parallel, (:initial, :parallel_offset, :parallel_real),
         ComplexF64[0, -0.02, -0.02 - 0.05im]),
        (kink, (:initial, :kink_imaginary, :kink_real, :kink_return),
         ComplexF64[0, -0.02, -0.02 - 0.05im, -0.03 - 0.05im]),
    )
        raw = GraftImpurity._solver_complex_time(
            result.ground_state.state, result.energy, result.lowered,
            request_value, (correlator,),
        )
        @test raw.particle.convention === :raw_correlator
        @test raw.particle.metadata.segment_labels == labels
        @test raw.particle.z_grid == expected_grid
        @test raw.particle.values ≈ _solver_exact_complex_correlator(
            result.ground_state.state, result.energy, result.lowered,
            correlator, expected_grid,
        ) atol=1e-7 rtol=1e-7
    end

    set_hybridization!(solver, delta; h_loc0=h_loc)
    @test solver.last_result === nothing
    @test solver.warm_start === nothing
    @test solver.lowered === nothing
    refreshed_result = solve!(
        solver, DensityDensityInteraction(zeros(1, 1), layout), request;
        initial_state=_solver_initial_state(layout),
    )
    @test refreshed_result isa ImpurityResult

    warm_result = solve!(solver, DensityDensityInteraction(zeros(1, 1), layout),
                         request)
    @test warm_result isa ImpurityResult
    @test warm_result.warm_identity == refreshed_result.warm_identity
    changed = DensityDensityInteraction(reshape([0.3], 1, 1), layout)
    @test_throws ArgumentError solve!(solver, changed, request)
    @test solver.last_result === nothing
    @test solver.warm_start === nothing
    @test solver.lowered === nothing
    @test_throws ArgumentError solve!(
        solver, DensityDensityInteraction(zeros(1, 1), layout), SolveRequest();
        initial_state=_solver_initial_state(layout; T=ComplexF32),
    )
    set_hybridization!(solver, delta; h_loc0=h_loc)
    @test solver.last_result === nothing
    @test solver.warm_start === nothing

    implicit_request = SolveRequest(
        ; complex_time=ComplexTimeRequest(
            (ComplexTimeSegment(-0.02, 1; label=:imaginary),
             ComplexTimeSegment(-0.05im, 1; label=:real));
            evolver=ImplicitLogTime(),
        ),
        correlators=(correlator,),
    )
    @test_throws ArgumentError solve!(
        solver, DensityDensityInteraction(zeros(1, 1), layout), implicit_request;
        initial_state=_solver_initial_state(layout),
    )
    implicit_real_request = SolveRequest(
        ; real_time=RealTimeRequest([0.0, 0.05]; evolver=ImplicitLogTime()),
        correlators=(correlator,),
    )
    @test_throws ArgumentError solve!(
        solver, DensityDensityInteraction(zeros(1, 1), layout), implicit_real_request;
        initial_state=_solver_initial_state(layout),
    )

    thermal_density = LocalCorrelator(
        :density,
        :imp => site_operators(operators, :imp).N[1],
        :imp => site_operators(operators, :imp).N[1],
    )
    finite_request = SolveRequest(
        ; ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=1, krylovdim=4,
        ),
        imaginary_time=ImaginaryTimeRequest(
            [0.0, 0.5], GraftImpurity.FiniteTemperature(1.0);
            evolver=TDVP1(krylovdim=4, verbose=false),
            thermal_nsteps=1, propagation_nsteps=1,
        ),
        correlators=(thermal_density,),
    )
    finite_result = solve!(
        solver, DensityDensityInteraction(zeros(1, 1), layout), finite_request;
        initial_state=_solver_initial_state(layout),
    )
    @test finite_result.imaginary_time isa ImaginaryTimeResult
    @test finite_result.imaginary_time.temperature.beta_eff == 1.0
    @test finite_result.imaginary_time.correlators.density.convention ===
          :raw_correlator
    @test finite_result.imaginary_time.correlators.density.metadata.coordinate === :tau
    # The Solver owns topology/physical-space assembly and exact request
    # forwarding; core's dedicated thermal suites own the TDVP-versus-ED
    # convergence evidence. Reproduce the same core trajectory independently
    # so a wrong beta, save grid, evolver, or insertion cannot pass here.
    direct_problem = purification_problem(
        finite_result.lowered.opsum, finite_result.lowered.mounted.topology,
        GraftImpurity._mounted_physical_spaces(finite_result.lowered.mounted);
        hermitian=true,
    )
    direct_trajectory = thermalize(
        Purified(), direct_problem, 1.0;
        evolver=finite_request.imaginary_time.evolver, nsteps=1,
        save_betas=[0.5, 1.0],
    )
    direct_series = thermal_correlator(
        Purified(), direct_problem,
        thermal_density.left_site => thermal_density.left,
        thermal_density.right_site => thermal_density.right,
        1.0, [0.0, 0.5];
        evolver=finite_request.imaginary_time.evolver,
        trajectory=direct_trajectory, prop_nsteps=1,
    )
    @test finite_result.imaginary_time.trajectory.final.logZ ≈
          direct_trajectory.final.logZ atol=1e-12
    @test finite_result.imaginary_time.correlators.density.values ≈
          direct_series.values atol=1e-12

    ftps_solver, _, _ = _solver_value(topology_plan=FTPS(layout))
    set_hybridization!(ftps_solver, delta; h_loc0=h_loc)
    ftps_result = solve!(
        ftps_solver, DensityDensityInteraction(zeros(1, 1), layout),
        SolveRequest(; ground_state=GroundStateRequest(
            trunc=TruncationScheme(maxdim=4), nsweeps=1, krylovdim=4,
        )); initial_state=_solver_initial_state(layout),
    )
    @test ftps_result isa ImpurityResult
    @test ftps_result.mounted.topology == impurity_topology(
        FTPS(layout), _solver_partition(), ftps_result.discretization.bath,
    )

    nonmountable_solver, _, _ = _solver_value(kernel=_SolverNonMountableKernel())
    set_hybridization!(nonmountable_solver, delta; h_loc0=h_loc)
    nonmountable = solve!(
        nonmountable_solver, DensityDensityInteraction(zeros(1, 1), layout),
        SolveRequest(),
    )
    @test nonmountable isa NonMountableImpurityResult
    @test nonmountable_solver.last_result === nonmountable
    @test nonmountable.discretization isa NonMountablePoleFit

    group = CayleyOwnershipGroup(:d, [1], [:d])
    mapping = CayleyTreeKernel(ScalarCayley(), (group,))
    @test_throws ArgumentError Solver(
        ; gf_struct=_solver_partition(), layout, topology_plan=T3NS(layout),
        bath_mapping=mapping, bath_fit_kernel=_SolverSyntheticKernel(0.2, 0.16im),
        ops=operators,
    )
    mapped_solver, _, _ = _solver_value(bath_mapping=mapping)
    set_hybridization!(mapped_solver, delta; h_loc0=h_loc)
    @test_throws ArgumentError solve!(
        mapped_solver, DensityDensityInteraction(zeros(1, 1), layout),
        SolveRequest(); initial_state=_solver_initial_state(layout),
    )
    @test mapped_solver.mapping_result isa CayleyMappingResult
end
