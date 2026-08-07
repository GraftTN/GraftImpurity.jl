using Test
using LinearAlgebra: dot, norm
using Random: Xoshiro, randn
using GraftFoundation
using GraftNetworks
using GraftContractions
using GraftSymbolic
using GraftStateDiagram
using GraftImpurityFoundations
using GraftImpurityInteractions
using GraftImpurityBaths
using GraftImpurityBathFit
using GraftImpurityProblems
using GraftImpuritySolver
using GreenFunc
using GraftTestUtils: categorical_coordinates, product_ttns, to_dense

function _ttnob_physical(mounted)
    names = propertynames(mounted.phys)
    first_space = getproperty(mounted.phys, first(names))
    physical = Dict{Symbol,typeof(first_space)}()
    for site in names
        physical[site] = getproperty(mounted.phys, site)
    end
    return physical
end

function _ttnob_random_one_particle_probe(mounted; seed)
    physical = _ttnob_physical(mounted)
    vacuum = FermionParity(0) ⊠ U1Irrep(0)
    occupied = FermionParity(1) ⊠ U1Irrep(1)
    basis = TTNS[]
    for site in propertynames(mounted.phys)
        for index in 1:dim(physical[site], occupied)
            sectors = Dict{Symbol,Any}(name => vacuum for name in keys(physical))
            sectors[site] = occupied => index
            push!(basis, product_ttns(
                ComplexF64, mounted.topology, physical, sectors,
            ))
        end
    end
    coefficients = randn(Xoshiro(seed), ComplexF64, length(basis))
    return normalize!(exact_linear_combination(basis, coefficients))
end

function _ttnob_compare_lowered(lowered; seed)
    matrices = to_dense.([value.operator for value in lowered])
    for matrix in matrices[2:end]
        @test matrix ≈ first(matrices) atol=1e-10 rtol=1e-10
    end

    rng = Xoshiro(seed)
    for _ in 1:3
        state = randn(rng, ComplexF64, size(first(matrices), 1))
        state ./= norm(state)
        expectations = [dot(state, matrix * state) for matrix in matrices]
        @test expectations[2:end] ≈
            fill(first(expectations), length(expectations) - 1) atol=1e-10 rtol=1e-10
    end

    probe = _ttnob_random_one_particle_probe(first(lowered).mounted; seed=seed + 1)
    expectations = [expect(probe, value.operator) for value in lowered]
    @test expectations[2:end] ≈
        fill(first(expectations), length(expectations) - 1) atol=1e-10 rtol=1e-10
    actions = [
        categorical_coordinates(apply(value.operator, probe; optimize=false))
        for value in lowered
    ]
    for action in actions[2:end]
        @test action ≈ first(actions) atol=1e-10 rtol=1e-10
    end
    return matrices
end

function _ttnob_anderson_fixture()
    layout = FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis=:ttno_builder_anderson,
    )
    partition = Partition(:spin => [:up, :down])
    orbitals = BathOrbitals(
        [0.37], [ComplexF64[0.22 + 0.03im, -0.11im]], [1], [1], [:up];
        layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    mounted = mount_bath(
        TreeTopology(:impurity, [:impurity => :bath_spin_1]), bath;
        site_labels=[:bath_spin_1], sector=ParticleNumberSector(),
    )
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    interaction = DensityDensityInteraction(ComplexF64[0 1.1; 1.1 0], layout)
    h_loc = ImpurityOneBody(
        ComplexF64[0.13 0.04im; -0.04im -0.21], layout,
    )
    return (; layout, mounted, operators, interaction, h_loc)
end

function _ttnob_cayley_fixture()
    layout = FlavorLayout(
        [:d], Dict(:d => :impurity), Dict(:impurity => [:d]);
        basis=:ttno_builder_cayley,
    )
    partition = Partition(:d => [:d])
    energies = [-0.65, 0.45]
    couplings = ComplexF64[0.31 + 0.08im, -0.19im]
    orbitals = BathOrbitals(
        energies, [[value] for value in couplings], [1, 2], [1, 1], [:d, :d];
        layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    group = CayleyOwnershipGroup(:d_chain, [1, 2], [:d])
    mapping = map_bath(
        CayleyTreeKernel(ScalarCayley(), [group]; branching=2), bath,
    )
    mounted = mount_bath(mapping; sector=ParticleNumberSector())
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    interaction = DensityDensityInteraction(zeros(ComplexF64, 1, 1), layout)
    h_loc = ImpurityOneBody(reshape(ComplexF64[0.17], 1, 1), layout)
    return (; mounted, operators, interaction, h_loc, mapping)
end

function _ttnob_lower(fixture, builder)
    return lower_hamiltonian(
        fixture.mounted, fixture.interaction, fixture.operators;
        h_loc=fixture.h_loc, ttno_builder=builder, compression_atol=1e-12,
    )
end

function _ttnob_full_coulomb_conventions(layout::FlavorLayout)
    bare = zeros(ComplexF64, 2, 2, 2, 2)
    bare[1, 2, 1, 2] = 1.1
    bare[2, 1, 2, 1] = 1.1

    vertex = zeros(ComplexF64, 2, 2, 2, 2)
    vertex[1, 2, 1, 2] = 1.1
    vertex[2, 1, 1, 2] = -1.1
    vertex[1, 2, 2, 1] = -1.1
    vertex[2, 1, 2, 1] = 1.1

    return (
        bare=FullCoulombInteraction(bare, BareCoulombTensor(), layout),
        antisymmetrized=FullCoulombInteraction(
            vertex, AntisymmetrizedVertex(), layout,
        ),
    )
end

function _ttnob_assert_provenance_compression(direct_builder, lowered)
    provenance = lowered.build_provenance
    witnesses = [
        witness
        for edge in lowered.compression.edges
        for witness in edge.witnesses
    ]
    if !isempty(provenance.relations)
        @test any(witness -> witness.source === :provenance, witnesses)
        return
    end

    spin = spin_ops()
    topology = mps_topology(3)
    physical = Dict(Symbol(:site, index) => spin.P for index in 1:3)
    hamiltonian =
        OpSum() +
        Term(0.7, SiteOp(:site1, :X, spin.X), SiteOp(:site2, :Z, spin.Z)) +
        Term(-0.31, SiteOp(:site1, :X, spin.X), SiteOp(:site2, :Z, spin.Z)) +
        Term(0.4, SiteOp(:site2, :X, spin.X), SiteOp(:site3, :X, spin.X))
    operator, report, fallback_provenance = build_ttno(
        direct_builder, hamiltonian, topology, physical,
    )
    @test report.plan_kind === :direct_sum
    @test fallback_provenance isa TTNOExactProvenance
    @test !isempty(fallback_provenance.relations)
    reference = to_dense(operator)
    compression = compress!(
        operator; sector_aware=true, mode=:exact_rank,
        compression_atol=1e-12, provenance=fallback_provenance,
    )
    fallback_witnesses = [
        witness
        for edge in compression.edges
        for witness in edge.witnesses
    ]
    @test any(witness -> witness.source === :provenance, fallback_witnesses)
    @test to_dense(operator) ≈ reference atol=1e-10 rtol=1e-10
end

@testset "typed TTNO builder policy and Anderson lowering" begin
    legacy = LegacyTTNOBuilder()
    direct = CompiledTTNOBuilder(merge=DirectSumMerge())
    direct_copy = CompiledTTNOBuilder(
        lowering=AbelianScalarLowering(), merge=DirectSumMerge(),
    )
    sge = CompiledTTNOBuilder()
    sge_copy = CompiledTTNOBuilder(
        lowering=AbelianScalarLowering(),
        merge=StateDiagramMerge(SGEOptimizer()),
    )

    @test direct.lowering isa AbelianScalarLowering
    @test direct.merge isa DirectSumMerge
    @test sge.lowering isa AbelianScalarLowering
    @test sge.merge isa StateDiagramMerge{SGEOptimizer}
    @test direct == direct_copy
    @test isequal(direct, direct_copy)
    @test hash(direct) == hash(direct_copy)
    @test sge == sge_copy
    @test hash(sge) == hash(sge_copy)
    @test direct != sge
    @test hash(direct) != hash(sge)

    fixture = _ttnob_anderson_fixture()
    default_lowered = lower_hamiltonian(
        fixture.mounted, fixture.interaction, fixture.operators;
        h_loc=fixture.h_loc, compression_atol=1e-12,
    )
    lowered = [
        _ttnob_lower(fixture, legacy),
        _ttnob_lower(fixture, direct),
        _ttnob_lower(fixture, sge),
    ]
    @test default_lowered.builder isa LegacyTTNOBuilder
    @test isnothing(default_lowered.build_report)
    @test isnothing(default_lowered.build_provenance)
    @test to_dense(default_lowered.operator) ≈
        to_dense(first(lowered).operator) atol=1e-10 rtol=1e-10

    @test first(lowered).builder == legacy
    @test isnothing(first(lowered).build_report)
    @test isnothing(first(lowered).build_provenance)
    @test lowered[2].builder == direct
    @test lowered[2].build_report isa TTNOBuildReport
    @test lowered[2].build_report.plan_kind === :direct_sum
    @test lowered[2].build_provenance isa TTNOExactProvenance
    @test lowered[3].builder == sge
    @test lowered[3].build_report isa TTNOBuildReport
    @test lowered[3].build_report.plan_kind === :state_diagram
    @test isnothing(lowered[3].build_provenance)

    _ttnob_compare_lowered(lowered; seed=20260731)
    _ttnob_assert_provenance_compression(direct, lowered[2])
end

@testset "FullCoulomb convention x TTNO builder matrix" begin
    fixture = _ttnob_anderson_fixture()
    interactions = _ttnob_full_coulomb_conventions(fixture.layout)
    builders = (
        legacy=LegacyTTNOBuilder(),
        direct_sum=CompiledTTNOBuilder(merge=DirectSumMerge()),
        state_diagram=CompiledTTNOBuilder(),
    )

    lowered = LoweredImpurityHamiltonian[]
    for (convention_name, interaction) in pairs(interactions)
        convention_results = LoweredImpurityHamiltonian[]
        for (builder_name, builder) in pairs(builders)
            result = lower_hamiltonian(
                fixture.mounted, interaction, fixture.operators;
                h_loc=fixture.h_loc, ttno_builder=builder,
                compression_atol=1e-12,
            )
            push!(convention_results, result)
            push!(lowered, result)

            if builder_name === :legacy
                @test isnothing(result.build_report)
                @test isnothing(result.build_provenance)
            elseif builder_name === :direct_sum
                @test result.build_report isa TTNOBuildReport
                @test result.build_report.plan_kind === :direct_sum
                @test result.build_provenance isa TTNOExactProvenance
            else
                @test result.build_report isa TTNOBuildReport
                @test result.build_report.plan_kind === :state_diagram
                @test isnothing(result.build_provenance)
            end
        end

        convention_dense = to_dense.([result.operator for result in convention_results])
        @test all(
            matrix -> isapprox(
                matrix, first(convention_dense); atol=1e-10, rtol=1e-10,
            ),
            convention_dense[2:end],
        )
        @test convention_name in (:bare, :antisymmetrized)
        _ttnob_assert_provenance_compression(
            builders.direct_sum, convention_results[2],
        )
    end

    matrices = _ttnob_compare_lowered(lowered; seed=20260802)
    @test all(
        matrix -> isapprox(matrix, first(matrices); atol=1e-10, rtol=1e-10),
        matrices[2:end],
    )
end

@testset "typed TTNO builders preserve Cayley-chain action" begin
    fixture = _ttnob_cayley_fixture()
    @test length(fixture.mapping.mapped.edges) == 1
    builders = (
        LegacyTTNOBuilder(),
        CompiledTTNOBuilder(merge=DirectSumMerge()),
        CompiledTTNOBuilder(),
    )
    lowered = [_ttnob_lower(fixture, builder) for builder in builders]
    @test lowered[2].build_report.plan_kind === :direct_sum
    @test lowered[2].build_provenance isa TTNOExactProvenance
    @test lowered[3].build_report.plan_kind === :state_diagram
    @test isnothing(lowered[3].build_provenance)
    _ttnob_compare_lowered(lowered; seed=20260801)
end

@testset "compiled TTNO builder fails closed on unsupported category" begin
    physical_space = Vect[SU2Irrep](SU2Irrep(0) => 1, SU2Irrep(1 // 2) => 1)
    topology = mps_topology(2)
    hamiltonian = OpSum() + Term(
        1.0, SiteOp(:site1, :I, id(physical_space)),
    )
    error = try
        build_ttno(
            CompiledTTNOBuilder(), hamiltonian, topology,
            Dict(:site1 => physical_space, :site2 => physical_space),
        )
        nothing
    catch caught
        caught
    end
    @test error isa TTNOBuilderCapabilityError
    @test error.cause isa MissingCategoryCapability
    @test occursin("LegacyTTNOBuilder()", sprint(showerror, error))
end

function _ttnob_solver(builder=nothing)
    layout = FlavorLayout(
        [:d], Dict(:d => :imp), Dict(:imp => [:d]);
        basis=:ttno_builder_solver,
    )
    partition = Partition(:d => [:d])
    orbitals = BathOrbitals(
        [0.2], [ComplexF64[0.4]], [1], [1], [:d]; layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    problem = ImpurityProblem(
        bath, ImpurityOneBody(zeros(ComplexF64, 1, 1), layout),
        DensityDensityInteraction(zeros(ComplexF64, 1, 1), layout),
    )
    arguments = (topology_plan=T3NS(layout), compression_atol=1e-12)
    solver = builder === nothing ?
        TTNSSolver(; arguments...) :
        TTNSSolver(; arguments..., ttno_builder=builder)
    mounted = mount_bath(impurity_topology(T3NS(layout), partition, bath), bath;
                         sector=ParticleNumberSector())
    physical = _ttnob_physical(mounted)
    vacuum = FermionParity(0) ⊠ U1Irrep(0)
    initial = product_ttns(
        ComplexF64, mounted.topology, physical,
        Dict(site => vacuum for site in keys(physical)),
    )
    target = TargetIrrep(only(symmetry_actions(problem.symmetry)), U1Irrep(0))
    return solver, problem, target, initial
end

@testset "TTNSSolver TTNO builder identity is warm-start relevant" begin
    explicit_builder = CompiledTTNOBuilder(merge=DirectSumMerge())
    default_solver, default_problem, default_target, default_initial = _ttnob_solver()
    explicit_solver, explicit_problem, explicit_target, explicit_initial =
        _ttnob_solver(explicit_builder)
    @test default_solver.ttno_builder isa LegacyTTNOBuilder
    @test explicit_solver.ttno_builder == explicit_builder

    default_request = TTNSSolveRequest(default_target; ground_state=GroundStateRequest(
        trunc=TruncationScheme(maxdim=4),
        nsweeps=1,
        tolerance=1e-10,
        krylovdim=4,
        verbose=false,
    ))
    explicit_request = TTNSSolveRequest(explicit_target;
                                        ground_state=default_request.ground_state)
    default_result = solve!(
        TTNSWorkspace(), default_solver, default_problem, default_request;
        initial_state=default_initial,
    )
    explicit_result = solve!(
        TTNSWorkspace(), explicit_solver, explicit_problem, explicit_request;
        initial_state=explicit_initial,
    )
    @test default_result.lowered.builder isa LegacyTTNOBuilder
    @test explicit_result.lowered.builder == explicit_builder
    @test default_result.warm_identity != explicit_result.warm_identity
end
