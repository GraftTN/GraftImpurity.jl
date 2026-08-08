using Test
using LinearAlgebra: inv
using GraftImpurityFoundations
using GraftImpurityInteractions
using GraftImpurityBaths
using GraftImpurityBathFit
import GraftImpurityProblems
using GreenFunc

struct _PreparationSyntheticKernel <: AbstractRealPoleBathFitKernel
    energy::Float64
    residue::ComplexF64
end

mutable struct _PreparationMutableKernel <: AbstractRealPoleBathFitKernel
    energy::Vector{Float64}
    residue::Vector{ComplexF64}
end

struct _PreparationExternalInteraction <: AbstractImpurityInteraction
    basis::FlavorLayout
    coefficients::Vector{Float64}
end

GraftImpurityFoundations.interaction_layout(
    interaction::_PreparationExternalInteraction,
) = interaction.basis

GraftImpurityFoundations.interaction_identity(
    interaction::_PreparationExternalInteraction,
) = (
    family=:preparation_external,
    basis=interaction.basis,
    coefficients=Tuple(interaction.coefficients),
)

mutable struct _PreparationMutableSymmetry <:
        GraftImpurityProblems.AbstractImpuritySymmetryDeclaration
    basis::FlavorLayout
    weights::Vector{Float64}
end

GraftImpurityProblems.symmetry_layout(symmetry::_PreparationMutableSymmetry) =
    symmetry.basis

function GraftImpurityFoundations.real_pole_bath_fit(
        input::BathFitInput, kernel::_PreparationSyntheticKernel,
        partition::Partition)
    plan = DiscretizationPlan(
        :d => BlockDiscretizationPlan([SpectralInterval(-2.0, 2.0, 1)]);
        shared_grid=true,
    )
    poles = BlockRealPoles(
        input.layout, partition, [kernel.energy], [kernel.residue], [1];
        statistics=:fermion,
    )
    return PoleExpansion(
        poles; kernel=:preparation_synthetic, trace=(; plan),
    )
end

function GraftImpurityFoundations.real_pole_bath_fit(
        input::BathFitInput, kernel::_PreparationMutableKernel,
        partition::Partition)
    plan = DiscretizationPlan(
        :d => BlockDiscretizationPlan([SpectralInterval(-2.0, 2.0, 1)]);
        shared_grid=true,
    )
    poles = BlockRealPoles(
        input.layout, partition, [only(kernel.energy)], [only(kernel.residue)], [1];
        statistics=:fermion,
    )
    return PoleExpansion(
        poles; kernel=:preparation_mutable_synthetic, trace=(; plan),
    )
end

function _preparation_fixture()
    layout = FlavorLayout(
        [:d], Dict(:d => :imp), Dict(:imp => [:d]);
        basis=:preparation_fixture,
    )
    partition = Partition(:d => [:d])
    h_loc = ImpurityOneBody(zeros(ComplexF64, 1, 1), layout)
    interaction = DensityDensityInteraction(zeros(ComplexF64, 1, 1), layout)
    symmetry = GraftImpurityProblems.ImpuritySymmetryDeclaration(
        GraftImpurityInteractions.ChargeU1(layout),
    )
    return (; layout, partition, h_loc, interaction, symmetry)
end

function _preparation_hybridization_gf(; beta=10.0, energy=0.2, residue=0.16)
    mesh = ImFreq(beta, true; grid=[-2, -1, 0, 1, 2])
    data = ComplexF64[
        residue / (im * mesh[index] - energy) for index in eachindex(mesh)
    ]
    return Gf(mesh; data, statistics=true, component=:matsubara)
end

function _preparation_weiss_gf(;
        beta=10.0, energy=0.2, residue=0.16, local_energy=0.0)
    mesh = ImFreq(beta, true; grid=[-2, -1, 0, 1, 2])
    data = ComplexF64[]
    for index in eachindex(mesh)
        delta = residue / (im * mesh[index] - energy)
        push!(data, inv(im * mesh[index] - local_energy - delta))
    end
    return Gf(mesh; data, statistics=true, component=:matsubara)
end

@testset "immutable impurity preparation" begin
    fixture = _preparation_fixture()
    delta = _preparation_hybridization_gf()
    original = copy(delta.data)
    direct = HybridizationPreparationInput(
        delta, fixture.partition; h_loc=fixture.h_loc,
    )
    @test direct isa AbstractImpurityPreparationInput
    @test direct.source === direct.hybridization
    @test direct.source.source_template isa GreenFunc.Gf
    @test direct.source.source_template !== delta
    delta.data[1] = 99.0 + 0.0im
    @test direct.source.source_template.data == original
    caller_h_loc = ImpurityOneBody(
        reshape(ComplexF64[0.3], 1, 1), fixture.layout,
    )
    owned_h_loc_direct = HybridizationPreparationInput(
        _preparation_hybridization_gf(), fixture.partition;
        h_loc=caller_h_loc,
    )
    @test owned_h_loc_direct.h_loc !== caller_h_loc
    caller_h_loc.matrix[1, 1] = 7.0
    @test owned_h_loc_direct.h_loc.matrix[1, 1] == 0.3
    raw_input = BathFitInput(
        fixture.layout, [-1.0, 0.0, 1.0],
        :d => ComplexF64[0.0, 0.5, 0.0];
        domain=:real_axis, statistics=:fermion,
    )
    raw_direct = HybridizationPreparationInput(
        raw_input, fixture.partition; h_loc=fixture.h_loc,
    )
    @test raw_direct.source !== raw_input
    raw_input.blocks.d[1][1, 1] = 17.0
    @test raw_direct.source.blocks.d[1][1, 1] == 0.0
    block_direct = HybridizationPreparationInput(
        BlockGf(:d => _preparation_hybridization_gf()), fixture.partition;
        h_loc=fixture.h_loc,
    )
    @test block_direct.source.source_template isa GreenFunc.BlockGf
    replacement = HybridizationPreparationInput(
        _preparation_hybridization_gf(residue=0.25), fixture.partition;
        h_loc=fixture.h_loc,
    )
    @test replacement !== direct
    @test direct.source.source_template.data == original

    weiss = WeissPreparationInput(
        _preparation_weiss_gf(), fixture.partition; h_loc=fixture.h_loc,
    )
    @test weiss.source !== weiss.hybridization
    @test weiss.source.source_template isa GreenFunc.Gf
    @test weiss.hybridization.source_template isa GreenFunc.Gf
    @test weiss.hybridization.metadata.weiss_conversion === :explicit_inverse
    expected = ComplexF64[
        0.16 / (im * frequency - 0.2)
        for frequency in weiss.hybridization.frequencies
    ]
    @test [only(sample) for sample in weiss.hybridization.blocks.d] ≈ expected

    caller_weiss_h_loc = ImpurityOneBody(
        reshape(ComplexF64[0.35], 1, 1), fixture.layout,
    )
    direct_consistent = HybridizationPreparationInput(
        _preparation_hybridization_gf(), fixture.partition;
        h_loc=caller_weiss_h_loc,
    )
    weiss_consistent = WeissPreparationInput(
        _preparation_weiss_gf(local_energy=0.35), fixture.partition;
        h_loc=caller_weiss_h_loc,
    )
    caller_weiss_h_loc.matrix[1, 1] = 8.0
    @test direct_consistent.h_loc.matrix[1, 1] == 0.35
    @test weiss_consistent.h_loc.matrix[1, 1] == 0.35
    @test weiss_consistent.h_loc !== caller_weiss_h_loc
    @test all(
        isapprox(weiss_sample, direct_sample)
        for (weiss_sample, direct_sample) in zip(
            weiss_consistent.hybridization.blocks.d,
            direct_consistent.hybridization.blocks.d,
        )
    )

    near_singular = Gf(
        ImFreq(10.0, true; grid=[0]); data=ComplexF64[1e-320],
        statistics=true, component=:matsubara,
    )
    @test_throws ArgumentError WeissPreparationInput(
        near_singular, fixture.partition; h_loc=fixture.h_loc,
    )
    boson = Gf(
        ImFreq(10.0, false; grid=[0, 1]); data=ComplexF64[1, 1],
        statistics=false, component=:matsubara,
    )
    @test_throws ArgumentError HybridizationPreparationInput(
        boson, fixture.partition; h_loc=fixture.h_loc,
    )

    multi_layout = FlavorLayout(
        [:a, :b], Dict(:a => :imp_a, :b => :imp_b),
        Dict(:imp_a => [:a], :imp_b => [:b]); basis=:preparation_multiblock,
    )
    multi_partition = Partition(:a => [:a], :b => [:b])
    cross_block = ImpurityOneBody(
        ComplexF64[0 0.1; 0.1 0], multi_layout,
    )
    @test_throws ArgumentError WeissPreparationInput(
        BlockGf(:a => _preparation_weiss_gf(),
                :b => _preparation_weiss_gf()),
        multi_partition; h_loc=cross_block,
    )

    criteria = BathFitCriteria(require_mountable=true)
    policy = ImpurityPreparationPolicy(
        _PreparationSyntheticKernel(0.2, 0.16 + 0.0im), criteria,
    )
    @test policy.criteria == criteria
    @test_throws ArgumentError ImpurityPreparationPolicy(
        policy.kernel, BathFitCriteria(),
    )

    caller_kernel = _PreparationMutableKernel([0.2], [0.16 + 0.0im])
    mutable_policy = ImpurityPreparationPolicy(caller_kernel, criteria)
    @test mutable_policy.kernel !== caller_kernel
    caller_kernel.energy[1] = 9.0
    caller_kernel.residue[1] = 9.0 + 0.0im
    @test only(mutable_policy.kernel.energy) == 0.2
    @test only(mutable_policy.kernel.residue) == 0.16 + 0.0im

    prepared = prepare_impurity_problem(
        direct, fixture.interaction, fixture.symmetry, policy,
    )
    @test prepared isa PreparedImpurityProblem
    @test prepared isa AbstractImpurityPreparationOutcome
    @test prepared.problem isa GraftImpurityProblems.ImpurityProblem
    @test prepared.problem.bath === prepared.provenance.realization.bath
    @test prepared.problem.h_loc === prepared.provenance.input.h_loc
    @test prepared.problem.interaction === prepared.provenance.interaction
    @test prepared.problem.symmetry === prepared.provenance.symmetry
    @test prepared.problem.h_loc == fixture.h_loc
    @test prepared.problem.interaction == fixture.interaction
    @test prepared.problem.symmetry == fixture.symmetry
    @test prepared.provenance.input !== direct
    @test prepared.provenance.expansion ===
          prepared.provenance.realization.expansion
    @test prepared.provenance.realization.report.source ===
          prepared.provenance.input.hybridization
    @test prepared.provenance.realization_options.atol == 0.0
    @test prepared.provenance.realization_options.rtol == sqrt(eps(Float64))
    @test prepared.provenance.audit.passed

    density = DensityDensityInteraction(
        zeros(ComplexF64, 1, 1), fixture.layout,
    )
    mutable_symmetry = _PreparationMutableSymmetry(fixture.layout, [1.0])
    mutable_input = HybridizationPreparationInput(
        _preparation_hybridization_gf(), fixture.partition;
        h_loc=fixture.h_loc,
    )
    mutable_prepared = prepare_impurity_problem(
        mutable_input, density, mutable_symmetry, mutable_policy,
    )
    @test mutable_prepared isa PreparedImpurityProblem
    @test mutable_prepared.provenance.input !== mutable_input
    @test mutable_prepared.provenance.interaction !== density
    @test mutable_prepared.provenance.symmetry !== mutable_symmetry
    @test mutable_prepared.provenance.policy.kernel !== mutable_policy.kernel
    @test mutable_prepared.problem.h_loc ===
          mutable_prepared.provenance.input.h_loc
    @test mutable_prepared.problem.interaction ===
          mutable_prepared.provenance.interaction
    @test mutable_prepared.problem.symmetry ===
          mutable_prepared.provenance.symmetry
    mutable_input.h_loc.matrix[1, 1] = 4.0
    mutable_input.source.blocks.d[1][1, 1] = 5.0
    density.U[1, 1] = 6.0
    mutable_symmetry.weights[1] = 7.0
    mutable_policy.kernel.energy[1] = 8.0
    mutable_policy.kernel.residue[1] = 8.0 + 0.0im
    @test mutable_prepared.problem.h_loc.matrix[1, 1] == 0.0
    @test mutable_prepared.provenance.input.source.blocks.d[1][1, 1] != 5.0
    @test mutable_prepared.problem.interaction.U[1, 1] == 0.0
    @test only(mutable_prepared.problem.symmetry.weights) == 1.0
    @test only(mutable_prepared.provenance.policy.kernel.energy) == 0.2
    @test only(mutable_prepared.provenance.policy.kernel.residue) == 0.16 + 0.0im

    full = FullCoulombInteraction(
        zeros(ComplexF64, 1, 1, 1, 1), BareCoulombTensor(), fixture.layout,
    )
    full_prepared = prepare_impurity_problem(
        direct, full, fixture.symmetry, policy,
    )
    full.U[1, 1, 1, 1] = 11.0
    @test full_prepared.problem.interaction.U[1, 1, 1, 1] == 0.0
    @test full_prepared.problem.interaction ===
          full_prepared.provenance.interaction

    external = _PreparationExternalInteraction(fixture.layout, [0.25])
    @test !hasproperty(external, :layout)
    external_prepared = prepare_impurity_problem(
        direct, external, fixture.symmetry, policy,
    )
    @test external_prepared isa PreparedImpurityProblem
    @test external_prepared.problem.interaction isa
          _PreparationExternalInteraction
    external.coefficients[1] = 12.0
    @test only(external_prepared.problem.interaction.coefficients) == 0.25

    direct_prepared = prepare_impurity_problem(
        direct_consistent, fixture.interaction, fixture.symmetry, policy,
    )
    weiss_prepared = prepare_impurity_problem(
        weiss_consistent, fixture.interaction, fixture.symmetry, policy,
    )
    @test direct_prepared.problem.h_loc == weiss_prepared.problem.h_loc
    @test direct_prepared.problem.bath == weiss_prepared.problem.bath

    horizon_policy = ImpurityPreparationPolicy(
        policy.kernel,
        BathFitCriteria(
            beta=10.0, request_horizon=0.1, require_mountable=true,
        ),
    )
    horizon_prepared = prepare_impurity_problem(
        direct, fixture.interaction, fixture.symmetry, horizon_policy,
    )
    @test horizon_prepared isa PreparedImpurityProblem
    @test horizon_prepared.provenance.policy.criteria.beta == 10.0
    @test horizon_prepared.provenance.policy.criteria.request_horizon == 0.1
    @test !horizon_prepared.provenance.audit.passed

    nonmountable_policy = ImpurityPreparationPolicy(
        _PreparationSyntheticKernel(0.2, 1.0 + 0.1im), criteria,
    )
    nonmountable = prepare_impurity_problem(
        direct, fixture.interaction, fixture.symmetry, nonmountable_policy,
    )
    @test nonmountable isa NonMountableImpurityPreparation
    @test nonmountable isa AbstractImpurityPreparationOutcome
    @test nonmountable.provenance.realization isa NonMountablePoleFit
    @test nonmountable.provenance.expansion ===
          nonmountable.provenance.realization.expansion
    @test !nonmountable.provenance.audit.passed
    @test !hasproperty(nonmountable, :problem)
    @test !(nonmountable isa AbstractImpuritySolveResult)

    loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
    @test "GraftTTNSSolver" ∉ loaded
end
