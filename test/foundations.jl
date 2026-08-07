using Test
using GraftImpurity
using GraftImpurityBathFit
using GraftImpurityBaths
using GraftImpurityFoundations
using GraftImpurityInteractions
using GraftImpurityPoleFits
using GraftImpurityProblems
using GraftImpuritySolver
using GraftImpurityValidation
using GraftStateDiagram
using GraftTTNOBuild
using JLD2

const _UMBRELLA_OWNERS = (
    GraftImpurityFoundations,
    GraftImpurityInteractions,
    GraftImpurityBaths,
    GraftImpurityProblems,
    GraftImpurityPoleFits,
    GraftImpurityBathFit,
    GraftImpuritySolver,
    GraftImpurityValidation,
)

struct _UmbrellaFitKernel <: GraftImpurity.AbstractRealPoleBathFitKernel end

function GraftImpurity.real_pole_bath_fit(
        ::_UmbrellaFitKernel, marker::Val{:umbrella_extension})
    return marker
end

@testset "public binding identity" begin
    public_names = filter(!=(:GraftImpurity), names(GraftImpurity))
    @test !isempty(public_names)

    for name in public_names
        owners = filter(_UMBRELLA_OWNERS) do owner
            name in names(owner) && isdefined(owner, name)
        end
        @testset "$name" begin
            @test !isempty(owners)
            isempty(owners) && continue
            umbrella_binding = getfield(GraftImpurity, name)
            @test all(owner -> getfield(owner, name) === umbrella_binding, owners)
        end
    end
end

@testset "solver protocol umbrella surface" begin
    expected_bindings = (
        :AbstractImpuritySolver => GraftImpurityFoundations,
        :AbstractImpurityWorkspace => GraftImpurityFoundations,
        :AbstractImpuritySolveRequest => GraftImpurityFoundations,
        :AbstractImpuritySolveResult => GraftImpurityFoundations,
        :interaction_layout => GraftImpurityFoundations,
        :interaction_identity => GraftImpurityFoundations,
        :solve! => GraftImpurityFoundations,
        :AbstractImpuritySymmetryDeclaration => GraftImpurityProblems,
        :ImpuritySymmetryDeclaration => GraftImpurityProblems,
        :AbstractPhysicalActionSemantics => GraftImpurityProblems,
        :ChargeU1ActionSemantics => GraftImpurityProblems,
        :FlavorU1ActionSemantics => GraftImpurityProblems,
        :SU2ActionSemantics => GraftImpurityProblems,
        :SymmetryActionIdentity => GraftImpurityProblems,
        :action_identity => GraftImpurityProblems,
        :action_layout => GraftImpurityProblems,
        :action_semantics => GraftImpurityProblems,
        :category_product => GraftImpurityProblems,
        :symmetry_actions => GraftImpurityProblems,
        :symmetry_action_identities => GraftImpurityProblems,
        :symmetry_layout => GraftImpurityProblems,
        :AbstractImpurityManifold => GraftImpurityProblems,
        :TargetIrrep => GraftImpurityProblems,
        :IrrepScan => GraftImpurityProblems,
        :manifold_action_identity => GraftImpurityProblems,
        :manifold_targets => GraftImpurityProblems,
        :manifold_identity => GraftImpurityProblems,
        :validate_manifold => GraftImpurityProblems,
        :validate_response_target => GraftImpurityProblems,
        :ResponseReachabilityError => GraftImpurityProblems,
        :validate_response_reachability => GraftImpurityProblems,
        :ImpurityProblem => GraftImpurityProblems,
        :problem_layout => GraftImpurityProblems,
        :problem_partition => GraftImpurityProblems,
        :problem_statistics => GraftImpurityProblems,
        :problem_identity => GraftImpurityProblems,
        :AbstractImpurityPreparationInput => GraftImpurityBathFit,
        :AbstractImpurityPreparationOutcome => GraftImpurityBathFit,
        :HybridizationPreparationInput => GraftImpurityBathFit,
        :WeissPreparationInput => GraftImpurityBathFit,
        :ImpurityPreparationPolicy => GraftImpurityBathFit,
        :ImpurityPreparationProvenance => GraftImpurityBathFit,
        :PreparedImpurityProblem => GraftImpurityBathFit,
        :NonMountableImpurityPreparation => GraftImpurityBathFit,
        :prepare_impurity_problem => GraftImpurityBathFit,
        :TTNSSolver => GraftImpuritySolver,
        :TTNSWorkspace => GraftImpuritySolver,
        :TTNSSolveRequest => GraftImpuritySolver,
        :TTNSSolveResult => GraftImpuritySolver,
        :TTNSScanResult => GraftImpuritySolver,
        :TTNSCapabilityError => GraftImpuritySolver,
    )
    for (name, owner) in expected_bindings
        @test name in names(GraftImpurity)
        @test getfield(GraftImpurity, name) === getfield(owner, name)
    end

    removed_names = (
        :Solver, :SolveRequest, :ImpurityResult, :NonMountableImpurityResult,
        :TTNSNonMountableSolveResult, :set_weiss!, :set_hybridization!,
    )
    for name in removed_names
        @test name ∉ names(GraftImpurity)
        @test !isdefined(GraftImpurity, name)
    end
end

@testset "generic method extension through umbrella" begin
    @test GraftImpurity.real_pole_bath_fit ===
          GraftImpurityFoundations.real_pole_bath_fit
    marker = Val(:umbrella_extension)
    @test GraftImpurity.real_pole_bath_fit(_UmbrellaFitKernel(), marker) === marker
end

@testset "direct TTNO owners" begin
    @test GraftImpuritySolver.ttno_from_opsum === GraftTTNOBuild.ttno_from_opsum
    @test GraftImpuritySolver.compile_ttno === GraftStateDiagram.compile_ttno
end

@testset "private concrete serialization aliases" begin
    @test GraftImpurity._MountedHamiltonianCertificate ===
          GraftImpurityBaths._MountedHamiltonianCertificate
    @test GraftImpurity._DMFTMeasureSnapshot ===
          GraftImpurityBathFit._DMFTMeasureSnapshot
    @test GraftImpurity._DMFTPositiveMeasure ===
          GraftImpurityBathFit._DMFTPositiveMeasure

    # Written by the unsplit GraftImpurity module at Phase-1 commit 6f9931a.
    # Its JLD2 type paths are GraftImpurity._*, so this is the backward-
    # compatibility check that the umbrella aliases are intended to support.
    legacy_path = joinpath(
        @__DIR__, "data", "pre_split_private_types.jld2",
    )
    legacy = JLD2.load(legacy_path, "payload")
    @test legacy.certificate isa GraftImpurity._MountedHamiltonianCertificate
    @test legacy.certificate.hamiltonian_hash == 0x01
    @test legacy.certificate.parametrization_hash == 0x02
    @test legacy.snapshot isa GraftImpurity._DMFTMeasureSnapshot
    @test legacy.snapshot.blocks.spin isa GraftImpurity._DMFTPositiveMeasure
    @test legacy.snapshot.blocks.spin.energies == [0.25]
    @test legacy.snapshot.blocks.spin.weights == [0.75]
    @test legacy.snapshot.blocks.spin.mass == 0.75

    certificate = GraftImpurity._MountedHamiltonianCertificate(0x01, 0x02)
    measure = GraftImpurity._DMFTPositiveMeasure([0.25], [0.75], 0.75)
    snapshot = GraftImpurity._DMFTMeasureSnapshot((spin=measure,))
    payload = (; certificate, snapshot)

    mktempdir() do directory
        path = joinpath(directory, "umbrella_alias_roundtrip.jld2")
        JLD2.jldsave(path; payload)
        restored = JLD2.load(path, "payload")

        @test restored.certificate isa
              GraftImpurity._MountedHamiltonianCertificate
        @test restored.certificate.hamiltonian_hash ==
              certificate.hamiltonian_hash
        @test restored.certificate.parametrization_hash ==
              certificate.parametrization_hash
        @test restored.snapshot isa GraftImpurity._DMFTMeasureSnapshot
        @test restored.snapshot.blocks.spin isa
              GraftImpurity._DMFTPositiveMeasure
        @test restored.snapshot.blocks.spin.energies == measure.energies
        @test restored.snapshot.blocks.spin.weights == measure.weights
        @test restored.snapshot.blocks.spin.mass == measure.mass
    end
end
