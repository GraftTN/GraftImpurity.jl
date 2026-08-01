using Test
using GraftImpurity
using GraftImpurityBathFit
using GraftImpurityBaths
using GraftImpurityFoundations
using GraftImpurityInteractions
using GraftImpurityPoleFits
using GraftImpuritySolver
using GraftImpurityValidation
using GraftStateDiagram
using GraftTTNOBuild
using JLD2

const _UMBRELLA_OWNERS = (
    GraftImpurityFoundations,
    GraftImpurityInteractions,
    GraftImpurityBaths,
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
