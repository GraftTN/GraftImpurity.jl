using Test
using LinearAlgebra: diag
using GraftImpurityFoundations: FlavorLayout, ParticleNumberSector
using GraftImpurityInteractions

function two_flavor_layout(; basis=:canonical)
    return FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis,
    )
end

function kanamori_layout()
    ordered = [:a_up, :a_down, :b_up, :b_down]
    return FlavorLayout(
        ordered,
        Dict(flavor => :impurity for flavor in ordered),
        Dict(:impurity => ordered);
        basis=:kanamori,
    )
end

function opsum_signature(H)
    return [(ComplexF64(term.coeff),
             Tuple((operator.site, operator.name) for operator in term.ops))
            for term in H]
end

@testset "interaction construction and lowering" begin
    layout = two_flavor_layout()
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())

    density = DensityDensityInteraction(ComplexF64[0.4 2.0; 2.0 0.6], layout)
    @test density.layout === layout
    @test length(lower_interaction(density, operators, nothing)) == 3
    decomposition = split_density_density(density)
    @test decomposition.one_body.matrix == ComplexF64[0.4 0; 0 0.6]
    @test all(iszero, diag(decomposition.interaction.U))
    @test_throws ArgumentError DensityDensityInteraction(
        ComplexF64[0 1im; -1im 0], layout,
    )

    density_only = DensityDensityInteraction(ComplexF64[0 2; 2 0], layout)
    density_signature = opsum_signature(
        lower_interaction(density_only, operators, nothing),
    )

    bare = zeros(ComplexF64, 2, 2, 2, 2)
    bare[1, 2, 1, 2] = 2
    bare[2, 1, 2, 1] = 2
    full_bare = FullCoulombInteraction(bare, BareCoulombTensor(), layout)
    @test opsum_signature(lower_interaction(full_bare, operators, nothing)) ==
        density_signature

    vertex = zeros(ComplexF64, 2, 2, 2, 2)
    vertex[1, 2, 1, 2] = 2
    vertex[2, 1, 1, 2] = -2
    vertex[1, 2, 2, 1] = -2
    vertex[2, 1, 2, 1] = 2
    full_vertex = FullCoulombInteraction(
        vertex, AntisymmetrizedVertex(), layout,
    )
    @test opsum_signature(lower_interaction(full_vertex, operators, nothing)) ==
        density_signature
    @test_throws ArgumentError FullCoulombInteraction(
        bare, AntisymmetrizedVertex(), layout,
    )

    k_layout = kanamori_layout()
    flavor_map = KanamoriFlavorMap(
        k_layout, [(:a_up, :a_down), (:b_up, :b_down)],
    )
    density_kanamori = KanamoriInteraction(
        4.0, 2.0, 0.5, k_layout;
        flavor_map, spin_flip=false, pair_hopping=false,
    )
    full_kanamori = KanamoriInteraction(
        4.0, 2.0, 0.5, k_layout;
        flavor_map, spin_flip=true, pair_hopping=true,
    )
    k_operators = ImpurityOperators(k_layout; sector=ParticleNumberSector())
    @test density_kanamori.terms == KanamoriTerms(false, false)
    @test full_kanamori.terms == KanamoriTerms(true, true)
    @test length(lower_interaction(density_kanamori, k_operators, nothing)) == 6
    @test length(lower_interaction(full_kanamori, k_operators, nothing)) == 10
    @test_throws ArgumentError KanamoriFlavorMap(
        k_layout, [(:a_up, :a_down), (:b_up, :a_down)],
    )
end

@testset "one-body terms and basis rotation" begin
    layout = two_flavor_layout()
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    onebody = ImpurityOneBody(
        ComplexF64[0.2 0.1im; -0.1im -0.3], layout; label=:local,
    )
    lowered = lower_one_body(onebody, operators, nothing)
    @test length(lowered) == 4
    @test opsum_signature(
        one_body_opsum(onebody, nothing, operators, nothing),
    ) == opsum_signature(lowered)
    @test_throws ArgumentError ImpurityOneBody(ComplexF64[0 1; 0 0], layout)

    rotated_layout = two_flavor_layout(basis=:rotated)
    rotation = ComplexF64[1 1; 1 -1] / sqrt(2)
    rotated_onebody = rotate_one_body(onebody, rotation, rotated_layout)
    @test rotated_onebody.layout === rotated_layout
    @test rotated_onebody.label === :local
    @test rotated_onebody.matrix ≈ rotation' * onebody.matrix * rotation
    @test_throws ArgumentError rotate_one_body(
        onebody, ComplexF64[1 1; 0 1], rotated_layout,
    )

    density = DensityDensityInteraction(ComplexF64[0 2; 2 0], layout)
    rotated_interaction = rotate_interaction(density, rotation, rotated_layout)
    @test rotated_interaction isa FullCoulombInteraction
    @test rotated_interaction.convention isa AntisymmetrizedVertex
    @test rotated_interaction.layout === rotated_layout
    @test sum(abs, rotated_interaction.U) > 0
end

@testset "symmetry audit" begin
    layout = two_flavor_layout(basis=:symmetry)
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    spin_z = FlavorU1(:spin_z, [1.0, -1.0], layout)
    spec = SymmetrySpec(
        layout;
        abelian=(ChargeU1(layout), spin_z),
        nonabelian=(SU2Reduce(layout; name=:spin_su2, axial_generator=spin_z),),
    )

    diagonal = ImpurityOneBody(ComplexF64[0.2 0; 0 -0.1], layout)
    preserved = audit_symmetry(
        one_body_opsum(diagonal, nothing, operators, spec), spec;
        hermiticity=:certified,
    )
    @test preserved.hermiticity === :certified
    @test all(item -> item.status === :preserved, preserved.abelian)
    @test only(preserved.nonabelian).status === :candidate
    @test only(preserved.nonabelian).lowering_status === :unsupported

    mixing = ImpurityOneBody(ComplexF64[0 0.25; 0.25 0], layout)
    broken = audit_symmetry(
        one_body_opsum(mixing, nothing, operators, spec), spec,
    )
    @test broken.abelian[1].status === :preserved
    @test broken.abelian[2].status === :broken
    @test only(broken.nonabelian).status === :broken
    @test_throws ArgumentError SymmetrySpec(
        layout; abelian=(ChargeU1(layout), FlavorU1(:charge, [1, 1], layout)),
    )
end
