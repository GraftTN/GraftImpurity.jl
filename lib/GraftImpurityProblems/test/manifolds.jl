struct OpaqueCategoryA end
struct OpaqueCategoryB end

struct OpaqueIrrep
    label::Symbol
end

struct OpaqueActionSemantics <: AbstractPhysicalActionSemantics
    generator::Symbol
    parameters::Tuple
end

struct NestedSemanticWrapper{T}
    payload::T
end

struct NestedOpaqueSemantics{T} <: AbstractPhysicalActionSemantics
    descriptor::T
end

struct OpaqueAction
    identity::SymmetryActionIdentity
end

GraftImpurityProblems.action_identity(action::OpaqueAction) = action.identity

@testset "action-bound target manifolds" begin
    problem = problem_fixture()
    charge = only(symmetry_actions(problem.symmetry))
    identity = action_identity(charge)

    @test identity.action === :charge
    @test action_layout(identity) == problem_layout(problem)
    @test category_product(identity) == (U1Irrep,)
    @test action_semantics(identity) == ChargeU1ActionSemantics((1, 1))
    @test_throws MethodError SymmetryActionIdentity(
        :charge, problem_layout(problem), (U1Irrep,),
    )

    target = TargetIrrep(charge, U1Irrep(2))
    scan = IrrepScan(charge, (U1Irrep(0), U1Irrep(1), U1Irrep(2)))
    @test validate_manifold(problem, target) === target
    @test validate_manifold(problem, scan) === scan
    @test manifold_targets(target) == (U1Irrep(2),)
    @test manifold_targets(scan) ==
        (U1Irrep(0), U1Irrep(1), U1Irrep(2))
    @test manifold_identity(target) ==
        manifold_identity(TargetIrrep(charge, U1Irrep(2)))
    @test_throws ArgumentError IrrepScan(charge, ())
    @test_throws ArgumentError IrrepScan(
        charge, (U1Irrep(0), U1Irrep(0)),
    )

    basis = problem_layout(problem)
    opaque_identity = SymmetryActionIdentity(
        :opaque, basis, (OpaqueCategoryA, OpaqueCategoryB),
        OpaqueActionSemantics(:diagonal, (1, -1)),
    )
    opaque_action = OpaqueAction(opaque_identity)
    opaque_declaration = ImpuritySymmetryDeclaration((opaque_action,))
    opaque_problem = ImpurityProblem(
        problem.bath, problem.h_loc, problem.interaction, opaque_declaration,
    )
    opaque_target = TargetIrrep(opaque_action, OpaqueIrrep(:same_outer))
    opaque_scan = IrrepScan(
        opaque_action, (OpaqueIrrep(:first), OpaqueIrrep(:second)),
    )
    @test validate_manifold(opaque_problem, opaque_target) === opaque_target
    @test validate_manifold(opaque_problem, opaque_scan) === opaque_scan
    @test manifold_targets(opaque_scan) ==
        (OpaqueIrrep(:first), OpaqueIrrep(:second))

    swapped = SymmetryActionIdentity(
        :opaque, basis, (OpaqueCategoryB, OpaqueCategoryA),
        OpaqueActionSemantics(:diagonal, (1, -1)),
    )
    wrong_order = TargetIrrep(swapped, OpaqueIrrep(:same_outer))
    @test_throws ArgumentError validate_manifold(opaque_problem, wrong_order)

    wrong_basis = SymmetryActionIdentity(
        :opaque, problem_fixture(basis=:other).bath.layout,
        (OpaqueCategoryA, OpaqueCategoryB),
        OpaqueActionSemantics(:diagonal, (1, -1)),
    )
    @test_throws ArgumentError validate_manifold(
        opaque_problem, TargetIrrep(wrong_basis, OpaqueIrrep(:same_outer)),
    )

    other_generator = SymmetryActionIdentity(
        :opaque, basis, (OpaqueCategoryA, OpaqueCategoryB),
        OpaqueActionSemantics(:diagonal, (1, 1)),
    )
    @test other_generator != opaque_identity
    @test hash(other_generator) != hash(opaque_identity)
    @test manifold_identity(TargetIrrep(other_generator, OpaqueIrrep(:same_outer))) !=
        manifold_identity(opaque_target)
    @test_throws ResponseReachabilityError validate_response_target(
        opaque_target,
        TargetIrrep(other_generator, OpaqueIrrep(:response)),
    )
    opaque_response = TargetIrrep(opaque_action, OpaqueIrrep(:response))
    @test validate_response_target(opaque_target, opaque_response) ===
        opaque_response
    @test_throws ResponseReachabilityError validate_response_reachability(
        :synthetic_backend, opaque_target, opaque_response, :right, :left,
    )

    nested_mutable = NestedOpaqueSemantics(
        NestedSemanticWrapper(([1, 2],)),
    )
    @test_throws ArgumentError SymmetryActionIdentity(
        :nested, basis, (OpaqueCategoryA,), nested_mutable,
    )
end

@testset "built-in physical action semantics" begin
    problem = problem_fixture()
    layout = problem_layout(problem)
    spin = FlavorU1(:spin, [1, -1], layout)
    rescaled = FlavorU1(:spin, [0.5, -0.5], layout)
    spin_identity = action_identity(spin)
    rescaled_identity = action_identity(rescaled)
    @test spin_identity != rescaled_identity
    @test action_semantics(spin_identity) ==
        FlavorU1ActionSemantics((1.0, -1.0))

    su2_spin = SU2Reduce(layout; axial_generator=spin)
    su2_rescaled = SU2Reduce(layout; axial_generator=rescaled)
    @test category_product(action_identity(su2_spin)) == (SU2Irrep,)
    @test action_identity(su2_spin) != action_identity(su2_rescaled)
    declaration = ImpuritySymmetryDeclaration((ChargeU1(layout), spin, su2_spin))
    @test length(symmetry_actions(declaration)) == 3
end
