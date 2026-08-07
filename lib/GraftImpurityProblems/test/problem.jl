struct _MissingIdentityInteraction <: AbstractImpurityInteraction
    basis::FlavorLayout
end

GraftImpurityFoundations.interaction_layout(
    interaction::_MissingIdentityInteraction,
) = interaction.basis

struct _BadIdentityInteraction <: AbstractImpurityInteraction
    basis::FlavorLayout
end

GraftImpurityFoundations.interaction_layout(
    interaction::_BadIdentityInteraction,
) = interaction.basis
GraftImpurityFoundations.interaction_identity(
    ::_BadIdentityInteraction,
) = (family=:bad_extension, coefficients=[1.0])

struct _NestedBadIdentity{T}
    payload::T
end

struct _NestedBadIdentityInteraction <: AbstractImpurityInteraction
    basis::FlavorLayout
end

GraftImpurityFoundations.interaction_layout(
    interaction::_NestedBadIdentityInteraction,
) = interaction.basis
GraftImpurityFoundations.interaction_identity(
    ::_NestedBadIdentityInteraction,
) = (family=:bad_nested, descriptor=_NestedBadIdentity(([1.0],)))

@testset "canonical impurity problem" begin
    problem = problem_fixture()
    equivalent = problem_fixture()

    @test fieldnames(typeof(problem)) == (:bath, :h_loc, :interaction, :symmetry)
    @test problem_layout(problem) == problem.bath.layout
    @test problem_partition(problem) == problem.bath.partition
    @test problem_statistics(problem) === :fermion
    @test problem == equivalent
    @test hash(problem) == hash(equivalent)
    @test problem_identity(problem) == problem_identity(equivalent)
    @test only(symmetry_actions(problem.symmetry)) isa ChargeU1
    @test only(symmetry_action_identities(problem.symmetry)) ==
        action_identity(only(symmetry_actions(problem.symmetry)))

    forbidden = (
        :layout, :partition, :gf_struct, :source, :fit, :topology, :mounted,
        :mapping, :opsum, :ttno, :ttns, :ci, :workspace, :request, :result,
    )
    @test all(name -> name ∉ fieldnames(typeof(problem)), forbidden)

    copied = copy(problem)
    @test copied == problem
    @test copied !== problem
    @test copied.bath.orbitals.energies !== problem.bath.orbitals.energies
    @test copied.bath.orbitals.couplings !== problem.bath.orbitals.couplings
    @test copied.bath.orbitals.couplings[1] !== problem.bath.orbitals.couplings[1]
    @test copied.h_loc.matrix !== problem.h_loc.matrix
    @test copied.interaction.U !== problem.interaction.U
    copied_identity = problem_identity(copied)
    copied.interaction.U[1, 2] = copied.interaction.U[2, 1] = 3.0
    @test problem_identity(copied) != copied_identity
    @test copied != problem

    buffer = IOBuffer()
    serialize(buffer, problem)
    seekstart(buffer)
    restored = deserialize(buffer)
    @test restored == problem
    @test problem_identity(restored) == problem_identity(problem)

    mismatched = problem_fixture(basis=:different_basis)
    @test_throws ArgumentError ImpurityProblem(
        problem.bath, mismatched.h_loc, problem.interaction, problem.symmetry,
    )
    @test_throws ArgumentError ImpurityProblem(
        problem.bath, problem.h_loc, mismatched.interaction, problem.symmetry,
    )
    @test_throws ArgumentError ImpurityProblem(
        problem.bath, problem.h_loc, problem.interaction, mismatched.symmetry,
    )

    invalid_partition = Partition(:down => [:down], :up => [:up])
    invalid_bath = DiscreteBath(
        problem.bath.layout, invalid_partition, problem.bath.orbitals,
        :fermion, Val(:validated),
    )
    @test_throws ArgumentError ImpurityProblem(
        invalid_bath, problem.h_loc, problem.interaction, problem.symmetry,
    )

    @test_throws MethodError ImpurityProblem(
        problem.bath, problem.h_loc,
        _MissingIdentityInteraction(problem_layout(problem)), problem.symmetry,
    )
    @test_throws ArgumentError ImpurityProblem(
        problem.bath, problem.h_loc,
        _BadIdentityInteraction(problem_layout(problem)), problem.symmetry,
    )
    @test_throws ArgumentError ImpurityProblem(
        problem.bath, problem.h_loc,
        _NestedBadIdentityInteraction(problem_layout(problem)), problem.symmetry,
    )
end
