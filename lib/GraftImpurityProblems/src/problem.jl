"""
    ImpurityProblem(bath, h_loc, interaction, symmetry)

Canonical finite impurity Hamiltonian with a topology-free physical symmetry
declaration. `bath` remains the sole owner of layout, partition, statistics,
and diagonal-star pole data.
"""
struct ImpurityProblem{
    B<:DiscreteBath,
    H<:ImpurityOneBody,
    I<:AbstractImpurityInteraction,
    S<:AbstractImpuritySymmetryDeclaration,
}
    bath::B
    h_loc::H
    interaction::I
    symmetry::S

    function ImpurityProblem(bath::B, h_loc::H, interaction::I, symmetry::S,
                             ::Val{:validated}) where {
                                 B<:DiscreteBath,H<:ImpurityOneBody,
                                 I<:AbstractImpurityInteraction,
                                 S<:AbstractImpuritySymmetryDeclaration}
        new{B,H,I,S}(bath, h_loc, interaction, symmetry)
    end
end

function ImpurityProblem(bath::DiscreteBath, h_loc::ImpurityOneBody,
                         interaction::AbstractImpurityInteraction,
                         symmetry::AbstractImpuritySymmetryDeclaration)
    layout = bath_layout(bath)
    validate_partition(bath_partition(bath), layout)
    one_body_layout(h_loc) == layout || throw(ArgumentError(
        "ImpurityProblem h_loc FlavorLayout must exactly match bath.layout",
    ))
    interaction_layout(interaction) == layout || throw(ArgumentError(
        "ImpurityProblem interaction FlavorLayout must exactly match bath.layout",
    ))
    identity = interaction_identity(interaction)
    _has_mutable_semantic_payload(identity) && throw(ArgumentError(
        "ImpurityProblem interaction_identity must contain only immutable physical data",
    ))
    symmetry_layout(symmetry) == layout || throw(ArgumentError(
        "ImpurityProblem symmetry FlavorLayout must exactly match bath.layout",
    ))
    return ImpurityProblem(bath, h_loc, interaction, symmetry, Val(:validated))
end

function ImpurityProblem(bath::DiscreteBath, h_loc::ImpurityOneBody,
                         interaction::AbstractImpurityInteraction)
    symmetry = ImpuritySymmetryDeclaration(ChargeU1(bath_layout(bath)))
    return ImpurityProblem(bath, h_loc, interaction, symmetry)
end

"""Authoritative impurity `FlavorLayout`, derived from the canonical bath."""
problem_layout(problem::ImpurityProblem) = bath_layout(problem.bath)

"""Authoritative named hybridization partition, derived from the bath."""
problem_partition(problem::ImpurityProblem) = bath_partition(problem.bath)

"""Particle statistics declared by the canonical bath."""
problem_statistics(problem::ImpurityProblem) = bath_statistics(problem.bath)

Base.:(==)(left::ImpurityProblem, right::ImpurityProblem) =
    left.bath == right.bath &&
    left.h_loc == right.h_loc &&
    interaction_identity(left.interaction) ==
        interaction_identity(right.interaction) &&
    left.symmetry == right.symmetry

Base.hash(problem::ImpurityProblem, seed::UInt) = hash(
    (problem.bath, problem.h_loc, interaction_identity(problem.interaction),
     problem.symmetry),
    hash(:ImpurityProblem, seed),
)

function Base.copy(problem::ImpurityProblem)
    return ImpurityProblem(
        copy(problem.bath), deepcopy(problem.h_loc),
        deepcopy(problem.interaction), deepcopy(problem.symmetry),
    )
end

"""Deterministic bounded identity fingerprint for a finite impurity problem."""
problem_identity(problem::ImpurityProblem) = hash(problem, zero(UInt))

"""
    validate_manifold(problem, manifold) -> manifold

Require the manifold's complete action/basis/category-product identity to be
declared by `problem`. The opaque target itself is never inspected or lowered.
"""
function validate_manifold(problem::ImpurityProblem,
                           manifold::AbstractImpurityManifold)
    identity = manifold_action_identity(manifold)
    identity in symmetry_action_identities(problem.symmetry) ||
        throw(ArgumentError(
            "impurity manifold action identity is not declared by the problem",
        ))
    return manifold
end
