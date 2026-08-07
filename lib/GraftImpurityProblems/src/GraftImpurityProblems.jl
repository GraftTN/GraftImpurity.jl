"""
Backend-neutral finite impurity problems, physical symmetry declarations, and
many-body target manifolds.

This package deliberately owns no bath fitting, Green-function conversion,
topology, tensor-network carrier, compiler route, or numerical workspace.
"""
module GraftImpurityProblems

using GraftFoundation: U1Irrep, SU2Irrep
using GraftImpurityFoundations: FlavorLayout, flavors, AbstractImpurityInteraction,
    bath_layout, bath_partition, bath_statistics, interaction_layout,
    interaction_identity, validate_partition
using GraftImpurityInteractions: ChargeU1, FlavorU1, SU2Reduce,
    ImpurityOneBody, one_body_layout
using GraftImpurityBaths: DiscreteBath

export AbstractImpuritySymmetryDeclaration, ImpuritySymmetryDeclaration,
    AbstractPhysicalActionSemantics, ChargeU1ActionSemantics,
    FlavorU1ActionSemantics, SU2ActionSemantics, SymmetryActionIdentity,
    ChargeU1, FlavorU1, SU2Reduce, action_identity, action_layout,
    action_semantics, category_product, symmetry_actions,
    symmetry_action_identities, symmetry_layout
export AbstractImpurityManifold, TargetIrrep, IrrepScan,
    manifold_action_identity, manifold_targets, manifold_identity,
    validate_manifold, validate_response_target, ResponseReachabilityError,
    validate_response_reachability
export ImpurityProblem, problem_layout, problem_partition,
    problem_statistics, problem_identity

include("symmetry.jl")
include("manifolds.jl")
include("problem.jl")

end # module GraftImpurityProblems
