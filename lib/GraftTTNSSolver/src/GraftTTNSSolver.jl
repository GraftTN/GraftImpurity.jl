module GraftTTNSSolver

import LinearAlgebra
using LinearAlgebra: I

using GraftFoundation: AbstractTensorMap, ElementarySpace, TreeTopology,
    TruncationScheme, FermionParity, U1Irrep, codomain, domain, sectors, dual,
    ⊠, nodeindex, spacetype
using GraftNetworks: TTNO, TTNS, apply_local, compress!, normalize!, physspace,
    topology
using GraftContractions: expect, inner
using GraftSymbolic: OpSum, SiteOp, Term, charge
using GraftGroundState: dmrg2!
using GraftEvolution: Evolver, correlator_series, step!, supports_complex_step
using GraftThermal: PurificationTrajectory, Purified, purification_problem,
    thermal_correlator, thermalize
using GraftTTNOBuild: ttno_from_opsum
using GraftStateDiagram: compile_ttno, AbstractOperatorLoweringKernel,
    AbelianScalarLowering, AbstractTTNOMergeKernel, DirectSumMerge,
    StateDiagramMerge, SGEOptimizer, MissingCategoryCapability, TTNOBuildReport

using GraftImpurityFoundations: FlavorLayout, Partition, flavors, block_flavors,
    block_names, flavor_index, layout_sites, basis_identity, validate_partition,
    AbstractBathMappingKernel,
    AbstractImpurityTopologyPlan, AbstractMountedBath,
    AbstractImpurityInteraction, AbstractImpuritySolver, AbstractImpurityWorkspace,
    AbstractImpuritySolveRequest, AbstractImpuritySolveResult,
    AbstractFermionSector, bath_layout, bath_partition, bath_statistics,
    interaction_layout, interaction_identity, mount_bath,
    map_bath, impurity_topology, lower_interaction, audit_symmetry
import GraftImpurityFoundations: solve!
using GraftImpurityFoundations: ParticleNumberSector

using GraftImpurityInteractions: ImpurityOperators, ImpurityOneBody,
    KanamoriInteraction, SymmetryAudit, SymmetrySpec, ChargeU1, FlavorU1,
    SU2Reduce,
    one_body_opsum, site_operators, _interaction_tolerance,
    _require_supported_symmetry

using GraftImpurityProblems: ImpurityProblem, AbstractImpurityManifold,
    TargetIrrep, IrrepScan, action_identity, manifold_identity,
    manifold_targets, ResponseReachabilityError, validate_response_target,
    problem_identity, problem_layout, problem_partition, problem_statistics,
    symmetry_actions, validate_manifold
import GraftImpurityProblems: validate_response_reachability

using GraftImpurityBaths: AndersonBath, CayleyAndersonBath, CayleyMappingResult,
    CayleyTreeKernel, DiscreteBath, T3NS, FTPS,
    _canonical_mounted_owners, _cayley_mapping_integrity_hash,
    _cayley_mounted_ownership_hash, _discrete_bath_integrity_hash,
    _mounted_ownership_hash, _opsum_integrity_hash, _validate_impurity_nodes

export AbstractTTNOBuilder, LegacyTTNOBuilder, CompiledTTNOBuilder,
    TTNOBuilderCapabilityError, build_ttno,
    LoweredImpurityHamiltonian, lower_hamiltonian,
    ZeroTemperature, FiniteTemperature, GroundStateRequest, RealTimeRequest,
    ImaginaryTimeRequest, ComplexTimeSegment, ComplexTimeRequest,
    LocalObservable, LocalCorrelator, RawCorrelator, GroundStateResult,
    ImaginaryTimeResult, TTNSSolveRequest, TTNSSolveResult, TTNSScanResult,
    TTNSCapabilityError, ResponseReachabilityError,
    TTNSSolver, TTNSWorkspace, solve!

include("hamiltonian.jl")
include("types.jl")
include("requests.jl")
include("orchestration.jl")

end # module GraftTTNSSolver
