module GraftImpuritySolver

import LinearAlgebra
using LinearAlgebra: I

import GreenFunc

using GraftFoundation: AbstractTensorMap, ElementarySpace, TreeTopology,
    TruncationScheme, nodeindex, spacetype
using GraftNetworks: TTNO, TTNS, apply_local, compress!, normalize!, physspace,
    topology
using GraftContractions: expect, inner
using GraftSymbolic: OpSum, SiteOp, Term
using GraftGroundState: dmrg2!
using GraftEvolution: Evolver, correlator_series, step!, supports_complex_step
using GraftThermal: PurificationTrajectory, Purified, purification_problem,
    thermal_correlator, thermalize
using GraftTTNOBuild: ttno_from_opsum
using GraftStateDiagram: compile_ttno, AbstractOperatorLoweringKernel,
    AbelianScalarLowering, AbstractTTNOMergeKernel, DirectSumMerge,
    StateDiagramMerge, SGEOptimizer, MissingCategoryCapability, TTNOBuildReport

using GraftImpurityFoundations: FlavorLayout, Partition, block_flavors,
    block_names, flavor_index, layout_sites, basis_identity, validate_partition,
    AbstractRealPoleBathFitKernel, AbstractBathMappingKernel,
    AbstractImpurityTopologyPlan, AbstractMountedBath,
    AbstractImpurityInteraction, AbstractImpuritySolver,
    AbstractImpuritySolveRequest, AbstractImpuritySolveResult,
    bath_layout, bath_statistics, real_pole_bath_fit, realize_bath, mount_bath,
    map_bath, impurity_topology, lower_interaction, audit_bathfit, audit_symmetry
import GraftImpurityFoundations: set_weiss!, set_hybridization!, solve!
using GraftImpurityFoundations: ParticleNumberSector

using GraftImpurityInteractions: ImpurityOperators, ImpurityOneBody,
    KanamoriInteraction, SymmetryAudit, SymmetrySpec, ChargeU1,
    one_body_opsum, site_operators, _interaction_tolerance,
    _require_supported_symmetry

using GraftImpurityBaths: AndersonBath, CayleyAndersonBath, CayleyMappingResult,
    CayleyTreeKernel, DiscreteBath, PoleExpansion, T3NS, FTPS,
    _canonical_mounted_owners, _cayley_mapping_integrity_hash,
    _cayley_mounted_ownership_hash, _discrete_bath_integrity_hash,
    _mounted_ownership_hash, _opsum_integrity_hash, _validate_impurity_nodes

using GraftImpurityBathFit: BathFitAudit, BathFitCriteria, BathFitInput,
    BathFitReport, DiscretizationResult, NonMountablePoleFit,
    _reconstructed_template, _validate_fit_input

export AbstractTTNOBuilder, LegacyTTNOBuilder, CompiledTTNOBuilder,
    TTNOBuilderCapabilityError, build_ttno,
    LoweredImpurityHamiltonian, lower_hamiltonian,
    ZeroTemperature, FiniteTemperature, GroundStateRequest, RealTimeRequest,
    ImaginaryTimeRequest, ComplexTimeSegment, ComplexTimeRequest,
    LocalObservable, LocalCorrelator, RawCorrelator, GroundStateResult,
    ImaginaryTimeResult, TTNSSolveRequest, TTNSNonMountableSolveResult,
    TTNSSolveResult, TTNSSolver, set_weiss!, set_hybridization!, solve!

include("hamiltonian.jl")
include("types.jl")
include("requests.jl")
include("orchestration.jl")

end # module GraftImpuritySolver
