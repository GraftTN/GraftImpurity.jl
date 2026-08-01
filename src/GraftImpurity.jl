"""
GraftImpurity: impurity-model and bath-realization companion package for
Graft.jl.

This package owns impurity-specific basis identity, named hybridization
partitions, bath fitting/realization, topology planning, interaction lowering,
and solver orchestration. Graft remains dependency-free of this package.

Thermofield transformations belong only to EDMFT bosonic-bath chain mapping
and layout; they are not a finite-temperature state-preparation contract.
Finite-temperature solver requests use Graft's purification machinery.

The `spectra/` layer owns Green-function-adjacent numerics (Matsubara
transforms today; particle/hole assembly and self-energy are planned). The
`validation/` layer owns finite-mode Anderson-Holstein benchmark records and
Kondo/bath scaling analysis; CTSEG itself is never executed here —
cross-checks read the committed JLD2 reference data only.
"""
module GraftImpurity

using GraftImpurityFoundations
using GraftImpurityInteractions
using GraftImpurityBaths
using GraftImpurityPoleFits
using GraftImpurityBathFit
using GraftImpuritySolver
using GraftImpurityValidation

# Preserve concrete private bindings embedded in pre-split serialized public
# values. These are aliases to the owner-package types, not compatibility
# wrappers.
import GraftImpurityBaths: _MountedHamiltonianCertificate
import GraftImpurityBathFit: _DMFTMeasureSnapshot, _DMFTPositiveMeasure

export FlavorLayout, flavors, flavor_index, physical_site, site_modes,
    layout_sites, basis_identity,
    Partition, block_names, block_flavors, block_index, partition_flavors,
    validate_partition,
    AbstractRealPoleBathFitKernel, AbstractBathParametrization,
    AbstractBCFParametrization,
    AbstractHamiltonianBath, AbstractBathMappingKernel,
    AbstractImpurityTopologyPlan, AbstractMountedBath,
    AbstractImpurityInteraction, AbstractImpuritySolver,
    BlockRealPoles, PoleExpansion, BathOrbitals, DiscreteBath, ComplexPoles,
    bath_layout, bath_partition, bath_orbitals, bath_statistics,
    FermionParitySector, ParticleNumberSector, FermionSiteOperators,
    fermion_sector, local_annihilator, local_creator, local_number, rotate_bath,
    ImpurityOperators, site_operators,
    AndersonBath, BosonBath, CayleyAndersonBath,
    AbstractCayleyRoute, ScalarCayley, BlockCayley, AbstractCayleyPartitioner,
    BalancedCayleyPartitioner, EnergySplitCayleyPartitioner,
    CayleyOwnershipGroup, CayleyTreeKernel, AbstractCayleyBath,
    ScalarCayleyEdge, ScalarCayleyRoot, BlockCayleyEdge, BlockCayleyRoot,
    ScalarCayleyBath, BlockCayleyBath, CayleyGroupReport, BathMappingReport,
    CayleyMappingResult,
    T3NS, FTPS,
    BathFitInput, BCFFitInput, SpectralInterval, BlockDiscretizationPlan,
    DiscretizationPlan, plan_block, PoleBinDiagnostic, BathFitResidual,
    BathFitBlockReport, BathFitTiming, BathFitWarning, BathFitReport,
    BathFitCriteria, BathFitAuditItem, BathFitAudit,
    AbstractBathFitPerturbation, CovariancePerturbation,
    EmpiricalReplicaPerturbation, BathFitDiagnosticConfig,
    BathFitHealthCandidate, BathFitMetricSummary, BathFitOrderHealth,
    BathFitHealthThresholds, BathFitHealthReport,
    DiscretizationResult, NonMountablePoleFit,
    QuadratureKernel, BoundaryFitKernel, PESKernel, MiniPoleKernel,
    CouplingFitKernel, ESPRITTauKernel, CouplingBlockTie,
    FreeModeAllocation, SignedModeAllocation,
    ComplexComponents, RealComponents, EqualTie, ConjugateTie,
    real_pole_bath_fit, fit_complex_bcf, evaluate_bcf,
    realize_bath, mount_bath, map_bath, realize_quasi_lindblad,
    realize_coupled_lindblad,
    impurity_topology, lower_interaction, audit_partition, factorize_residues,
    DensityDensityInteraction, KanamoriTerms, KanamoriFlavorMap,
    KanamoriInteraction, BareCoulombTensor, AntisymmetrizedVertex,
    FullCoulombInteraction, ImpurityOneBody, DensityDensityDecomposition,
    split_density_density, lower_one_body, one_body_opsum,
    rotate_one_body, rotate_interaction,
    ChargeU1, FlavorU1, SU2Reduce, SymmetrySpec, SymmetryAuditItem,
    SymmetryAudit, AbstractTTNOBuilder, LegacyTTNOBuilder, CompiledTTNOBuilder,
    TTNOBuilderCapabilityError, build_ttno,
    LoweredImpurityHamiltonian, lower_hamiltonian,
    reconstruct_hybridization, residual_hybridization, audit_bathfit,
    attach_bathfit_health, analyze_bathfit, run_bathfit_diagnostics,
    DMFTBathBlockRecord, DMFTBathComplexity, DMFTBathIterationRecord,
    DMFTBathVerdictEvidence, DMFTBathMonitorReport, DMFTBathMonitor,
    update!, dmft_bath_report,
    audit_symmetry,
    ZeroTemperature, FiniteTemperature, GroundStateRequest, RealTimeRequest,
    ImaginaryTimeRequest, ComplexTimeSegment, ComplexTimeRequest,
    LocalObservable, LocalCorrelator, RawCorrelator, GroundStateResult,
    ImaginaryTimeResult, SolveRequest, NonMountableImpurityResult,
    ImpurityResult, Solver, set_weiss!, set_hybridization!, solve!,
    IRCoefficients, fit_ir, evaluate_ir, to_imtime_ir, to_imfreq_ir,
    PESPoleFit, pes_fit, evaluate_poles,
    LorentzianPSD, MatrixLorentzianPSD, lorentzian_fit, spectral_density,
    complex_poles,
    MatsubaraSeries, matsubara_transform,
    KondoScalingResult, fit_kondo_scaling,
    SemicircularBathReport, semicircular_hybridization,
    gauss_semicircular_bath, discrete_bath_hybridization,
    validate_semicircular_bath, read_bath_csv,
    FiniteModeAction, finite_mode_hash, fermionic_frequency,
    bosonic_frequency, hybridization_iw, retarded_interaction_iv,
    ThermalBenchmarkDatum,
    FiniteModeBenchmarkCell, BosonCutoffReport, assess_boson_cutoff,
    RepresentationComparison, compare_representations

include(joinpath(@__DIR__, "precompile.jl"))

end # module GraftImpurity
