"""
Bath fitting, realization reports, diagnostics, DMFT monitoring, and
GreenFunc/SparseIR adapters for the GraftImpurity package family.
"""
module GraftImpurityBathFit

using LinearAlgebra: Diagonal, Hermitian, I, diag, eigen, eigvals, norm, opnorm,
    svd, tr
import LinearAlgebra
import GreenFunc
import Optim
import SparseIR

using GraftImpurityFoundations: AbstractFermionSector,
    AbstractRealPoleBathFitKernel, FlavorLayout, ParticleNumberSector,
    Partition, block_flavors, block_index, block_names, validate_partition,
    bath_layout, bath_partition, bath_statistics
import GraftImpurityFoundations: audit_bathfit, evaluate_bcf, fit_complex_bcf,
    mount_bath, real_pole_bath_fit, realize_bath, reconstruct_hybridization

using GraftImpurityBaths: BlockRealPoles, ComplexPoles, DiscreteBath,
    PoleBinDiagnostic, PoleExpansion, TreeTopology
import GraftImpurityBaths: _attempt_factorization,
    _bathfit_health_mount_summary, _resolved_orbital_order
using GraftImpurityPoleFits: pes_fit
using GraftSpectral: AbstractNodeFailure, AllComponents, ClampedRank,
    DescendingRankSearch, ESPRIT, ESPRITDiagnostics, ExponentialSum, FailedFit,
    FirstControlled, IdentifiedFit, IdentifiedNodes, LeftSubspaceESPRIT,
    MinimumTrainingRelativeL2, NodeEstimationFailure, NumericalRank,
    RelativeThresholdRank, SampleReductionPolicy, StrictRank, UniformSequence,
    WeightNormPruning, ZeroFit, ZeroSequence, estimate_nodes, evaluate,
    fit_exponential_sum

export BathFitInput, BCFFitInput, SpectralInterval, BlockDiscretizationPlan,
    DiscretizationPlan, plan_block
export BathFitResidual, BathFitBlockReport, BathFitTiming, BathFitWarning,
    BathFitReport, BathFitCriteria, BathFitAuditItem, BathFitAudit
export AbstractBathFitPerturbation, CovariancePerturbation,
    EmpiricalReplicaPerturbation, BathFitDiagnosticConfig,
    BathFitHealthCandidate, BathFitMetricSummary, BathFitOrderHealth,
    BathFitHealthThresholds, BathFitHealthReport
export DiscretizationResult, NonMountablePoleFit
export QuadratureKernel, BoundaryFitKernel, PESKernel, MiniPoleKernel,
    CouplingFitKernel, ESPRITTauKernel, CouplingBlockTie, FreeModeAllocation,
    SignedModeAllocation, ComplexComponents, RealComponents, EqualTie,
    ConjugateTie
export real_pole_bath_fit, fit_complex_bcf, evaluate_bcf, realize_bath,
    mount_bath, reconstruct_hybridization, residual_hybridization,
    audit_bathfit, attach_bathfit_health, analyze_bathfit,
    run_bathfit_diagnostics
export DMFTBathBlockRecord, DMFTBathComplexity, DMFTBathIterationRecord,
    DMFTBathVerdictEvidence, DMFTBathMonitorReport, DMFTBathMonitor, update!,
    dmft_bath_report
export IRCoefficients, fit_ir, evaluate_ir, to_imtime_ir, to_imfreq_ir

include(joinpath(@__DIR__, "fitting", "input.jl"))
include(joinpath(@__DIR__, "fitting", "bcf_input.jl"))
include(joinpath(@__DIR__, "fitting", "plans.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_types.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_report_types.jl"))
include(joinpath(@__DIR__, "diagnostics", "types.jl"))
include(joinpath(@__DIR__, "diagnostics", "reconstruction.jl"))
include(joinpath(@__DIR__, "diagnostics", "report.jl"))
include(joinpath(@__DIR__, "fitting", "realization.jl"))
include(joinpath(@__DIR__, "diagnostics", "audit.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_residuals.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_measures.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_spectral_details.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_profiles.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_sensitivity.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_order_data.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_analyzer.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_replicas.jl"))
include(joinpath(@__DIR__, "diagnostics", "health_runner.jl"))
include(joinpath(@__DIR__, "diagnostics", "dmft_monitor_types.jl"))
include(joinpath(@__DIR__, "diagnostics", "dmft_monitor_state.jl"))
include(joinpath(@__DIR__, "diagnostics", "dmft_monitor_updates.jl"))
include(joinpath(@__DIR__, "diagnostics", "dmft_monitor_report.jl"))
include(joinpath(@__DIR__, "fitting", "kernels.jl"))
include(joinpath(@__DIR__, "fitting", "quadrature.jl"))
include(joinpath(@__DIR__, "fitting", "pes_kernel.jl"))
include(joinpath(@__DIR__, "fitting", "minipole.jl"))
include(joinpath(@__DIR__, "fitting", "esprit_tau.jl"))
include(joinpath(@__DIR__, "fitting", "complex_bcf.jl"))
include(joinpath(@__DIR__, "fitting", "coupling_fit.jl"))
include(joinpath(@__DIR__, "fitting", "boundary_fit.jl"))
include("sparseir_adapter.jl")

end # module GraftImpurityBathFit
