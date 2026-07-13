"""
GraftImpurity: impurity-model and bath-realization companion package for
Graft.jl.

This package owns impurity-specific basis identity, named hybridization
partitions, bath fitting/realization, topology planning, interaction lowering,
and solver orchestration. Graft remains dependency-free of this package.
"""
module GraftImpurity

using LinearAlgebra: Diagonal, Hermitian, I, diag, eigen, eigvals, norm, opnorm,
    svd, tr
using Graft
import GreenFunc
using Graft.Backend: ElementarySpace
using Graft.Symbolic: OpSum
using Graft.Trees: TreeTopology

export FlavorLayout, flavors, flavor_index, physical_site, site_modes,
    layout_sites, basis_identity,
    Partition, block_names, block_flavors, block_index, partition_flavors,
    validate_partition,
    AbstractRealPoleBathFitKernel, AbstractBathParametrization,
    AbstractHamiltonianBath, AbstractBathMappingKernel,
    AbstractImpurityTopologyPlan, AbstractMountedBath,
    AbstractImpurityInteraction, AbstractImpuritySolver,
    BlockRealPoles, PoleExpansion, BathOrbitals, DiscreteBath,
    bath_layout, bath_partition, bath_orbitals, bath_statistics,
    MountedBath,
    T3NS, FTPS,
    BathFitInput, SpectralInterval, BlockDiscretizationPlan,
    DiscretizationPlan, plan_block, PoleBinDiagnostic,
    DiscretizationResult, NonMountablePoleFit,
    real_pole_bath_fit, realize_bath, mount_bath, map_bath,
    impurity_topology, lower_interaction, audit_partition, factorize_residues,
    reconstruct_hybridization, audit_bathfit, audit_symmetry,
    set_weiss!, set_hybridization!, solve!,
    IRCoefficients, fit_ir, evaluate_ir, to_imtime_ir, to_imfreq_ir,
    PESPoleFit, pes_fit, evaluate_poles,
    LorentzianPSD, MatrixLorentzianPSD, lorentzian_fit, spectral_density,
    complex_poles

include(joinpath(@__DIR__, "foundations", "layout.jl"))
include(joinpath(@__DIR__, "foundations", "partition.jl"))
include(joinpath(@__DIR__, "foundations", "abstractions.jl"))
include(joinpath(@__DIR__, "bath", "parametrizations.jl"))
include(joinpath(@__DIR__, "bath", "discrete_bath.jl"))
include(joinpath(@__DIR__, "bath", "mounted_baths.jl"))
include(joinpath(@__DIR__, "topology", "plans.jl"))
include(joinpath(@__DIR__, "fitting", "nnls.jl"))
include(joinpath(@__DIR__, "fitting", "input.jl"))
include(joinpath(@__DIR__, "fitting", "plans.jl"))
include(joinpath(@__DIR__, "bath", "realization.jl"))
include(joinpath(@__DIR__, "pes_pole_fitting.jl"))
include(joinpath(@__DIR__, "lorentzian_psd.jl"))
include(joinpath(@__DIR__, "sparseir_adapter.jl"))

end # module GraftImpurity
