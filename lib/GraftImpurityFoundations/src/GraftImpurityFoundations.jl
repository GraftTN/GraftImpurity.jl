"""
Low-level impurity-domain identities, layouts, partitions, local fermion
carriers, and shared numerical kernels for the GraftImpurity package family.
"""
module GraftImpurityFoundations

using LinearAlgebra: I, norm, opnorm
using GraftFoundation: ElementarySpace, AbstractTensorMap, FermionParity, Vect,
    U1Irrep, TensorMap, ⊠, ⊗, ←

export FlavorLayout, flavors, flavor_index, physical_site, site_modes,
    layout_sites, basis_identity
export Partition, block_names, block_flavors, block_index, partition_flavors,
    validate_partition
export AbstractRealPoleBathFitKernel, AbstractBathParametrization,
    AbstractBCFParametrization, AbstractHamiltonianBath,
    AbstractBathMappingKernel, AbstractImpurityTopologyPlan,
    AbstractMountedBath, AbstractImpurityInteraction, AbstractImpuritySolver,
    AbstractImpurityWorkspace, AbstractImpuritySolveRequest,
    AbstractImpuritySolveResult
export real_pole_bath_fit, fit_complex_bcf, evaluate_bcf,
    realize_quasi_lindblad, realize_coupled_lindblad, realize_bath, mount_bath,
    map_bath, impurity_topology, lower_interaction, audit_partition,
    factorize_residues, reconstruct_hybridization, audit_bathfit,
    audit_symmetry, solve!, bath_layout, bath_partition, bath_orbitals,
    bath_statistics, interaction_layout, interaction_identity
export FermionParitySector, ParticleNumberSector, FermionSiteOperators,
    fermion_sector, local_annihilator, local_creator, local_number

include("layout.jl")
include("partition.jl")
include("abstractions.jl")
include("local_fermions.jl")
include("nnls.jl")

end # module GraftImpurityFoundations
