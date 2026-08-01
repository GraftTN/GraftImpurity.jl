"""
Canonical impurity-bath parametrizations, residue factorization, topology
planning, Hamiltonian mounting, basis rotation, and Cayley mappings.
"""
module GraftImpurityBaths

using LinearAlgebra: Diagonal, Hermitian, I, diag, eigen, eigvals, norm,
    opnorm, qr, svd
using GraftFoundation: ElementarySpace, TensorMap, TreeTopology, is_t3ns,
    nodeid, nodeindex, nnodes, ←
using GraftSymbolic: OpSum, SiteOp, Term
using GraftImpurityFoundations: FlavorLayout, flavors, flavor_index,
    physical_site, site_modes, layout_sites, Partition, block_names,
    block_flavors, block_index, validate_partition, AbstractBathParametrization,
    AbstractBCFParametrization, AbstractHamiltonianBath,
    AbstractBathMappingKernel, AbstractImpurityTopologyPlan, AbstractMountedBath,
    FermionParitySector, ParticleNumberSector, FermionSiteOperators,
    local_annihilator, local_creator
import GraftImpurityFoundations: AbstractFermionSector, _local_mode_index,
    bath_layout, bath_partition, bath_orbitals, bath_statistics, evaluate_bcf,
    mount_bath, map_bath, impurity_topology, factorize_residues
import GraftImpurityInteractions: _FermionFactor, _annihilator, _creator,
    _local_product, _validated_basis_rotation

export BlockRealPoles, PoleExpansion, BathOrbitals, DiscreteBath, ComplexPoles,
    AndersonBath, BosonBath, CayleyAndersonBath
export AbstractCayleyRoute, ScalarCayley, BlockCayley,
    AbstractCayleyPartitioner, BalancedCayleyPartitioner,
    EnergySplitCayleyPartitioner, CayleyOwnershipGroup, CayleyTreeKernel,
    AbstractCayleyBath, ScalarCayleyEdge, ScalarCayleyRoot, BlockCayleyEdge,
    BlockCayleyRoot, ScalarCayleyBath, BlockCayleyBath, CayleyGroupReport,
    BathMappingReport, CayleyMappingResult
export T3NS, FTPS, PoleBinDiagnostic, rotate_bath
export bath_layout, bath_partition, bath_orbitals, bath_statistics, evaluate_bcf,
    mount_bath, map_bath, impurity_topology, factorize_residues

include(joinpath("bath", "parametrizations.jl"))
include(joinpath("bath", "factorization.jl"))
include(joinpath("bath", "complex_poles.jl"))
include(joinpath("bath", "discrete_bath.jl"))
include(joinpath("bath", "mounted_baths.jl"))
include(joinpath("topology", "plans.jl"))
include(joinpath("topology", "builders.jl"))
include(joinpath("bath", "mounting.jl"))
include(joinpath("mapping", "types.jl"))
include(joinpath("mapping", "scalar.jl"))
include(joinpath("mapping", "block.jl"))
include(joinpath("bath", "cayley_mounting.jl"))
include(joinpath("bath", "rotation.jl"))

end # module GraftImpurityBaths
