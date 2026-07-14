"""
    CayleyAndersonBath

Concrete mounted fermionic bath produced from a [`CayleyMappingResult`](@ref)
whose physical sites each carry exactly one fermionic mode. It retains the
mapped bath-only one-particle Hamiltonian and coupling matrix exactly and is a
mapped Hamiltonian realization, not a canonical-star fallback. Multi-mode
physical carriers remain preserved in the mapping result but fail closed at
this mounting boundary until the core labelled-site braid supports physical
sector degeneracy.
"""
struct CayleyAndersonBath{M<:CayleyMappingResult,P<:NamedTuple,SM<:Tuple,
                          D<:NamedTuple} <: AbstractMountedBath
    mapping::M
    topology::TreeTopology
    phys::P
    sites::Tuple{Vararg{Symbol}}
    site_modes::SM
    H::OpSum
    diagnostics::D
    certificate::_MountedHamiltonianCertificate
end

Base.length(bath::CayleyAndersonBath) = length(bath.mapping.canonical)

_cayley_site_dimensions(bath::ScalarCayleyBath) = fill(1, length(bath.sites))
_cayley_site_dimensions(bath::BlockCayleyBath) = copy(bath.site_dimensions)

function _cayley_mode_labels(sites::AbstractVector{<:Symbol},
                             dimensions::AbstractVector{<:Integer})
    length(sites) == length(dimensions) || throw(DimensionMismatch(
        "Cayley mounted sites and dimensions must align",
    ))
    labels = Tuple(Tuple(Symbol("cayley_", String(site), "_", local_index)
                         for local_index in 1:dimension)
                   for (site, dimension) in zip(sites, dimensions))
    all(dimension -> dimension > 0, dimensions) || throw(ArgumentError(
        "Cayley mounted site dimensions must be positive",
    ))
    allunique(Iterators.flatten(labels)) || throw(ArgumentError(
        "Cayley mounted local-mode labels must be globally unique",
    ))
    return labels
end

function _cayley_mount_topology(mapping::CayleyMappingResult)
    mapped = mapping.mapped
    source = mapped.topology
    layout = bath_layout(mapped)
    impurity_sites = layout_sites(layout)
    any(site -> site in source.ids, impurity_sites) && throw(ArgumentError(
        "Cayley mapped bath-site labels may not reuse impurity physical sites",
    ))

    root = Graft.Trees.nodeid(source, source.root)
    edges = _topology_edges(source)
    for impurity_site in impurity_sites
        push!(edges, root => impurity_site)
    end
    return TreeTopology(root, edges)
end

function _cayley_mounted_site_operators(mapping::CayleyMappingResult,
                                         sector::AbstractFermionSector)
    mapped = mapping.mapped
    sites = Tuple(mapped.sites)
    dimensions = _cayley_site_dimensions(mapped)
    labels = _cayley_mode_labels(mapped.sites, dimensions)
    impurity = _mount_layout_operators(bath_layout(mapped), sector)
    bath = Dict{Symbol,FermionSiteOperators}(
        site => FermionSiteOperators(collect(modes); sector)
        for (site, modes) in zip(sites, labels)
    )
    phys = Dict{Symbol,ElementarySpace}(
        site => operators.P for (site, operators) in impurity
    )
    for (site, operators) in bath
        phys[site] = operators.P
    end
    return sites, labels, impurity, bath, phys
end

function _cayley_mode_locations(sites::Tuple{Vararg{Symbol}},
                                labels::Tuple)
    locations = Tuple{Symbol,Symbol,Int}[]
    for (site, modes) in zip(sites, labels)
        for (local_index, mode) in enumerate(modes)
            push!(locations, (site, mode, local_index))
        end
    end
    return locations
end

function _cayley_annihilator_siteop(site::Symbol, operators::FermionSiteOperators,
                                    mode::Symbol)
    index = _local_mode_index(operators, mode)
    return SiteOp(site, Symbol("cayley_C_", index),
                  local_annihilator(operators, mode))
end

function _cayley_creator_siteop(site::Symbol, operators::FermionSiteOperators,
                                mode::Symbol)
    index = _local_mode_index(operators, mode)
    return SiteOp(site, Symbol("cayley_Cd_", index),
                  local_creator(operators, mode))
end

function _cayley_local_bilinear(site::Symbol, operators::FermionSiteOperators,
                                 left::Symbol, right::Symbol)
    matrix, charge = _local_product(
        operators, _FermionFactor[_creator(left), _annihilator(right)],
    )
    iszero(charge) || throw(ArgumentError(
        "Cayley local one-body product must be charge neutral",
    ))
    left_index = _local_mode_index(operators, left)
    right_index = _local_mode_index(operators, right)
    return SiteOp(
        site, Symbol("cayley_bilinear_", left_index, "_", right_index),
        TensorMap(matrix, operators.P ← operators.P),
    )
end

function _cayley_mount_hamiltonian(mapping::CayleyMappingResult,
                                   sites::Tuple{Vararg{Symbol}}, labels::Tuple,
                                   impurity::AbstractDict{Symbol,<:FermionSiteOperators},
                                   bath::AbstractDict{Symbol,<:FermionSiteOperators})
    mapped = mapping.mapped
    locations = _cayley_mode_locations(sites, labels)
    length(locations) == length(mapped) || throw(DimensionMismatch(
        "Cayley local Fock carriers must cover every transformed bath mode",
    ))
    H_bath = mapped.bath_hamiltonian
    W = mapped.coupling_matrix
    size(H_bath) == (length(locations), length(locations)) || throw(DimensionMismatch(
        "Cayley mapped bath Hamiltonian has the wrong mode dimension",
    ))
    size(W) == (length(flavors(bath_layout(mapped))), length(locations)) ||
        throw(DimensionMismatch("Cayley mapped coupling matrix has the wrong shape"))

    H = OpSum()
    retained_bath_terms = 0
    for column in axes(H_bath, 2), row in axes(H_bath, 1)
        coefficient = H_bath[row, column]
        iszero(coefficient) && continue
        left_site, left_mode, _ = locations[row]
        right_site, right_mode, _ = locations[column]
        if left_site == right_site
            H += Term(coefficient, _cayley_local_bilinear(
                left_site, bath[left_site], left_mode, right_mode,
            ))
        else
            H += Term(
                coefficient,
                _cayley_creator_siteop(left_site, bath[left_site], left_mode),
                _cayley_annihilator_siteop(right_site, bath[right_site], right_mode),
            )
        end
        retained_bath_terms += 1
    end

    retained_couplings = 0
    layout = bath_layout(mapped)
    for mode_index in axes(W, 2), flavor_index_value in axes(W, 1)
        coefficient = W[flavor_index_value, mode_index]
        iszero(coefficient) && continue
        flavor = flavors(layout)[flavor_index_value]
        impurity_site = physical_site(layout, flavor)
        bath_site, bath_mode, _ = locations[mode_index]
        H += Term(
            coefficient,
            SiteOp(impurity_site, Symbol("Cd_", flavor),
                   local_creator(impurity[impurity_site], flavor)),
            _cayley_annihilator_siteop(bath_site, bath[bath_site], bath_mode),
        )
        H += Term(
            conj(coefficient),
            SiteOp(impurity_site, Symbol("C_", flavor),
                   local_annihilator(impurity[impurity_site], flavor)),
            _cayley_creator_siteop(bath_site, bath[bath_site], bath_mode),
        )
        retained_couplings += 1
    end
    return H, retained_bath_terms, retained_couplings
end

function _validate_cayley_mount_carriers(mapped::AbstractCayleyBath)
    layout = bath_layout(mapped)
    impurity_widths = Int[
        length(site_modes(layout, site)) for site in layout_sites(layout)
    ]
    bath_widths = _cayley_site_dimensions(mapped)
    all(==(1), impurity_widths) && all(==(1), bath_widths) && return nothing
    throw(ArgumentError(
        "Cayley mapped Hamiltonian mounting requires exactly one fermionic mode " *
        "per physical site; multi-mode physical carriers remain preserved in " *
        "CayleyMappingResult but are non-mountable until the core labelled-site " *
        "braid supports physical-sector degeneracy",
    ))
end

function _cayley_integrity_hash_matrix(values::AbstractMatrix{<:Number}, state::UInt)
    state = hash(size(values), state)
    for value in values
        state = hash(value, state)
    end
    return state
end

function _cayley_integrity_hash_topology(topology::TreeTopology, state::UInt)
    state = hash((topology.root, Tuple(topology.ids), Tuple(topology.parent),
                  Tuple(topology.depth)), state)
    for children in topology.children
        state = hash(Tuple(children), state)
    end
    for label in sort!(collect(keys(topology.index)); by=string)
        state = hash((label, topology.index[label]), state)
    end
    return state
end

function _cayley_mapping_integrity_hash(mapping::CayleyMappingResult)
    mapped = mapping.mapped
    state = _discrete_bath_integrity_hash(mapping.canonical)
    state = hash((:GraftImpurityCayleyMappingIntegrity, typeof(mapped),
                  Tuple(mapped.sites)), state)
    state = _cayley_integrity_hash_topology(mapped.topology, state)
    state = _cayley_integrity_hash_matrix(mapping.transform, state)
    state = _cayley_integrity_hash_matrix(mapped.bath_hamiltonian, state)
    state = _cayley_integrity_hash_matrix(mapped.coupling_matrix, state)
    if mapped isa BlockCayleyBath
        state = hash(Tuple(mapped.site_dimensions), state)
    end
    return state
end

function _cayley_mounted_ownership_hash(mapping_hash::UInt,
                                        topology::TreeTopology,
                                        sites::Tuple{Vararg{Symbol}})
    state = hash((:cayley_mounted_ownership, mapping_hash, sites))
    return _cayley_integrity_hash_topology(topology, state)
end

function _cayley_mount_diagnostics(user::NamedTuple,
                                   mapping::CayleyMappingResult,
                                   topology::TreeTopology,
                                   sites::Tuple{Vararg{Symbol}}, H::OpSum,
                                   retained_bath_terms::Int,
                                   retained_couplings::Int,
                                   mapping_hash::UInt)
    required = (
        ; kind=:cayley_anderson,
        mapping_kind=mapping.mapped isa ScalarCayleyBath ? :scalar : :block,
        mapping_hash,
        ownership_hash=_cayley_mounted_ownership_hash(mapping_hash, topology, sites),
        retained_bath_terms,
        retained_couplings,
        hamiltonian_hash=_opsum_integrity_hash(H),
        mapping_approximate=mapping.report.approximate,
    )
    any(name -> name in keys(required), keys(user)) && throw(ArgumentError(
        "Cayley mount diagnostics may not overwrite canonical fields",
    ))
    return merge(required, user)
end

"""
    mount_bath(mapping::CayleyMappingResult; sector=ParticleNumberSector(),
               diagnostics=(;)) -> CayleyAndersonBath

Materialize a transformed one-particle Hamiltonian whose impurity and mapped
physical sites each carry one fermionic mode on an exact extension of its
declared mapping topology. The mapped root,
virtual hubs, forest components, child order, and every mapped edge are retained;
impurity physical sites are appended beneath that root. The canonical star data
is retained inside `mapping`; this method never rotates back to it or deletes
mapped off-diagonal hoppings/couplings.
"""
function mount_bath(mapping::CayleyMappingResult;
                    sector::AbstractFermionSector=ParticleNumberSector(),
                    diagnostics::NamedTuple=(;))
    mapped = mapping.mapped
    bath_statistics(mapped) === :fermion || throw(ArgumentError(
        "Cayley mounting currently requires a fermionic mapped bath",
    ))
    _validate_cayley_mount_carriers(mapped)
    topology = _cayley_mount_topology(mapping)
    sites, labels, impurity, bath, phys = _cayley_mounted_site_operators(mapping, sector)
    H, retained_bath_terms, retained_couplings = _cayley_mount_hamiltonian(
        mapping, sites, labels, impurity, bath,
    )
    mapping_hash = _cayley_mapping_integrity_hash(mapping)
    certificate = _MountedHamiltonianCertificate(
        _opsum_integrity_hash(H), mapping_hash,
    )
    fields = _canonical_phys(phys)
    return CayleyAndersonBath(
        mapping, topology, fields, sites, labels, H,
        _cayley_mount_diagnostics(
            diagnostics, mapping, topology, sites, H, retained_bath_terms,
            retained_couplings, mapping_hash,
        ),
        certificate,
    )
end
