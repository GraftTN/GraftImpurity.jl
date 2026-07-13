"""
    AndersonBath(parametrization, topology, phys, sites, anchors, H;
                 diagnostics=(;))

Concrete mounted fermionic realization. Every canonical bath mode owns exactly
one mounted site, in the same order as `BathOrbitals`. `owners` freezes the
declared flavor action at mounting time; it is never inferred later from a
mutable canonical-bath vector. Anchors are explicit declared ownership sites
and are never inferred from a largest coupling component.
"""
struct _MountedHamiltonianCertificate
    hamiltonian_hash::UInt
    parametrization_hash::UInt
end

struct AndersonBath{B<:AbstractHamiltonianBath,P<:NamedTuple,D<:NamedTuple} <:
        AbstractMountedBath
    parametrization::B
    topology::TreeTopology
    phys::P
    sites::Tuple{Vararg{Symbol}}
    anchors::Tuple{Vararg{Symbol}}
    owners::Union{Nothing,Tuple{Vararg{Symbol}}}
    H::OpSum
    diagnostics::D
    certificate::Union{Nothing,_MountedHamiltonianCertificate}

    function AndersonBath(parametrization::B, topology::TreeTopology,
                          phys::P, sites::Tuple{Vararg{Symbol}},
                          anchors::Tuple{Vararg{Symbol}}, H::OpSum,
                          owners::Union{Nothing,Tuple{Vararg{Symbol}}},
                          diagnostics::D,
                          certificate::Union{Nothing,_MountedHamiltonianCertificate},
                          ::Val{:validated}) where {
                             B<:AbstractHamiltonianBath,P<:NamedTuple,D<:NamedTuple}
        new{B,P,D}(parametrization, topology, phys, sites, anchors, owners, H,
                   diagnostics, certificate)
    end
end

"""
    BosonBath(parametrization, topology, phys, sites, anchors, H;
              diagnostics=(;))

Concrete mounted bosonic realization.  It shares the exact mounted-data fields
with `AndersonBath`, but construction requires an explicitly supplied bosonic
physical-space and Hamiltonian convention.  Canonical `DiscreteBath` data does
not contain a local cutoff or a matter-coupling operator convention, so M5 does
not guess either when mounting it.
"""
struct BosonBath{B<:AbstractHamiltonianBath,P<:NamedTuple,D<:NamedTuple} <:
        AbstractMountedBath
    parametrization::B
    topology::TreeTopology
    phys::P
    sites::Tuple{Vararg{Symbol}}
    anchors::Tuple{Vararg{Symbol}}
    H::OpSum
    diagnostics::D

    function BosonBath(parametrization::B, topology::TreeTopology,
                       phys::P, sites::Tuple{Vararg{Symbol}},
                       anchors::Tuple{Vararg{Symbol}}, H::OpSum,
                       diagnostics::D, ::Val{:validated}) where {
                           B<:AbstractHamiltonianBath,P<:NamedTuple,D<:NamedTuple}
        new{B,P,D}(parametrization, topology, phys, sites, anchors, H,
                   diagnostics)
    end
end

function _canonical_phys(phys::AbstractDict)
    all(pair -> pair.first isa Symbol, pairs(phys)) ||
        throw(ArgumentError("mounted bath physical-space keys must be Symbols"))
    all(pair -> pair.second isa ElementarySpace, pairs(phys)) ||
        throw(ArgumentError("mounted bath physical spaces must be Graft ElementarySpaces"))
    labels = sort!(Symbol.(collect(keys(phys))); by=string)
    values = Tuple(phys[label] for label in labels)
    return NamedTuple{Tuple(labels)}(values)
end

function _require_topology_node(topology::TreeTopology, label::Symbol,
                                role::AbstractString)
    try
        Graft.Trees.nodeindex(topology, label)
    catch err
        err isa KeyError || rethrow()
        throw(ArgumentError("mounted bath $role $label is not present in the topology"))
    end
    return nothing
end

function _mounted_site_has_ancestor(topology::TreeTopology, node::Symbol,
                                    ancestor::Symbol)
    index = Graft.Trees.nodeindex(topology, node)
    ancestor_index = Graft.Trees.nodeindex(topology, ancestor)
    while index != 0
        index == ancestor_index && return true
        index = topology.parent[index]
    end
    return false
end

function _nearest_impurity_ancestor(topology::TreeTopology, node::Symbol,
                                    impurity_sites::Tuple{Vararg{Symbol}})
    index = Graft.Trees.nodeindex(topology, node)
    while index != 0
        label = Graft.Trees.nodeid(topology, index)
        label in impurity_sites && return label
        index = topology.parent[index]
    end
    return nothing
end

_canonical_mounted_anchors(::AbstractHamiltonianBath) = nothing

_canonical_mounted_owners(::AbstractHamiltonianBath) = nothing

function _canonical_mounted_anchors(bath::DiscreteBath)
    return Tuple(physical_site(bath_layout(bath), owner)
                 for owner in bath_orbitals(bath).associated_flavors)
end

_canonical_mounted_owners(bath::DiscreteBath) =
    Tuple(bath_orbitals(bath).associated_flavors)

function _mounted_ownership_hash(parametrization::AbstractHamiltonianBath,
                                 topology::TreeTopology,
                                 sites::Tuple{Vararg{Symbol}},
                                 anchors::Tuple{Vararg{Symbol}})
    return hash((:mounted_ownership, topology, sites, anchors))
end

function _mounted_ownership_hash(bath::DiscreteBath, topology::TreeTopology,
                                 sites::Tuple{Vararg{Symbol}},
                                 anchors::Tuple{Vararg{Symbol}})
    orbitals = bath_orbitals(bath)
    return hash((:discrete_bath_ownership, topology, sites, anchors,
                 Tuple(orbitals.associated_flavors),
                 Tuple(orbitals.block_indices), Tuple(orbitals.pole_indices)))
end

function _mounted_diagnostics(diagnostics::NamedTuple, ownership_hash::UInt)
    :ownership_hash in keys(diagnostics) && throw(ArgumentError(
        "mounted bath diagnostics may not overwrite the canonical ownership hash",
    ))
    return merge((; ownership_hash), diagnostics)
end

function _validated_mounted_fields(parametrization::AbstractHamiltonianBath,
                                   topology::TreeTopology, phys::AbstractDict,
                                   sites::AbstractVector{Symbol},
                                   anchors::AbstractVector{Symbol}, H::OpSum,
                                   diagnostics::NamedTuple, statistics::Symbol)
    bath_statistics(parametrization) == statistics ||
        throw(ArgumentError("mounted $(statistics) bath needs matching canonical statistics"))
    length(sites) == length(anchors) ||
        throw(DimensionMismatch("mounted bath needs one anchor per site"))
    length(sites) == length(parametrization) ||
        throw(DimensionMismatch("mounted bath needs one site per canonical bath mode"))
    allunique(sites) ||
        throw(ArgumentError("mounted bath site labels must be unique"))

    layout = bath_layout(parametrization)
    impurity_sites = layout_sites(layout)
    physical = _canonical_phys(phys)
    site_tuple = Tuple(Symbol.(sites))
    anchor_tuple = Tuple(Symbol.(anchors))
    canonical_anchors = _canonical_mounted_anchors(parametrization)
    canonical_anchors === nothing || anchor_tuple == canonical_anchors || throw(ArgumentError(
        "mounted bath anchors must match canonical associated_flavor ownership",
    ))
    canonical_owners = _canonical_mounted_owners(parametrization)
    canonical_owners === nothing || length(canonical_owners) == length(site_tuple) ||
        throw(ArgumentError("mounted bath canonical owners must match its sites"))
    for impurity_site in impurity_sites
        _require_topology_node(topology, impurity_site, "impurity site")
        hasproperty(physical, impurity_site) || throw(ArgumentError(
            "mounted impurity site $impurity_site has no physical space",
        ))
    end
    for (site, anchor) in zip(site_tuple, anchor_tuple)
        site in impurity_sites && throw(ArgumentError(
            "mounted bath sites must not reuse declared impurity sites",
        ))
        anchor in impurity_sites || throw(ArgumentError(
            "mounted bath anchor $anchor is not a declared impurity site",
        ))
        _require_topology_node(topology, site, "site")
        hasproperty(physical, site) ||
            throw(ArgumentError("mounted bath site $site has no physical space"))
        _require_topology_node(topology, anchor, "anchor")
        _mounted_site_has_ancestor(topology, site, anchor) || throw(ArgumentError(
            "mounted bath site $site is not in the declared owner arm rooted at $anchor",
        ))
        _nearest_impurity_ancestor(topology, site, impurity_sites) == anchor || throw(ArgumentError(
            "mounted bath site $site crosses another declared impurity site before $anchor",
        ))
    end
    ownership_hash = _mounted_ownership_hash(parametrization, topology,
                                             site_tuple, anchor_tuple)
    return (; physical, site_tuple, anchor_tuple, owner_tuple=canonical_owners, H,
            diagnostics=_mounted_diagnostics(diagnostics, ownership_hash))
end

function _anderson_bath(parametrization::AbstractHamiltonianBath,
                        topology::TreeTopology, phys::AbstractDict,
                        sites::AbstractVector{Symbol}, anchors::AbstractVector{Symbol},
                        H::OpSum,
                        certificate::Union{Nothing,_MountedHamiltonianCertificate};
                        diagnostics::NamedTuple=(;))
    fields = _validated_mounted_fields(parametrization, topology, phys, sites,
                                       anchors, H, diagnostics, :fermion)
    return AndersonBath(parametrization, topology, fields.physical,
                        fields.site_tuple, fields.anchor_tuple, fields.H,
                        fields.owner_tuple,
                        fields.diagnostics, certificate, Val(:validated))
end

function AndersonBath(parametrization::AbstractHamiltonianBath,
                      topology::TreeTopology, phys::AbstractDict,
                      sites::AbstractVector{Symbol}, anchors::AbstractVector{Symbol},
                      H::OpSum; diagnostics::NamedTuple=(;))
    return _anderson_bath(parametrization, topology, phys, sites, anchors, H,
                          nothing; diagnostics)
end

function _certified_anderson_bath(parametrization::AbstractHamiltonianBath,
                                  topology::TreeTopology, phys::AbstractDict,
                                  sites::AbstractVector{Symbol},
                                  anchors::AbstractVector{Symbol}, H::OpSum,
                                  certificate::_MountedHamiltonianCertificate;
                                  diagnostics::NamedTuple=(;))
    return _anderson_bath(parametrization, topology, phys, sites, anchors, H,
                          certificate; diagnostics)
end

function BosonBath(parametrization::AbstractHamiltonianBath,
                   topology::TreeTopology, phys::AbstractDict,
                   sites::AbstractVector{Symbol}, anchors::AbstractVector{Symbol},
                   H::OpSum; diagnostics::NamedTuple=(;))
    fields = _validated_mounted_fields(parametrization, topology, phys, sites,
                                       anchors, H, diagnostics, :boson)
    return BosonBath(parametrization, topology, fields.physical,
                     fields.site_tuple, fields.anchor_tuple, fields.H,
                     fields.diagnostics, Val(:validated))
end

Base.length(bath::AndersonBath) = length(bath.sites)
Base.length(bath::BosonBath) = length(bath.sites)
