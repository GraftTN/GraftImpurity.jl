"""
    MountedBath(parametrization, topology, phys, sites, anchors, H;
                diagnostics=(;))

Typed mounted realization of a canonical Hamiltonian bath. Every canonical
bath mode owns exactly one mounted site, in the same order as BathOrbitals.
anchors are explicit and never inferred from a largest coupling component.
"""
struct MountedBath{B<:AbstractHamiltonianBath,P<:NamedTuple,D<:NamedTuple} <:
        AbstractMountedBath
    parametrization::B
    topology::TreeTopology
    phys::P
    sites::Tuple{Vararg{Symbol}}
    anchors::Tuple{Vararg{Symbol}}
    H::OpSum
    diagnostics::D

    function MountedBath(parametrization::B, topology::TreeTopology,
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

function MountedBath(parametrization::AbstractHamiltonianBath,
                     topology::TreeTopology, phys::AbstractDict,
                     sites::AbstractVector{Symbol}, anchors::AbstractVector{Symbol},
                     H::OpSum; diagnostics::NamedTuple=(;))
    length(sites) == length(anchors) ||
        throw(DimensionMismatch("MountedBath needs one anchor per site"))
    length(sites) == length(parametrization) ||
        throw(DimensionMismatch("MountedBath needs one site per canonical bath mode"))
    allunique(sites) ||
        throw(ArgumentError("MountedBath site labels must be unique"))

    physical = _canonical_phys(phys)
    site_tuple = Tuple(Symbol.(sites))
    anchor_tuple = Tuple(Symbol.(anchors))
    for site in site_tuple
        _require_topology_node(topology, site, "site")
        hasproperty(physical, site) ||
            throw(ArgumentError("MountedBath site $site has no physical space"))
    end
    for anchor in anchor_tuple
        _require_topology_node(topology, anchor, "anchor")
    end
    return MountedBath(parametrization, topology, physical, site_tuple,
                       anchor_tuple, H, diagnostics, Val(:validated))
end
