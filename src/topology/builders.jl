# T3NS is the GraftImpurity name for the MT3N impurity topology
# ("minimal three-legged tree tensor network") from M. Grundner,
# "Tensor Network Impurity Solvers: Simulating Quantum Materials",
# PhD thesis, LMU Munich (2025), Ch. 6 (T3N framework developed with
# S. Mardazad, PhD thesis, LMU Munich, 2022). The three-legged
# representation itself is Gunst, Verstraete, Wouters, Legeza, and
# Van Neck, J. Chem. Theory Comput. 14, 2026 (2018),
# doi:10.1021/acs.jctc.8b00098. Grundner, Westhoff, Kugler, Parcollet,
# and Schollwöck, Phys. Rev. B 109, 155124 (2024),
# doi:10.1103/PhysRevB.109.155124 reuses the MT3N layout for complex-time
# evolution but does not define the term "MT3N".

"""Deterministic mounted-site label for one canonical bath mode."""
_bath_site_label(owner::Symbol, mode_index::Int) =
    Symbol(:bath_, owner, :_, mode_index)

"""Canonical bath-site labels in `BathOrbitals` storage order."""
function _canonical_bath_site_labels(bath::DiscreteBath)
    return Tuple(_bath_site_label(owner, mode_index)
                 for (mode_index, owner) in
                     enumerate(bath.orbitals.associated_flavors))
end

function _owner_mode_indices(bath::DiscreteBath,
                             flavor_order::Tuple{Vararg{Symbol}})
    groups = Dict(flavor => Int[] for flavor in flavor_order)
    for (mode_index, owner) in enumerate(bath.orbitals.associated_flavors)
        haskey(groups, owner) || throw(ArgumentError(
            "bath mode $mode_index has owner $owner absent from the topology plan",
        ))
        push!(groups[owner], mode_index)
    end
    return groups
end

function _validate_topology_inputs(plan::AbstractImpurityTopologyPlan,
                                   partition::Partition, bath::DiscreteBath)
    plan.layout == bath_layout(bath) ||
        throw(ArgumentError("topology plan FlavorLayout must match the bath layout"))
    bath_partition(bath) == partition ||
        throw(ArgumentError("topology Partition must match the bath partition"))
    validate_partition(partition, plan.layout)
    return nothing
end

function _require_flavor_spine_sites(layout::FlavorLayout,
                                     flavor_order::Tuple{Vararg{Symbol}})
    sites = Tuple(physical_site(layout, flavor) for flavor in flavor_order)
    allunique(sites) || throw(ArgumentError(
        "T3NS and FTPS require one declared physical site per flavor; " *
        "use explicit custom TreeTopology mounting for a shared local site",
    ))
    return sites
end

function _t3ns_junction_labels(tooth_count::Int)
    return Tuple(Symbol(:t3ns_junction_, index)
                 for index in 1:max(tooth_count - 2, 0))
end

function _validate_generated_labels(layout::FlavorLayout, bath::DiscreteBath,
                                    flavor_order::Tuple{Vararg{Symbol}};
                                    generated_labels::Tuple{Vararg{Symbol}}=())
    impurity_sites = _require_flavor_spine_sites(layout, flavor_order)
    bath_sites = _canonical_bath_site_labels(bath)
    allunique((impurity_sites..., bath_sites..., generated_labels...)) || throw(ArgumentError(
        "generated bath-site or T3NS junction labels collide with declared sites",
    ))
    return impurity_sites, bath_sites
end

function _append_bath_chain_edges!(edges::Vector{Pair{Symbol,Symbol}},
                                   layout::FlavorLayout, bath::DiscreteBath,
                                   flavor_order::Tuple{Vararg{Symbol}})
    grouped_indices = _owner_mode_indices(bath, flavor_order)
    for owner in flavor_order
        previous = physical_site(layout, owner)
        for mode_index in grouped_indices[owner]
            site = _bath_site_label(owner, mode_index)
            push!(edges, previous => site)
            previous = site
        end
    end
    return edges
end

function _t3ns_backbone_edges(layout::FlavorLayout,
                              flavor_order::Tuple{Vararg{Symbol}},
                              tooth_presence::Tuple{Vararg{Bool}})
    length(flavor_order) == length(tooth_presence) || throw(DimensionMismatch(
        "T3NS tooth-presence data must match the declared flavor order",
    ))
    root = physical_site(layout, first(flavor_order))
    length(flavor_order) == 1 && return root, Pair{Symbol,Symbol}[]

    # A nonroot physical node with a bath tooth cannot continue the impurity
    # backbone without exceeding two virtual legs.  Zero-tooth physical nodes
    # are therefore promoted into the connector path.  The remaining toothed
    # physical nodes are leaves of a minimal binary junction comb.  This uses
    # exactly max(number_of_teeth - 2, 0) pure branching nodes.
    connector_flavors = Symbol[]
    leaf_flavors = Symbol[]
    for position in 2:length(flavor_order)
        if tooth_presence[position]
            push!(leaf_flavors, flavor_order[position])
        else
            push!(connector_flavors, flavor_order[position])
        end
    end
    edges = Pair{Symbol,Symbol}[]
    open_parents = Symbol[]
    append!(open_parents, fill(root, tooth_presence[1] ? 1 : 2))
    connector_parent = root
    for flavor in connector_flavors
        parent_position = findlast(==(connector_parent), open_parents)
        parent_position === nothing && throw(ArgumentError(
            "T3NS connector construction exhausted the physical-leg budget",
        ))
        deleteat!(open_parents, parent_position)
        connector = physical_site(layout, flavor)
        push!(edges, connector_parent => connector)
        push!(open_parents, connector)
        connector_parent = connector
    end

    junction_index = 0
    while length(leaf_flavors) > length(open_parents)
        parent = pop!(open_parents)
        junction_index += 1
        junction = Symbol(:t3ns_junction_, junction_index)
        push!(edges, parent => junction)
        push!(open_parents, junction, junction)
    end
    for (flavor, parent) in zip(leaf_flavors, open_parents)
        push!(edges, parent => physical_site(layout, flavor))
    end
    return root, edges
end

"""
    impurity_topology(plan::T3NS, partition, bath) -> TreeTopology

Build the ownership-preserving MT3N/T3NS route.  One bath physical site is
created per canonical mode and each mode is placed in the branch selected by
its explicit `associated_flavor`; full coupling vectors are deliberately not
used for placement and remain in `bath` for Hamiltonian lowering.
"""
function impurity_topology(plan::T3NS, partition::Partition,
                           bath::DiscreteBath)
    _validate_topology_inputs(plan, partition, bath)
    owner_indices = _owner_mode_indices(bath, plan.flavor_order)
    tooth_presence = Tuple(!isempty(owner_indices[flavor])
                           for flavor in plan.flavor_order)
    impurity_sites, bath_sites = _validate_generated_labels(
        plan.layout, bath, plan.flavor_order;
        generated_labels=_t3ns_junction_labels(count(identity, tooth_presence)),
    )
    root, edges = _t3ns_backbone_edges(
        plan.layout, plan.flavor_order, tooth_presence,
    )
    _append_bath_chain_edges!(edges, plan.layout, bath, plan.flavor_order)
    topology = TreeTopology(root, edges)
    physical = Symbol[impurity_sites...; bath_sites...]
    Graft.is_t3ns(topology; physical) || throw(ArgumentError(
        "constructed T3NS violates the core three-legged physical-node invariant",
    ))
    return topology
end

# FTPS is the per-spin-orbital-flavor fork topology of Bauernfeind, Zingl,
# Triebl, Aichhorn, and Evertz, Phys. Rev. X 7, 031013 (2017).  Each ordered
# flavor gets one impurity spine node and one separately owned bath tooth.
# This construction intentionally retains full off-diagonal coupling vectors in
# the Hamiltonian; topology ownership is never inferred from |V|.
"""
    impurity_topology(plan::FTPS, partition, bath) -> TreeTopology

Build the production flavor-spine FTPS route with variable-length ownership
teeth.  Unlike T3NS, an interior physical spine node is not expected to satisfy
the three-legged physical-tensor predicate.
"""
function impurity_topology(plan::FTPS, partition::Partition,
                           bath::DiscreteBath)
    _validate_topology_inputs(plan, partition, bath)
    impurity_sites, _ = _validate_generated_labels(plan.layout, bath,
                                                    plan.flavor_order)
    root = first(impurity_sites)
    edges = Pair{Symbol,Symbol}[]
    for position in 2:length(impurity_sites)
        push!(edges, impurity_sites[position - 1] => impurity_sites[position])
    end
    _append_bath_chain_edges!(edges, plan.layout, bath, plan.flavor_order)
    return TreeTopology(root, edges)
end
