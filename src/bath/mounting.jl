function _topology_edges(topology::TreeTopology)
    return Pair{Symbol,Symbol}[
        Graft.Trees.nodeid(topology, topology.parent[node]) =>
            Graft.Trees.nodeid(topology, node)
        for node in 2:Graft.Trees.nnodes(topology)
    ]
end

function _mount_site_labels(bath::DiscreteBath, site_labels)
    if site_labels === nothing
        return _canonical_bath_site_labels(bath)
    end
    labels = Tuple(Symbol.(site_labels))
    length(labels) == length(bath) ||
        throw(DimensionMismatch("mount site_labels needs one label per canonical bath mode"))
    allunique(labels) ||
        throw(ArgumentError("mount site_labels must be unique"))
    return labels
end

function _validate_impurity_nodes(topology::TreeTopology, layout::FlavorLayout)
    for site in layout_sites(layout)
        try
            Graft.Trees.nodeindex(topology, site)
        catch err
            err isa KeyError || rethrow()
            throw(ArgumentError("mount topology is missing declared impurity site $site"))
        end
    end
    return nothing
end

function _topology_with_mounted_sites(topology::TreeTopology, bath::DiscreteBath,
                                      sites::Tuple{Vararg{Symbol}})
    _validate_impurity_nodes(topology, bath_layout(bath))
    present = map(site -> haskey(topology.index, site), sites)
    all(present) && return topology, :prebuilt
    any(present) && throw(ArgumentError(
        "mount topology contains only a subset of the requested bath site labels",
    ))

    edges = _topology_edges(topology)
    grouped_indices = _owner_mode_indices(bath, flavors(bath_layout(bath)))
    for owner in flavors(bath_layout(bath))
        previous = physical_site(bath_layout(bath), owner)
        for mode_index in grouped_indices[owner]
            site = sites[mode_index]
            push!(edges, previous => site)
            previous = site
        end
    end
    return TreeTopology(Graft.Trees.nodeid(topology, topology.root), edges), :extended
end

function _mount_layout_operators(layout::FlavorLayout,
                                 sector::AbstractFermionSector)
    return Dict(site => FermionSiteOperators(layout, site; sector)
                for site in layout_sites(layout))
end

function _mount_phys(layout_operators::AbstractDict{Symbol,<:FermionSiteOperators},
                     sites::Tuple{Vararg{Symbol}}, bath_operators::FermionSiteOperators)
    phys = Dict{Symbol,ElementarySpace}()
    for (site, operators) in layout_operators
        phys[site] = operators.P
    end
    for site in sites
        phys[site] = bath_operators.P
    end
    return phys
end

function _mount_hamiltonian(bath::DiscreteBath,
                            sites::Tuple{Vararg{Symbol}},
                            layout_operators::AbstractDict{Symbol,<:FermionSiteOperators},
                            bath_operators::FermionSiteOperators)
    H = OpSum()
    retained_couplings = 0
    orbitals = bath_orbitals(bath)
    partition = bath_partition(bath)
    for mode_index in eachindex(orbitals.energies)
        bath_site = sites[mode_index]
        H += Term(orbitals.energies[mode_index],
                  SiteOp(bath_site, :N, bath_operators.N[1]))
        block = block_names(partition)[orbitals.block_indices[mode_index]]
        block_order = block_flavors(partition, block)
        coupling = orbitals.couplings[mode_index]
        for (component, flavor) in enumerate(block_order)
            value = coupling[component]
            iszero(value) && continue
            impurity_site = physical_site(bath_layout(bath), flavor)
            impurity_operators = layout_operators[impurity_site]
            H += Term(value,
                      SiteOp(impurity_site, Symbol(:Cd_, flavor),
                             local_creator(impurity_operators, flavor)),
                      SiteOp(bath_site, :C, bath_operators.C[1]))
            H += Term(conj(value),
                      SiteOp(impurity_site, Symbol(:C_, flavor),
                             local_annihilator(impurity_operators, flavor)),
                      SiteOp(bath_site, :Cd, bath_operators.Cd[1]))
            retained_couplings += 1
        end
    end
    return H, retained_couplings
end

function _opsum_integrity_hash(H::OpSum)
    state = hash(:GraftImpurityOpSumIntegrity)
    for term in H.terms
        state = hash(term.coeff, state)
        for operator in term.ops
            state = hash((operator.site, operator.name, operator.charge), state)
            values = convert(Array, operator.op)
            state = hash(size(values), state)
            for value in values
                state = hash(value, state)
            end
        end
    end
    return state
end

function _discrete_bath_integrity_hash(bath::DiscreteBath)
    orbitals = bath_orbitals(bath)
    state = hash((:GraftImpurityDiscreteBathIntegrity, bath_layout(bath),
                  bath_partition(bath), bath_statistics(bath)))
    for mode in eachindex(orbitals.energies)
        state = hash(orbitals.energies[mode], state)
        state = hash(orbitals.pole_indices[mode], state)
        state = hash(orbitals.block_indices[mode], state)
        state = hash(orbitals.associated_flavors[mode], state)
        for value in orbitals.couplings[mode]
            state = hash(value, state)
        end
    end
    return state
end

function _mount_diagnostics(user::NamedTuple, topology_source::Symbol,
                            retained_couplings::Int, H::OpSum)
    required = (; kind=:anderson, topology_source, retained_couplings,
                hamiltonian_hash=_opsum_integrity_hash(H))
    any(name -> name in keys(required), keys(user)) && throw(ArgumentError(
        "mount diagnostics may not overwrite canonical ownership fields",
    ))
    return merge(required, user)
end

"""
    mount_bath(topology, bath::DiscreteBath; site_labels=nothing,
               sector=ParticleNumberSector(), diagnostics=(;))
        -> AndersonBath

Mount a canonical fermionic star bath without changing its one-particle basis.
Each canonical mode occurs once, stays below the physical site of its explicit
`associated_flavor`, and contributes every nonzero component of its complete
block-local coupling vector to `H`.  Supplying a custom topology is supported:
when none of the requested bath labels is present, deterministic owner chains
are appended; a partial pre-existing label set is rejected.

Bosonic `DiscreteBath` values need an explicit cutoff and matter-operator
convention and therefore are rejected here rather than being interpreted as
fermionic sites.
"""
function mount_bath(topology::TreeTopology, bath::DiscreteBath;
                    site_labels=nothing,
                    sector::AbstractFermionSector=ParticleNumberSector(),
                    diagnostics::NamedTuple=(;))
    bath_statistics(bath) === :fermion || throw(ArgumentError(
        "mount_bath currently requires a fermionic DiscreteBath; " *
        "construct BosonBath with explicit cutoff/operator data instead",
    ))
    sites = _mount_site_labels(bath, site_labels)
    impurity_sites = layout_sites(bath_layout(bath))
    any(site -> site in impurity_sites, sites) && throw(ArgumentError(
        "mount bath-site labels must not reuse declared impurity sites",
    ))
    mounted_topology, topology_source = _topology_with_mounted_sites(
        topology, bath, sites,
    )
    anchors = Tuple(physical_site(bath_layout(bath), owner)
                    for owner in bath_orbitals(bath).associated_flavors)
    for (site, anchor) in zip(sites, anchors)
        _mounted_site_has_ancestor(mounted_topology, site, anchor) || throw(ArgumentError(
            "bath site $site is not in the declared owner arm rooted at $anchor",
        ))
    end

    layout_operators = _mount_layout_operators(bath_layout(bath), sector)
    bath_operators = FermionSiteOperators([:bath_mode]; sector)
    phys = _mount_phys(layout_operators, sites, bath_operators)
    H, retained_couplings = _mount_hamiltonian(
        bath, sites, layout_operators, bath_operators,
    )
    certificate = _MountedHamiltonianCertificate(
        _opsum_integrity_hash(H), _discrete_bath_integrity_hash(bath),
    )
    return _certified_anderson_bath(
        bath, mounted_topology, phys, collect(sites), collect(anchors), H,
        certificate;
        diagnostics=_mount_diagnostics(
            diagnostics, topology_source, retained_couplings, H,
        ),
    )
end
