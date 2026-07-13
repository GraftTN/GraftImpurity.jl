"""
    LoweredImpurityHamiltonian

Concrete output of complete impurity Hamiltonian assembly. `operator` has
already passed the mandatory abelian exact-rank Graft compression pipeline;
the uncompressed symbolic `opsum`, symmetry audit, interaction identity, and
concrete compression report remain available for checkpoint/audit consumers.
"""
struct LoweredImpurityHamiltonian{M<:AndersonBath,I<:AbstractImpurityInteraction,
                                  H<:OpSum,O<:TTNO,A<:SymmetryAudit,R,D<:NamedTuple}
    mounted::M
    interaction::I
    opsum::H
    operator::O
    audit::A
    compression::R
    diagnostics::D
end

function _mounted_physical_spaces(mounted::AndersonBath)
    return Dict{Symbol,ElementarySpace}(
        site => getproperty(mounted.phys, site) for site in propertynames(mounted.phys)
    )
end

function _require_mounted_topology_integrity(topology::TreeTopology)
    count = length(topology.ids)
    count > 0 && length(topology.parent) == count &&
        length(topology.children) == count && length(topology.depth) == count ||
        throw(ArgumentError("mounted topology has inconsistent node storage"))
    1 <= topology.root <= count || throw(ArgumentError(
        "mounted topology has an invalid root index",
    ))
    allunique(topology.ids) || throw(ArgumentError(
        "mounted topology has duplicate node labels",
    ))
    length(topology.index) == count || throw(ArgumentError(
        "mounted topology has an inconsistent label-index cache",
    ))
    for index in eachindex(topology.ids)
        get(topology.index, topology.ids[index], nothing) == index || throw(ArgumentError(
            "mounted topology label-index cache changed after mounting",
        ))
    end
    topology.parent[topology.root] == 0 && topology.depth[topology.root] == 0 ||
        throw(ArgumentError("mounted topology root cache changed after mounting"))
    observed_children = zeros(Int, count)
    for parent in eachindex(topology.children)
        allunique(topology.children[parent]) || throw(ArgumentError(
            "mounted topology repeats a child edge",
        ))
        for child in topology.children[parent]
            1 <= child <= count || throw(ArgumentError(
                "mounted topology has an out-of-range child index",
            ))
            topology.parent[child] == parent || throw(ArgumentError(
                "mounted topology parent/child caches disagree",
            ))
            observed_children[child] += 1
        end
    end
    for node in eachindex(topology.ids)
        if node == topology.root
            observed_children[node] == 0 || throw(ArgumentError(
                "mounted topology root has a parent edge",
            ))
        else
            parent = topology.parent[node]
            1 <= parent <= count && observed_children[node] == 1 || throw(ArgumentError(
                "mounted topology has an invalid parent cache",
            ))
            topology.depth[node] == topology.depth[parent] + 1 || throw(ArgumentError(
                "mounted topology depth cache changed after mounting",
            ))
        end
    end
    return nothing
end

function _validate_mounted_operator_spaces(mounted::AndersonBath,
                                           operators::ImpurityOperators)
    bath_layout(mounted.parametrization) == operators.layout || throw(ArgumentError(
        "mounted bath FlavorLayout must match ImpurityOperators.layout",
    ))
    for site in layout_sites(operators.layout)
        mounted_space = getproperty(mounted.phys, site)
        local_space = site_operators(operators, site).P
        mounted_space == local_space || throw(ArgumentError(
            "mounted bath and impurity operators use different sector spaces at $site",
        ))
    end
    return nothing
end

function _mounted_bath_owners(mounted::AndersonBath)
    owners = mounted.owners
    owners === nothing && throw(ArgumentError(
        "lower_hamiltonian requires frozen bath owner actions for symmetry auditing",
    ))
    return Tuple(site => owner for (site, owner) in zip(mounted.sites, owners))
end

function _same_bath_owners(left::Tuple, right::Tuple)
    length(left) == length(right) || return false
    return all(pair -> any(other -> other.first == pair.first &&
                            other.second == pair.second, right), left)
end

function _complete_symmetry_spec(spec::SymmetrySpec, mounted::AndersonBath)
    expected = _mounted_bath_owners(mounted)
    isempty(spec.bath_owners) || _same_bath_owners(spec.bath_owners, expected) ||
        throw(ArgumentError(
            "SymmetrySpec bath owners must exactly match the mounted bath ownership",
        ))
    return SymmetrySpec(spec.layout;
                        abelian=spec.abelian,
                        nonabelian=spec.nonabelian,
                        bath_owners=expected)
end

function _require_mounted_hamiltonian_integrity(mounted::AndersonBath)
    certificate = mounted.certificate
    certificate === nothing && throw(ArgumentError(
        "lower_hamiltonian requires a mount_bath-generated AndersonBath with a Hamiltonian integrity certificate",
    ))
    certificate.hamiltonian_hash == _opsum_integrity_hash(mounted.H) ||
        throw(ArgumentError(
            "mounted AndersonBath symbolic Hamiltonian changed after its integrity certificate was created",
        ))
    certificate.parametrization_hash ==
        _discrete_bath_integrity_hash(mounted.parametrization) || throw(ArgumentError(
            "mounted AndersonBath canonical bath data changed after its integrity certificate was created",
        ))
    return nothing
end

function _require_mounted_ownership_integrity(mounted::AndersonBath)
    hasproperty(mounted.diagnostics, :ownership_hash) || throw(ArgumentError(
        "lower_hamiltonian requires a mounted bath ownership integrity certificate",
    ))
    expected = _mounted_ownership_hash(mounted.parametrization, mounted.topology,
                                       mounted.sites, mounted.anchors)
    mounted.diagnostics.ownership_hash == expected || throw(ArgumentError(
        "mounted bath ownership data changed after mounting",
    ))
    frozen = _canonical_mounted_owners(mounted.parametrization)
    frozen === mounted.owners || throw(ArgumentError(
        "mounted bath owner actions changed after mounting",
    ))
    return nothing
end

function _require_charge_carrier(spec::SymmetrySpec, operators::ImpurityOperators)
    any(candidate -> candidate isa ChargeU1, spec.abelian) || return nothing
    operators.sector isa ParticleNumberSector || throw(ArgumentError(
        "a requested charge U(1) audit requires ParticleNumberSector() local operators",
    ))
    return nothing
end

function _validate_onebody_layout(onebody::Union{Nothing,ImpurityOneBody},
                                  layout::FlavorLayout, name::AbstractString)
    onebody === nothing && return nothing
    onebody.layout == layout || throw(ArgumentError(
        "$name FlavorLayout must match the interaction and mounted bath layout",
    ))
    return nothing
end

function _hamiltonian_diagnostics(interaction::AbstractImpurityInteraction,
                                  mounted::AndersonBath)
    kanamori_terms = interaction isa KanamoriInteraction ? interaction.terms : nothing
    return (
        basis=basis_identity(interaction.layout),
        interaction_hash=hash(interaction),
        kanamori_terms,
        ownership_hash=mounted.diagnostics.ownership_hash,
    )
end

function _nonempty_hamiltonian_opsum(H::OpSum, operators::ImpurityOperators)
    isempty(H.terms) || return H
    site = first(layout_sites(operators.layout))
    identity = site_operators(operators, site).I
    return H + Term(0.0, SiteOp(site, :I, identity))
end

"""
    lower_hamiltonian(mounted, interaction, operators;
                      h_loc=nothing, soc=nothing,
                      symmetry=SymmetrySpec(...), compression_atol, scheme=TruncationScheme())
        -> LoweredImpurityHamiltonian

Assemble the complete fermionic Hamiltonian, certify that its typed components
are Hermitian, audit requested full-Hamiltonian symmetries, construct a Graft
TTNO, and unconditionally invoke core sector-aware exact-rank compression.
No interaction-only compression or dense fallback is exposed.
"""
function lower_hamiltonian(mounted::AndersonBath,
                           interaction::AbstractImpurityInteraction,
                           operators::ImpurityOperators;
                           h_loc::Union{Nothing,ImpurityOneBody}=nothing,
                           soc::Union{Nothing,ImpurityOneBody}=nothing,
                           symmetry::SymmetrySpec=SymmetrySpec(interaction.layout),
                           compression_atol::Real,
                           scheme::TruncationScheme=TruncationScheme())
    mounted.parametrization isa DiscreteBath || throw(ArgumentError(
        "lower_hamiltonian currently requires an AndersonBath from canonical DiscreteBath data",
    ))
    bath_statistics(mounted.parametrization) === :fermion || throw(ArgumentError(
        "lower_hamiltonian currently supports fermionic AndersonBath values only",
    ))
    interaction.layout == operators.layout || throw(ArgumentError(
        "interaction FlavorLayout must match ImpurityOperators.layout",
    ))
    symmetry.layout == interaction.layout || throw(ArgumentError(
        "SymmetrySpec FlavorLayout must match the interaction layout",
    ))
    _require_mounted_topology_integrity(mounted.topology)
    _validate_mounted_operator_spaces(mounted, operators)
    _require_mounted_hamiltonian_integrity(mounted)
    _require_mounted_ownership_integrity(mounted)
    _validate_onebody_layout(h_loc, interaction.layout, "h_loc")
    _validate_onebody_layout(soc, interaction.layout, "soc")
    complete_spec = _complete_symmetry_spec(symmetry, mounted)
    _require_charge_carrier(complete_spec, operators)
    tolerance = Float64(compression_atol)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError(
        "compression_atol must be finite and nonnegative",
    ))

    H = mounted.H + one_body_opsum(h_loc, soc, operators, complete_spec) +
        lower_interaction(interaction, operators, complete_spec)
    H = _nonempty_hamiltonian_opsum(H, operators)
    audit = audit_symmetry(H, complete_spec; hermiticity=:certified)
    _require_supported_symmetry(audit)
    physical = _mounted_physical_spaces(mounted)
    operator = ttno_from_opsum(H, mounted.topology, physical; hermitian=true)
    report = compress!(operator; sector_aware=true, mode=:exact_rank,
                       compression_atol=tolerance, scheme)
    return LoweredImpurityHamiltonian(
        mounted, interaction, H, operator, audit, report,
        _hamiltonian_diagnostics(interaction, mounted),
    )
end
