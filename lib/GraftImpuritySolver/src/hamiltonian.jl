"""
    AbstractTTNOBuilder

Typed policy for constructing the complete impurity TTNO before the mandatory
compression pipeline. Builder selection is solver-lifetime configuration, not
part of an individual solve request.
"""
abstract type AbstractTTNOBuilder end

"""
    LegacyTTNOBuilder()

Build through the direct `GraftTTNOBuild.ttno_from_opsum` interface. This
remains the default while the compiled path is validated on impurity
Hamiltonians.
"""
struct LegacyTTNOBuilder <: AbstractTTNOBuilder end

Base.:(==)(::LegacyTTNOBuilder, ::LegacyTTNOBuilder) = true
Base.isequal(::LegacyTTNOBuilder, ::LegacyTTNOBuilder) = true
Base.hash(::LegacyTTNOBuilder, seed::UInt) = hash(:LegacyTTNOBuilder, seed)

"""
    CompiledTTNOBuilder(; lowering=AbelianScalarLowering(),
                        merge=StateDiagramMerge(SGEOptimizer()))

Build through the direct `GraftStateDiagram.compile_ttno` interface. The
lowering and merge kernels are part of the policy identity, so
`DirectSumMerge()` can serve as an uncompressed correctness oracle
independently of the optimized StateDiagram kernels.
"""
struct CompiledTTNOBuilder{
        L<:AbstractOperatorLoweringKernel,M<:AbstractTTNOMergeKernel} <:
        AbstractTTNOBuilder
    lowering::L
    merge::M
end

function CompiledTTNOBuilder(;
        lowering::AbstractOperatorLoweringKernel=AbelianScalarLowering(),
        merge::AbstractTTNOMergeKernel=StateDiagramMerge(SGEOptimizer()))
    return CompiledTTNOBuilder(lowering, merge)
end

Base.:(==)(left::CompiledTTNOBuilder, right::CompiledTTNOBuilder) =
    left.lowering == right.lowering && left.merge == right.merge
Base.isequal(left::CompiledTTNOBuilder, right::CompiledTTNOBuilder) =
    isequal(left.lowering, right.lowering) && isequal(left.merge, right.merge)
Base.hash(builder::CompiledTTNOBuilder, seed::UInt) =
    hash(builder.merge, hash(builder.lowering, hash(:CompiledTTNOBuilder, seed)))

"""
    TTNOBuilderCapabilityError

Fail-closed compiled-builder error retaining GraftStateDiagram's typed
category-capability failure while pointing callers to the explicit legacy
migration oracle.
"""
struct TTNOBuilderCapabilityError{
        B<:CompiledTTNOBuilder,E<:MissingCategoryCapability} <: Exception
    builder::B
    cause::E
end

function Base.showerror(io::IO, error::TTNOBuilderCapabilityError)
    print(io, "CompiledTTNOBuilder cannot lower this category profile: ")
    showerror(io, error.cause)
    print(io, ". Retry explicitly with LegacyTTNOBuilder().")
end

"""
    build_ttno(builder, H, topology, physical; hermitian=false, elt=ComplexF64)
        -> (operator, build_report_or_nothing, exact_provenance_or_nothing)

Uniform impurity-facing TTNO construction contract. Compiler-certified exact
provenance is returned when the selected direct owner can supply it.
"""
function build_ttno(::LegacyTTNOBuilder, H::OpSum, topology::TreeTopology,
                    physical::Dict{Symbol,<:ElementarySpace};
                    hermitian::Bool=false,
                    elt::Type{<:Number}=ComplexF64)
    operator = ttno_from_opsum(H, topology, physical; hermitian, elt)
    return operator, nothing, nothing
end

function build_ttno(builder::CompiledTTNOBuilder, H::OpSum,
                    topology::TreeTopology,
                    physical::Dict{Symbol,<:ElementarySpace};
                    hermitian::Bool=false,
                    elt::Type{<:Number}=ComplexF64)
    try
        return compile_ttno(
            H, topology, physical;
            lowering=builder.lowering, merge=builder.merge, hermitian, elt,
        )
    catch error
        error isa MissingCategoryCapability || rethrow()
        throw(TTNOBuilderCapabilityError(builder, error))
    end
end

"""
    LoweredImpurityHamiltonian

Concrete output of complete impurity Hamiltonian assembly. `operator` has
already passed the mandatory abelian exact-rank Graft compression pipeline;
the uncompressed symbolic `opsum`, builder policy, build report, exact build
provenance, symmetry audit, interaction identity, and concrete compression
report remain available for checkpoint/audit consumers.
"""
struct LoweredImpurityHamiltonian{M<:AbstractMountedBath,I<:AbstractImpurityInteraction,
                                  H<:OpSum,O<:TTNO,B<:AbstractTTNOBuilder,
                                  R,P,A<:SymmetryAudit,C,D<:NamedTuple}
    mounted::M
    interaction::I
    opsum::H
    operator::O
    builder::B
    build_report::R
    build_provenance::P
    audit::A
    compression::C
    diagnostics::D
end

function _mounted_physical_spaces(mounted::AbstractMountedBath)
    return Dict{Symbol,ElementarySpace}(
        site => getproperty(mounted.phys, site) for site in propertynames(mounted.phys)
    )
end

_mounted_layout(mounted::AndersonBath) = bath_layout(mounted.parametrization)
_mounted_layout(mounted::CayleyAndersonBath) = bath_layout(mounted.mapping.mapped)

_mounted_bath_integrity_hash(mounted::AndersonBath) =
    _discrete_bath_integrity_hash(mounted.parametrization)
_mounted_bath_integrity_hash(mounted::CayleyAndersonBath) =
    _cayley_mapping_integrity_hash(mounted.mapping)

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

function _validate_mounted_operator_spaces(mounted::Union{AndersonBath,CayleyAndersonBath},
                                           operators::ImpurityOperators)
    _mounted_layout(mounted) == operators.layout || throw(ArgumentError(
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

function _complete_symmetry_spec(spec::SymmetrySpec, mounted::CayleyAndersonBath)
    isempty(spec.bath_owners) || throw(ArgumentError(
        "Cayley mapped bath sites do not carry inferred flavor-owner actions; " *
        "request only generators that can be audited from the mapped Hamiltonian",
    ))
    return spec
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

function _require_mounted_hamiltonian_integrity(mounted::CayleyAndersonBath)
    certificate = mounted.certificate
    certificate.hamiltonian_hash == _opsum_integrity_hash(mounted.H) ||
        throw(ArgumentError(
            "mounted CayleyAndersonBath symbolic Hamiltonian changed after mounting",
        ))
    certificate.parametrization_hash == _cayley_mapping_integrity_hash(mounted.mapping) ||
        throw(ArgumentError(
            "mounted Cayley mapping data changed after mounting",
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

function _require_mounted_ownership_integrity(mounted::CayleyAndersonBath)
    hasproperty(mounted.diagnostics, :mapping_hash) &&
        hasproperty(mounted.diagnostics, :ownership_hash) || throw(ArgumentError(
            "lower_hamiltonian requires a Cayley mapping ownership certificate",
        ))
    mapping_hash = _cayley_mapping_integrity_hash(mounted.mapping)
    mounted.diagnostics.mapping_hash == mapping_hash || throw(ArgumentError(
        "mounted Cayley mapping provenance changed after mounting",
    ))
    expected = _cayley_mounted_ownership_hash(mapping_hash, mounted.topology,
                                               mounted.sites)
    mounted.diagnostics.ownership_hash == expected || throw(ArgumentError(
        "mounted Cayley topology or ownership provenance changed after mounting",
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
                                  mounted::Union{AndersonBath,CayleyAndersonBath})
    kanamori_terms = interaction isa KanamoriInteraction ? interaction.terms : nothing
    return (
        basis=basis_identity(interaction.layout),
        interaction_hash=hash(interaction),
        kanamori_terms,
        ownership_hash=mounted.diagnostics.ownership_hash,
    )
end

function _validate_lowerable_bath(mounted::AndersonBath)
    mounted.parametrization isa DiscreteBath || throw(ArgumentError(
        "lower_hamiltonian requires an AndersonBath from canonical DiscreteBath data",
    ))
    bath_statistics(mounted.parametrization) === :fermion || throw(ArgumentError(
        "lower_hamiltonian currently supports fermionic AndersonBath values only",
    ))
    return nothing
end

function _validate_lowerable_bath(mounted::CayleyAndersonBath)
    bath_statistics(mounted.mapping.mapped) === :fermion || throw(ArgumentError(
        "lower_hamiltonian currently supports fermionic Cayley mapped baths only",
    ))
    return nothing
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
                      symmetry=SymmetrySpec(...),
                      ttno_builder=LegacyTTNOBuilder(),
                      compression_atol, scheme=TruncationScheme())
        -> LoweredImpurityHamiltonian

Assemble the complete fermionic Hamiltonian, certify that its typed components
are Hermitian, audit requested full-Hamiltonian symmetries, construct a Graft
TTNO through the selected typed builder, and unconditionally invoke core
sector-aware exact-rank compression. Compiler-certified exact provenance is
passed into compression when available. No interaction-only compression,
automatic legacy fallback, or dense fallback is exposed.
"""
function lower_hamiltonian(mounted::Union{AndersonBath,CayleyAndersonBath},
                           interaction::AbstractImpurityInteraction,
                           operators::ImpurityOperators;
                           h_loc::Union{Nothing,ImpurityOneBody}=nothing,
                           soc::Union{Nothing,ImpurityOneBody}=nothing,
                           symmetry::SymmetrySpec=SymmetrySpec(interaction.layout),
                           ttno_builder::AbstractTTNOBuilder=LegacyTTNOBuilder(),
                           compression_atol::Real,
                           scheme::TruncationScheme=TruncationScheme())
    _validate_lowerable_bath(mounted)
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
    operator, build_report, build_provenance = build_ttno(
        ttno_builder, H, mounted.topology, physical; hermitian=true,
    )
    report = compress!(operator; sector_aware=true, mode=:exact_rank,
                       compression_atol=tolerance, scheme,
                       provenance=build_provenance)
    return LoweredImpurityHamiltonian(
        mounted, interaction, H, operator, ttno_builder,
        build_report, build_provenance, audit, report,
        _hamiltonian_diagnostics(interaction, mounted),
    )
end
