abstract type AbstractSymmetryCandidate end

"""The actual particle-number `U(1)_N` candidate for one FlavorLayout."""
struct ChargeU1{L<:FlavorLayout} <: AbstractSymmetryCandidate
    layout::L
end

"""
    FlavorU1(name, weights, layout)

Explicit diagonal abelian generator in the declared flavor basis. It can model
`S_z`, `N_up`, `N_down`, or a supplied axial `J_z`; its weights are never
inferred from flavor names. The current production carrier encodes only
`ChargeU1`, so a preserved `FlavorU1` reports `lowering_status=:audit_only`
until an explicit fZ2 × `U(1)_N` × axial carrier is supplied.
"""
struct FlavorU1{L<:FlavorLayout,W<:Tuple{Vararg{Float64}}} <:
        AbstractSymmetryCandidate
    name::Symbol
    weights::W
    layout::L
end

function FlavorU1(name::Symbol, weights::AbstractVector{<:Real},
                  layout::FlavorLayout)
    isempty(String(name)) && throw(ArgumentError(
        "FlavorU1 name must be nonempty",
    ))
    canonical = Tuple(Float64.(weights))
    length(canonical) == length(flavors(layout)) || throw(DimensionMismatch(
        "FlavorU1 needs one weight per FlavorLayout flavor",
    ))
    all(isfinite, canonical) || throw(ArgumentError(
        "FlavorU1 weights must be finite",
    ))
    return FlavorU1(name, canonical, layout)
end

"""
    SU2Reduce(layout; name=:su2, axial_generator=nothing)

Typed non-abelian sector candidate. Its mathematical presence can be recorded
only after an explicit diagonal axial generator has survived the complete
Hamiltonian audit. Graft's current abelian TTNO path cannot lower or compress
the non-abelian candidate.
"""
struct SU2Reduce{L<:FlavorLayout,A<:Union{Nothing,FlavorU1}} <:
        AbstractSymmetryCandidate
    name::Symbol
    layout::L
    axial_generator::A
end

function SU2Reduce(layout::FlavorLayout;
                   name::Symbol=:su2,
                   axial_generator::Union{Nothing,FlavorU1}=nothing)
    isempty(String(name)) && throw(ArgumentError("SU2Reduce name must be nonempty"))
    axial_generator === nothing || axial_generator.layout == layout || throw(ArgumentError(
        "SU2Reduce axial_generator must use the declared FlavorLayout",
    ))
    return SU2Reduce(name, layout, axial_generator)
end

_candidate_name(candidate::ChargeU1) = :charge
_candidate_name(candidate::FlavorU1) = candidate.name
_candidate_name(candidate::SU2Reduce) = candidate.name
_candidate_layout(candidate::AbstractSymmetryCandidate) = candidate.layout

"""
    SymmetrySpec(layout; abelian=(ChargeU1(layout),), nonabelian=(), bath_owners=())

Explicit candidates for a full-Hamiltonian audit. `bath_owners` maps mounted
bath-site labels to their declared owner flavor, supplying a generator action
without guessing it from a site label.
"""
struct SymmetrySpec{L<:FlavorLayout,A<:Tuple,N<:Tuple,B<:Tuple}
    layout::L
    abelian::A
    nonabelian::N
    bath_owners::B
end

function SymmetrySpec(layout::FlavorLayout;
                      abelian=(ChargeU1(layout),), nonabelian=(),
                      bath_owners=Pair{Symbol,Symbol}[])
    abelian_candidates = Tuple(abelian)
    nonabelian_candidates = Tuple(nonabelian)
    all(candidate -> candidate isa Union{ChargeU1,FlavorU1}, abelian_candidates) ||
        throw(ArgumentError("SymmetrySpec abelian candidates must be ChargeU1 or FlavorU1"))
    all(candidate -> candidate isa SU2Reduce, nonabelian_candidates) ||
        throw(ArgumentError("SymmetrySpec nonabelian candidates must be SU2Reduce"))
    all(candidate -> _candidate_layout(candidate) == layout,
        (abelian_candidates..., nonabelian_candidates...)) || throw(ArgumentError(
        "every SymmetrySpec candidate must use its declared FlavorLayout",
    ))
    names = Symbol[_candidate_name(candidate)
                   for candidate in (abelian_candidates..., nonabelian_candidates...)]
    allunique(names) || throw(ArgumentError(
        "SymmetrySpec candidate names must be unique",
    ))
    owners = Tuple(begin
        site, flavor = Symbol(pair.first), Symbol(pair.second)
        flavor in flavors(layout) || throw(ArgumentError(
            "SymmetrySpec bath owner $flavor is absent from its FlavorLayout",
        ))
        site => flavor
    end for pair in bath_owners)
    allunique(first.(owners)) || throw(ArgumentError(
        "SymmetrySpec bath-site labels must be unique",
    ))
    return SymmetrySpec(layout, abelian_candidates, nonabelian_candidates, owners)
end

"""
One requested symmetry candidate's complete-Hamiltonian audit result.

`status` is `:preserved`, `:broken`, or `:unavailable` for abelian candidates;
an axial-screened `SU2Reduce` uses `:candidate` when it survives that necessary
screen. `lowering_status` separately reports `:carried`, `:audit_only`, or
`:unsupported`, so an audit never implies an unavailable sector lowering.
"""
struct SymmetryAuditItem
    name::Symbol
    status::Symbol
    lowering_status::Symbol
    maximum_violation::Float64
    unknown_terms::Int
end

struct SymmetryAudit{A<:Tuple,N<:Tuple}
    hermiticity::Symbol
    abelian::A
    nonabelian::N
end

function _generator_weight(candidate::ChargeU1, flavor::Symbol)
    return 1.0
end

function _generator_weight(candidate::FlavorU1, flavor::Symbol)
    return candidate.weights[flavor_index(candidate.layout, flavor)]
end

function _bath_owner(spec::SymmetrySpec, site::Symbol)
    for pair in spec.bath_owners
        pair.first == site && return pair.second
    end
    return nothing
end

function _local_factor_delta(encoded::AbstractString,
                             candidate::Union{ChargeU1,FlavorU1})
    startswith(encoded, "local_") || return nothing
    body = encoded[length("local_") + 1:end]
    isempty(body) && return nothing
    delta = 0.0
    for factor in split(body, "__")
        if startswith(factor, "cd_")
            index = tryparse(Int, factor[length("cd_") + 1:end])
            index === nothing && return nothing
            1 <= index <= length(flavors(candidate.layout)) || return nothing
            flavor = flavors(candidate.layout)[index]
            delta += _generator_weight(candidate, flavor)
        elseif startswith(factor, "c_")
            index = tryparse(Int, factor[length("c_") + 1:end])
            index === nothing && return nothing
            1 <= index <= length(flavors(candidate.layout)) || return nothing
            flavor = flavors(candidate.layout)[index]
            delta -= _generator_weight(candidate, flavor)
        else
            return nothing
        end
    end
    return delta
end

function _siteop_delta(operator::SiteOp,
                       candidate::Union{ChargeU1,FlavorU1},
                       spec::SymmetrySpec)
    encoded = String(operator.name)
    startswith(encoded, "local_") && return _local_factor_delta(encoded, candidate)
    if startswith(encoded, "Cd_")
        return _generator_weight(candidate, Symbol(encoded[length("Cd_") + 1:end]))
    elseif startswith(encoded, "C_")
        return -_generator_weight(candidate, Symbol(encoded[length("C_") + 1:end]))
    elseif startswith(encoded, "cayley_Cd_")
        return candidate isa ChargeU1 ? 1.0 : nothing
    elseif startswith(encoded, "cayley_C_")
        return candidate isa ChargeU1 ? -1.0 : nothing
    elseif startswith(encoded, "cayley_bilinear_")
        return candidate isa ChargeU1 ? 0.0 : nothing
    elseif encoded == "Cd" || encoded == "C"
        owner = _bath_owner(spec, operator.site)
        owner === nothing && return candidate isa ChargeU1 ?
            (encoded == "Cd" ? 1.0 : -1.0) : nothing
        sign = encoded == "Cd" ? 1.0 : -1.0
        return sign * _generator_weight(candidate, owner)
    elseif encoded in ("N", "I")
        return 0.0
    end
    return nothing
end

function _term_signature(term::Term)
    return Tuple(sort([(operator.site, operator.name, operator.charge)
                       for operator in term.ops];
                      by=entry -> (String(entry[1]), String(entry[2]),
                                   string(entry[3]))))
end

function _coalesced_terms(H::OpSum)
    terms = Dict{Tuple,Tuple{ComplexF64,Vector{SiteOp}}}()
    for term in H
        key = _term_signature(term)
        coefficient, operators = get(terms, key, (0.0 + 0.0im, copy(term.ops)))
        updated = coefficient + ComplexF64(term.coeff)
        if iszero(updated)
            delete!(terms, key)
        else
            terms[key] = (updated, operators)
        end
    end
    return values(terms)
end

function _audit_abelian(H::OpSum, candidate::Union{ChargeU1,FlavorU1},
                         spec::SymmetrySpec, tolerance::Float64)
    maximum_violation = 0.0
    unknown_terms = 0
    for (coefficient, operators) in _coalesced_terms(H)
        delta = 0.0
        known = true
        for operator in operators
            contribution = _siteop_delta(operator, candidate, spec)
            if contribution === nothing
                known = false
                break
            end
            delta += contribution
        end
        if known
            maximum_violation = max(maximum_violation, abs(coefficient * delta))
        else
            unknown_terms += 1
        end
    end
    status = unknown_terms > 0 ? :unavailable :
        (maximum_violation <= tolerance ? :preserved : :broken)
    lowering_status = candidate isa ChargeU1 ? :carried : :audit_only
    return SymmetryAuditItem(_candidate_name(candidate), status, lowering_status,
                             maximum_violation, unknown_terms)
end

function _audit_nonabelian(H::OpSum, candidate::SU2Reduce,
                           spec::SymmetrySpec, tolerance::Float64)
    axis = candidate.axial_generator
    axis === nothing && return SymmetryAuditItem(
        _candidate_name(candidate), :unavailable, :unsupported, 0.0, 0,
    )
    axial = _audit_abelian(H, axis, spec, tolerance)
    status = axial.status === :preserved ? :candidate : axial.status
    return SymmetryAuditItem(_candidate_name(candidate), status, :unsupported,
                             axial.maximum_violation, axial.unknown_terms)
end

"""
    audit_symmetry(H, spec; hermiticity=:unverified) -> SymmetryAudit

Audit the complete symbolic Hamiltonian against explicit candidate generators.
`hermiticity=:certified` is reserved for a builder that has assembled validated
Hermitian components; generic external `OpSum` inputs remain unverified rather
than receiving a dense fallback. `tolerance` is explicit: its zero default
requires exact symbolic conservation after coefficient coalescing.

`SU2Reduce` requires `axial_generator` metadata for a necessary axial screen.
An intact screen reports `status=:candidate` and
`lowering_status=:unsupported`; this is not a false claim that non-abelian
lowering has occurred.
"""
function audit_symmetry(H::OpSum, spec::SymmetrySpec;
                        hermiticity::Symbol=:unverified,
                        tolerance::Real=0.0)
    hermiticity in (:unverified, :certified, :failed) || throw(ArgumentError(
        "SymmetryAudit hermiticity must be :unverified, :certified, or :failed",
    ))
    audit_tolerance = Float64(tolerance)
    isfinite(audit_tolerance) && audit_tolerance >= 0 || throw(ArgumentError(
        "symmetry tolerance must be finite and nonnegative",
    ))
    abelian = Tuple(_audit_abelian(H, candidate, spec, audit_tolerance)
                     for candidate in spec.abelian)
    nonabelian = Tuple(_audit_nonabelian(H, candidate, spec, audit_tolerance)
                        for candidate in spec.nonabelian)
    return SymmetryAudit(hermiticity, abelian, nonabelian)
end

function _require_supported_symmetry(audit::SymmetryAudit)
    any(item -> item.status === :broken, audit.abelian) && throw(ArgumentError(
        "requested abelian symmetry is broken by the complete Hamiltonian",
    ))
    any(item -> item.status === :unavailable, audit.abelian) && throw(ArgumentError(
        "requested abelian symmetry cannot be audited from unlabelled symbolic operators",
    ))
    any(item -> item.lowering_status === :audit_only, audit.abelian) && throw(ArgumentError(
        "requested abelian symmetry is audit-only because the active carrier does not encode it",
    ))
    any(item -> item.status === :broken, audit.nonabelian) && throw(ArgumentError(
        "requested non-abelian candidate fails its explicit axial symmetry screen",
    ))
    any(item -> item.status === :unavailable, audit.nonabelian) && throw(ArgumentError(
        "requested non-abelian candidate lacks explicit axial generator metadata",
    ))
    any(item -> item.lowering_status === :unsupported, audit.nonabelian) && throw(ArgumentError(
        "non-abelian SU2Reduce is unsupported without a core SU2 lowering path",
    ))
    return audit
end
