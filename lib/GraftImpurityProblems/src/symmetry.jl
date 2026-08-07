"""Marker for an explicit immutable description of a physical action."""
abstract type AbstractPhysicalActionSemantics end

"""Fermion-number generator, with one integer weight per ordered flavor."""
struct ChargeU1ActionSemantics{W<:Tuple} <: AbstractPhysicalActionSemantics
    generator::W
end

"""Explicit diagonal `U(1)` generator in the ordered impurity flavor basis."""
struct FlavorU1ActionSemantics{W<:Tuple} <: AbstractPhysicalActionSemantics
    generator::W
end

"""Spin-rotation action descriptor and its optional explicit axial screen."""
struct SU2ActionSemantics{A<:Union{Nothing,Tuple}} <:
        AbstractPhysicalActionSemantics
    axial_descriptor::A
end

Base.:(==)(left::ChargeU1ActionSemantics,
           right::ChargeU1ActionSemantics) =
    left.generator == right.generator
Base.hash(semantics::ChargeU1ActionSemantics, seed::UInt) =
    hash(semantics.generator, hash(:ChargeU1ActionSemantics, seed))

Base.:(==)(left::FlavorU1ActionSemantics,
           right::FlavorU1ActionSemantics) =
    left.generator == right.generator
Base.hash(semantics::FlavorU1ActionSemantics, seed::UInt) =
    hash(semantics.generator, hash(:FlavorU1ActionSemantics, seed))

Base.:(==)(left::SU2ActionSemantics, right::SU2ActionSemantics) =
    left.axial_descriptor == right.axial_descriptor
Base.hash(semantics::SU2ActionSemantics, seed::UInt) =
    hash(semantics.axial_descriptor, hash(:SU2ActionSemantics, seed))

function _has_mutable_semantic_payload(value)
    value isa Union{Nothing,Missing,Symbol,Number,Char,AbstractString,Type} &&
        return false
    Base.ismutabletype(typeof(value)) && return true
    return any(
        index -> _has_mutable_semantic_payload(getfield(value, index)),
        1:fieldcount(typeof(value)),
    )
end

"""
    SymmetryActionIdentity(action, basis, category_product, semantics)

Structural identity of one declared physical action. `basis` is the complete
`FlavorLayout` identity, `category_product` is an explicit ordered tuple of
category identities, and `semantics` is a mandatory immutable physical
generator/action descriptor. Consequently, two actions with the same display
name, basis, and categories but distinct generators or parameters are not the
same action. The value contains no mounted bath ownership, local carrier,
fusion route, or backend capability claim.
"""
struct SymmetryActionIdentity{L<:FlavorLayout,C<:Tuple,
                              P<:AbstractPhysicalActionSemantics}
    action::Symbol
    basis::L
    category_product::C
    semantics::P

    function SymmetryActionIdentity(action::Symbol, basis::L,
                                    category_product::C, semantics::P,
                                    ::Val{:validated}) where {
                                        L<:FlavorLayout,C<:Tuple,
                                        P<:AbstractPhysicalActionSemantics}
        new{L,C,P}(action, basis, category_product, semantics)
    end
end

function SymmetryActionIdentity(action::Symbol, basis::FlavorLayout,
                                categories::Tuple,
                                semantics::AbstractPhysicalActionSemantics)
    isempty(String(action)) && throw(ArgumentError(
        "symmetry action identity must have a nonempty action name",
    ))
    isempty(categories) && throw(ArgumentError(
        "symmetry action identity must have a nonempty category product",
    ))
    _has_mutable_semantic_payload(semantics) && throw(ArgumentError(
        "symmetry action semantics must contain only immutable physical data",
    ))
    return SymmetryActionIdentity(
        action, basis, categories, semantics, Val(:validated),
    )
end

Base.:(==)(left::SymmetryActionIdentity, right::SymmetryActionIdentity) =
    left.action == right.action &&
    left.basis == right.basis &&
    left.category_product == right.category_product &&
    left.semantics == right.semantics

Base.hash(identity::SymmetryActionIdentity, seed::UInt) = hash(
    (identity.action, identity.basis, identity.category_product,
     identity.semantics),
    hash(:SymmetryActionIdentity, seed),
)

"""Complete impurity basis on which the symmetry action is declared."""
action_layout(identity::SymmetryActionIdentity) = identity.basis

"""Explicit ordered category product of a symmetry action."""
category_product(identity::SymmetryActionIdentity) = identity.category_product

"""Explicit immutable physical action descriptor."""
action_semantics(identity::SymmetryActionIdentity) = identity.semantics

"""
Structural particle-number identity for the existing topology-free `ChargeU1`
declaration. Fermion parity is intentionally absent: it is a local TTNS carrier
policy, not part of the physical charge action.
"""
action_identity(action::ChargeU1) = SymmetryActionIdentity(
    :charge, action.layout, (U1Irrep,),
    ChargeU1ActionSemantics(ntuple(_ -> 1, length(flavors(action.layout)))),
)

"""Structural identity of an explicit diagonal flavor `U(1)` generator."""
action_identity(action::FlavorU1) = SymmetryActionIdentity(
    action.name, action.layout, (U1Irrep,),
    FlavorU1ActionSemantics(action.weights),
)

"""Structural identity of an `SU(2)` candidate and its explicit axial screen."""
function action_identity(action::SU2Reduce)
    axial = action.axial_generator
    descriptor = axial === nothing ? nothing : (axial.name, axial.weights)
    return SymmetryActionIdentity(
        action.name, action.layout, (SU2Irrep,), SU2ActionSemantics(descriptor),
    )
end

"""Backend-neutral marker for impurity symmetry declaration containers."""
abstract type AbstractImpuritySymmetryDeclaration end

"""
    ImpuritySymmetryDeclaration(actions)

Topology-free physical actions declared on one common impurity basis. External
action types participate by implementing `action_identity(action)` and remain
opaque to this container.
"""
struct ImpuritySymmetryDeclaration{A<:Tuple} <:
        AbstractImpuritySymmetryDeclaration
    actions::A

    function ImpuritySymmetryDeclaration(actions::A,
                                         ::Val{:validated}) where {A<:Tuple}
        new{A}(actions)
    end
end

function ImpuritySymmetryDeclaration(actions::Tuple)
    isempty(actions) && throw(ArgumentError(
        "ImpuritySymmetryDeclaration needs at least one physical action",
    ))
    identities = Tuple(action_identity(action) for action in actions)
    all(identity -> identity isa SymmetryActionIdentity, identities) ||
        throw(ArgumentError(
            "every impurity symmetry action must supply a SymmetryActionIdentity",
        ))
    basis = first(identities).basis
    all(identity -> identity.basis == basis, identities) || throw(ArgumentError(
        "every impurity symmetry action must use the same FlavorLayout basis",
    ))
    allunique(identities) || throw(ArgumentError(
        "impurity symmetry action identities must be unique",
    ))
    return ImpuritySymmetryDeclaration(actions, Val(:validated))
end

ImpuritySymmetryDeclaration(action::ChargeU1) =
    ImpuritySymmetryDeclaration((action,))

ImpuritySymmetryDeclaration(layout::FlavorLayout) =
    ImpuritySymmetryDeclaration(ChargeU1(layout))

"""Declared physical action values in deterministic order."""
symmetry_actions(declaration::ImpuritySymmetryDeclaration) = declaration.actions

"""Structural action identities in the same order as `symmetry_actions`."""
symmetry_action_identities(declaration::ImpuritySymmetryDeclaration) =
    Tuple(action_identity(action) for action in declaration.actions)

"""Complete impurity basis shared by all declared actions."""
symmetry_layout(declaration::ImpuritySymmetryDeclaration) =
    first(symmetry_action_identities(declaration)).basis

Base.:(==)(left::ImpuritySymmetryDeclaration,
           right::ImpuritySymmetryDeclaration) =
    symmetry_action_identities(left) == symmetry_action_identities(right)

Base.hash(declaration::ImpuritySymmetryDeclaration, seed::UInt) = hash(
    symmetry_action_identities(declaration),
    hash(:ImpuritySymmetryDeclaration, seed),
)
