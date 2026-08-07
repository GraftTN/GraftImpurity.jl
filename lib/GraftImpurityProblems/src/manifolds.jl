"""Backend-neutral marker for one or more requested many-body subspaces."""
abstract type AbstractImpurityManifold end

"""One opaque outer irrep bound to one declared physical action."""
struct TargetIrrep{K<:SymmetryActionIdentity,Q} <: AbstractImpurityManifold
    action_identity::K
    target::Q
end

TargetIrrep(action, target) = TargetIrrep(action_identity(action), target)

"""A nonempty ordered scan of opaque outer irreps for one physical action."""
struct IrrepScan{K<:SymmetryActionIdentity,Q<:Tuple} <:
        AbstractImpurityManifold
    action_identity::K
    targets::Q

    function IrrepScan(action_identity::K, targets::Q,
                       ::Val{:validated}) where {
                           K<:SymmetryActionIdentity,Q<:Tuple}
        new{K,Q}(action_identity, targets)
    end
end

function IrrepScan(identity::SymmetryActionIdentity,
                   targets::Union{Tuple,AbstractVector})
    canonical = Tuple(targets)
    isempty(canonical) && throw(ArgumentError(
        "IrrepScan needs at least one target irrep",
    ))
    allunique(canonical) || throw(ArgumentError(
        "IrrepScan targets must be unique",
    ))
    return IrrepScan(identity, canonical, Val(:validated))
end

IrrepScan(action, targets::Union{Tuple,AbstractVector}) =
    IrrepScan(action_identity(action), targets)

"""Structural physical-action identity bound to a target manifold."""
manifold_action_identity(manifold::AbstractImpurityManifold) =
    manifold.action_identity

"""Requested target irreps in deterministic order."""
manifold_targets(manifold::TargetIrrep) = (manifold.target,)
manifold_targets(manifold::IrrepScan) = manifold.targets

Base.:(==)(left::TargetIrrep, right::TargetIrrep) =
    left.action_identity == right.action_identity && left.target == right.target
Base.:(==)(left::IrrepScan, right::IrrepScan) =
    left.action_identity == right.action_identity && left.targets == right.targets

Base.hash(manifold::TargetIrrep, seed::UInt) = hash(
    (manifold.action_identity, manifold.target), hash(:TargetIrrep, seed),
)
Base.hash(manifold::IrrepScan, seed::UInt) = hash(
    (manifold.action_identity, manifold.targets), hash(:IrrepScan, seed),
)

"""Deterministic bounded identity fingerprint for a target manifold."""
manifold_identity(manifold::AbstractImpurityManifold) =
    hash(manifold, zero(UInt))

"""
Typed failure from the backend-neutral response-reachability extension seam.
The common problem layer never infers category arithmetic from target values.
"""
struct ResponseReachabilityError <: Exception
    backend::Symbol
    reason::Symbol
    message::String
end

function Base.showerror(io::IO, error::ResponseReachabilityError)
    print(io, error.message, " (backend=", error.backend,
          ", reason=", error.reason, ")")
end

"""
    validate_response_target(source, response) -> response

Require an explicit response target to be bound to the exact same physical
action as its source. The outer irrep values remain opaque.
"""
function validate_response_target(source::TargetIrrep,
                                  response::TargetIrrep)
    source.action_identity == response.action_identity || throw(
        ResponseReachabilityError(
            :common, :action_identity_mismatch,
            "response target must use the source target's exact physical action identity",
        ),
    )
    return response
end

"""
    validate_response_reachability(backend, source, response, right, left)

Backend extension hook for validating the explicit intermediate target of an
ordered correlator. The default is deliberately fail-closed and makes no
integer, fusion, or category-specific assumption. Backends add methods for a
backend marker and operator representation they can certify.
"""
function validate_response_reachability(backend, source::TargetIrrep,
                                        response::TargetIrrep, right, left)
    validate_response_target(source, response)
    name = backend isa Symbol ? backend : Symbol(nameof(typeof(backend)))
    throw(ResponseReachabilityError(
        name, :unsupported,
        "no certified response-reachability rule exists for this backend and operator representation",
    ))
end
