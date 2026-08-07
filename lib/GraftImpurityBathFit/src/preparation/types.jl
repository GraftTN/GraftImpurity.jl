"""Backend-neutral immutable input to finite impurity-problem preparation."""
abstract type AbstractImpurityPreparationInput end

"""Typed outcome of preparation, distinct from every backend solve result."""
abstract type AbstractImpurityPreparationOutcome end

"""
    HybridizationPreparationInput(delta, partition; h_loc, block=nothing)

Owned direct-hybridization source and the complete local one-body Hamiltonian
used to construct a finite impurity problem. `source` and `hybridization` are
the same immutable `BathFitInput` snapshot; the repeated binding makes the
source/fit roles uniform with [`WeissPreparationInput`](@ref).
"""
struct HybridizationPreparationInput{
        S<:BathFitInput,H,P<:Partition} <: AbstractImpurityPreparationInput
    source::S
    hybridization::S
    h_loc::H
    partition::P

    function HybridizationPreparationInput(source::S, hybridization::S,
                                            h_loc::H, partition::P,
                                            ::Val{:validated}) where {
                                                S<:BathFitInput,H,P<:Partition}
        source === hybridization || throw(ArgumentError(
            "a direct hybridization preparation must fit its owned source snapshot",
        ))
        new{S,H,P}(source, hybridization, h_loc, partition)
    end
end

"""
    WeissPreparationInput(weiss, partition; h_loc, block=nothing)

Owned Weiss-field source plus the explicitly converted hybridization
`Delta(iw) = iw*I - h_loc - inv(G0(iw))`. The original and converted
`BathFitInput` snapshots remain distinct preparation provenance.
"""
struct WeissPreparationInput{
        S<:BathFitInput,F<:BathFitInput,H,P<:Partition} <:
        AbstractImpurityPreparationInput
    source::S
    hybridization::F
    h_loc::H
    partition::P

    function WeissPreparationInput(source::S, hybridization::F,
                                   h_loc::H, partition::P,
                                   ::Val{:validated}) where {
                                       S<:BathFitInput,F<:BathFitInput,H,P<:Partition}
        source === hybridization && throw(ArgumentError(
            "a Weiss preparation must retain distinct source and hybridization snapshots",
        ))
        new{S,F,H,P}(source, hybridization, h_loc, partition)
    end
end

"""
    ImpurityPreparationPolicy(kernel, criteria)

Immutable fit and audit policy. `criteria` must explicitly require a mountable
Hamiltonian bath; temperature and request-horizon criteria are supplied here
and are never inferred from a backend request or mutable solver state.
"""
struct ImpurityPreparationPolicy{K<:AbstractRealPoleBathFitKernel}
    kernel::K
    criteria::BathFitCriteria

    function ImpurityPreparationPolicy(kernel::K,
                                       criteria::BathFitCriteria) where {
                                           K<:AbstractRealPoleBathFitKernel}
        owned_kernel, owned_criteria = deepcopy((kernel, criteria))
        owned_criteria.require_mountable || throw(ArgumentError(
            "ImpurityPreparationPolicy criteria must set require_mountable=true",
        ))
        new{K}(owned_kernel, owned_criteria)
    end
end

"""Complete immutable provenance for one fit, realization, and audit attempt."""
struct ImpurityPreparationProvenance{
        I<:AbstractImpurityPreparationInput,X<:AbstractImpurityInteraction,
        S<:AbstractImpuritySymmetryDeclaration,
        P<:ImpurityPreparationPolicy,E<:PoleExpansion,
        D<:Union{DiscretizationResult,NonMountablePoleFit},
        A<:BathFitAudit,R<:NamedTuple}
    input::I
    interaction::X
    symmetry::S
    policy::P
    expansion::E
    realization::D
    audit::A
    realization_options::R

    function ImpurityPreparationProvenance(
            input::I, interaction::X, symmetry::S, policy::P,
            expansion::E, realization::D, audit::A,
            realization_options::R, ::Val{:validated}) where {
                I<:AbstractImpurityPreparationInput,
                X<:AbstractImpurityInteraction,
                S<:AbstractImpuritySymmetryDeclaration,
                P<:ImpurityPreparationPolicy,E<:PoleExpansion,
                D<:Union{DiscretizationResult,NonMountablePoleFit},
                A<:BathFitAudit,R<:NamedTuple}
        realization.expansion === expansion || throw(ArgumentError(
            "preparation realization must retain its exact pole expansion",
        ))
        realization.report.source === input.hybridization || throw(ArgumentError(
            "preparation realization report must retain its fitted hybridization input",
        ))
        new{I,X,S,P,E,D,A,R}(
            input, interaction, symmetry, policy, expansion, realization,
            audit, realization_options,
        )
    end
end

"""Successfully prepared canonical finite problem plus complete provenance."""
struct PreparedImpurityProblem{
        P<:ImpurityProblem,V<:ImpurityPreparationProvenance} <:
        AbstractImpurityPreparationOutcome
    problem::P
    provenance::V

    function PreparedImpurityProblem(problem::P, provenance::V,
                                     ::Val{:validated}) where {
                                         P<:ImpurityProblem,
                                         V<:ImpurityPreparationProvenance}
        provenance.realization isa DiscretizationResult || throw(ArgumentError(
            "PreparedImpurityProblem requires a mountable DiscretizationResult",
        ))
        new{P,V}(problem, provenance)
    end
end

"""
Typed preparation failure for a finite pole fit that cannot form a Hamiltonian
bath. It has no problem, backend request, workspace, or solve result.
"""
struct NonMountableImpurityPreparation{
        V<:ImpurityPreparationProvenance} <: AbstractImpurityPreparationOutcome
    provenance::V

    function NonMountableImpurityPreparation(provenance::V,
                                             ::Val{:validated}) where {
                                                 V<:ImpurityPreparationProvenance}
        provenance.realization isa NonMountablePoleFit || throw(ArgumentError(
            "NonMountableImpurityPreparation requires NonMountablePoleFit provenance",
        ))
        new{V}(provenance)
    end
end
