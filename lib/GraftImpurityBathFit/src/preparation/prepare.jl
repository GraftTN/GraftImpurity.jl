function _validate_preparation_components(input::AbstractImpurityPreparationInput,
                                          interaction::AbstractImpurityInteraction,
                                          symmetry::AbstractImpuritySymmetryDeclaration)
    layout = input.hybridization.layout
    _preparation_h_loc_layout(input.h_loc) == layout || throw(ArgumentError(
        "preparation h_loc FlavorLayout must match its hybridization input",
    ))
    interaction_layout(interaction) == layout ||
        throw(ArgumentError(
            "preparation interaction FlavorLayout must match its hybridization input",
        ))
    symmetry_layout(symmetry) == layout ||
        throw(ArgumentError(
            "preparation symmetry FlavorLayout must match its hybridization input",
        ))
    validate_partition(input.partition, layout)
    return layout
end

"""
    prepare_impurity_problem(input, interaction, symmetry, policy;
                             orbital_order=nothing, atol=0,
                             rtol=sqrt(eps()), broadening=nothing)

Fit, realize, and audit an immutable GreenFunc preparation before any backend
dispatch. A mountable realization constructs an `ImpurityProblem`; a
non-mountable fit returns `NonMountableImpurityPreparation` with the same
available provenance and no partial problem or projected fallback.
"""
function prepare_impurity_problem(
        input::AbstractImpurityPreparationInput,
        interaction::AbstractImpurityInteraction,
        symmetry::AbstractImpuritySymmetryDeclaration,
        policy::ImpurityPreparationPolicy;
        orbital_order=nothing,
        atol::Real=0.0,
        rtol::Real=sqrt(eps(Float64)),
        broadening=nothing)
    owned_input, owned_interaction, owned_symmetry, owned_policy,
        owned_orbital_order = deepcopy((
            input, interaction, symmetry, policy, orbital_order,
        ))
    _validate_preparation_components(
        owned_input, owned_interaction, owned_symmetry,
    )
    expansion = real_pole_bath_fit(
        owned_input.hybridization, owned_policy.kernel, owned_input.partition,
    )
    realization = realize_bath(
        owned_input.hybridization, expansion, owned_input.partition;
        orbital_order=owned_orbital_order, atol, rtol, broadening,
    )
    audit = audit_bathfit(realization.report, owned_policy.criteria)
    options = (
        orbital_order=owned_orbital_order,
        atol=Float64(atol),
        rtol=Float64(rtol),
        broadening=broadening === nothing ? nothing : Float64(broadening),
    )
    provenance = ImpurityPreparationProvenance(
        owned_input, owned_interaction, owned_symmetry, owned_policy,
        expansion, realization, audit, options, Val(:validated),
    )
    if realization isa NonMountablePoleFit
        return NonMountableImpurityPreparation(provenance, Val(:validated))
    end

    problem = ImpurityProblem(
        realization.bath, owned_input.h_loc, owned_interaction, owned_symmetry,
    )
    return PreparedImpurityProblem(problem, provenance, Val(:validated))
end
