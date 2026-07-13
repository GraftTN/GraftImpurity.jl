function _require_interaction_operators(interaction::AbstractImpurityInteraction,
                                        operators::ImpurityOperators)
    interaction.layout == operators.layout || throw(ArgumentError(
        "interaction FlavorLayout must match ImpurityOperators.layout",
    ))
    return nothing
end

_density_factors(left::Symbol, right::Symbol) = _FermionFactor[
    _creator(left), _annihilator(left), _creator(right), _annihilator(right),
]

function _add_hermitian_monomial!(coefficients::Dict{Tuple,ComplexF64},
                                  layout::FlavorLayout, coefficient::Number,
                                  factors::Vector{_FermionFactor})
    _add_monomial!(coefficients, layout, coefficient, factors)
    _add_monomial!(coefficients, layout, conj(coefficient), _adjoint_factors(factors))
    return coefficients
end

function _density_density_coefficients(interaction::DensityDensityInteraction)
    coefficients = Dict{Tuple,ComplexF64}()
    ordered = flavors(interaction.layout)
    for index in eachindex(ordered)
        _add_monomial!(coefficients, interaction.layout, interaction.U[index, index],
                       _FermionFactor[_creator(ordered[index]),
                                       _annihilator(ordered[index])])
    end
    for left in 1:(length(ordered) - 1), right in (left + 1):length(ordered)
        _add_monomial!(coefficients, interaction.layout, interaction.U[left, right],
                       _density_factors(ordered[left], ordered[right]))
    end
    return coefficients
end

function _kanamori_coefficients(interaction::KanamoriInteraction)
    coefficients = Dict{Tuple,ComplexF64}()
    pairs = interaction.flavor_map.orbital_pairs
    for (up, down) in pairs
        _add_monomial!(coefficients, interaction.layout, interaction.U,
                       _density_factors(up, down))
    end
    for left in 1:(length(pairs) - 1), right in (left + 1):length(pairs)
        up_left, down_left = pairs[left]
        up_right, down_right = pairs[right]
        for (first, second) in ((up_left, up_right), (down_left, down_right))
            _add_monomial!(coefficients, interaction.layout,
                           interaction.Uprime - interaction.J,
                           _density_factors(first, second))
        end
        for (first, second) in ((up_left, down_right), (down_left, up_right))
            _add_monomial!(coefficients, interaction.layout, interaction.Uprime,
                           _density_factors(first, second))
        end
        if interaction.terms.spin_flip
            spin_flip = _FermionFactor[
                _creator(up_left), _annihilator(down_left),
                _creator(down_right), _annihilator(up_right),
            ]
            _add_hermitian_monomial!(coefficients, interaction.layout,
                                     -interaction.J, spin_flip)
        end
        if interaction.terms.pair_hopping
            pair_hopping = _FermionFactor[
                _creator(up_left), _creator(down_left),
                _annihilator(down_right), _annihilator(up_right),
            ]
            _add_hermitian_monomial!(coefficients, interaction.layout,
                                     interaction.J, pair_hopping)
        end
    end
    return coefficients
end

function _full_coulomb_coefficients(interaction::FullCoulombInteraction)
    coefficients = Dict{Tuple,ComplexF64}()
    prefactor = interaction.convention isa BareCoulombTensor ? 0.5 : 0.25
    ordered = flavors(interaction.layout)
    for a in eachindex(ordered), b in eachindex(ordered),
        c in eachindex(ordered), d in eachindex(ordered)
        coefficient = prefactor * interaction.U[a, b, c, d]
        iszero(coefficient) && continue
        _add_monomial!(coefficients, interaction.layout, coefficient,
                       _FermionFactor[
                           _creator(ordered[a]), _creator(ordered[b]),
                           _annihilator(ordered[d]), _annihilator(ordered[c]),
                       ])
    end
    return coefficients
end

_interaction_coefficients(interaction::DensityDensityInteraction) =
    _density_density_coefficients(interaction)
_interaction_coefficients(interaction::KanamoriInteraction) =
    _kanamori_coefficients(interaction)
_interaction_coefficients(interaction::FullCoulombInteraction) =
    _full_coulomb_coefficients(interaction)

"""
    lower_interaction(interaction, operators, sector_spec) -> OpSum

Lower one concrete interaction family into ordinary Graft symbolic terms. The
compiler fixes the global `FlavorLayout` order, aggregates anticommuting
equivalents exactly, and pre-multiplies same-site local factors before creating
each `SiteOp`; it never treats named hybridization partitions as interaction
constraints.
"""
function lower_interaction(interaction::AbstractImpurityInteraction,
                           operators::ImpurityOperators, sector_spec)
    _require_interaction_operators(interaction, operators)
    coefficients = _interaction_coefficients(interaction)
    _validate_hermitian_monomials(coefficients, interaction.layout,
                                  string(nameof(typeof(interaction))))
    return _compile_fermion_monomials(interaction.layout, operators, coefficients)
end
