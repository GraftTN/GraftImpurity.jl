function _onebody_coefficients(onebody::ImpurityOneBody)
    coefficients = Dict{Tuple,ComplexF64}()
    ordered = flavors(onebody.layout)
    for row in eachindex(ordered), column in eachindex(ordered)
        coefficient = onebody.matrix[row, column]
        iszero(coefficient) && continue
        _add_monomial!(coefficients, onebody.layout, coefficient,
                       _FermionFactor[
                           _creator(ordered[row]), _annihilator(ordered[column]),
                       ])
    end
    return coefficients
end

"""
    lower_one_body(onebody, operators, sector_spec) -> OpSum

Lower a layout-owned Hermitian one-particle matrix, including complex SOC, to
canonical local fermion monomials. Off-diagonal terms never infer a bath arm or
partition change.
"""
function lower_one_body(onebody::ImpurityOneBody,
                        operators::ImpurityOperators, sector_spec)
    onebody.layout == operators.layout || throw(ArgumentError(
        "ImpurityOneBody FlavorLayout must match ImpurityOperators.layout",
    ))
    coefficients = _onebody_coefficients(onebody)
    _validate_hermitian_monomials(coefficients, onebody.layout,
                                  "ImpurityOneBody($(onebody.label))")
    return _compile_fermion_monomials(onebody.layout, operators, coefficients)
end

function one_body_opsum(h_loc::Union{Nothing,ImpurityOneBody},
                        soc::Union{Nothing,ImpurityOneBody},
                        operators::ImpurityOperators, sector_spec)
    H = OpSum()
    h_loc === nothing || (H = H + lower_one_body(h_loc, operators, sector_spec))
    soc === nothing || (H = H + lower_one_body(soc, operators, sector_spec))
    return H
end

function _validated_basis_rotation(rotation::AbstractMatrix{<:Number},
                                  old_layout::FlavorLayout,
                                  new_layout::FlavorLayout)
    dimension = length(flavors(old_layout))
    length(flavors(new_layout)) == dimension || throw(DimensionMismatch(
        "basis rotation layouts must have the same flavor count",
    ))
    size(rotation) == (dimension, dimension) || throw(DimensionMismatch(
        "basis rotation must be square in the old/new FlavorLayout dimension",
    ))
    all(_finite_interaction_number, rotation) || throw(ArgumentError(
        "basis rotation entries must be finite",
    ))
    canonical = Matrix{ComplexF64}(rotation)
    tolerance = _interaction_tolerance(canonical)
    isapprox(canonical' * canonical, Matrix{ComplexF64}(I, dimension, dimension);
             atol=tolerance, rtol=0) || throw(ArgumentError(
        "basis rotation must be unitary",
    ))
    return canonical
end

"""
    rotate_one_body(onebody, rotation, new_layout) -> ImpurityOneBody

Apply `h_new = rotation' * h_old * rotation` while replacing the immutable
basis token. Interaction rotation is deliberately a separate operation so a
simplified interaction is never silently relabelled in the new basis.
"""
function rotate_one_body(onebody::ImpurityOneBody,
                         rotation::AbstractMatrix{<:Number},
                         new_layout::FlavorLayout)
    matrix = _validated_basis_rotation(rotation, onebody.layout, new_layout)
    return ImpurityOneBody(matrix' * onebody.matrix * matrix, new_layout;
                           label=onebody.label)
end
