function _quartic_key_indices(key::Tuple, layout::FlavorLayout)
    length(key) == 4 || throw(ArgumentError(
        "interaction rotation requires quartic fermion monomials",
    ))
    key[1][2] === :Cd && key[2][2] === :Cd &&
        key[3][2] === :C && key[4][2] === :C || throw(ArgumentError(
        "interaction rotation requires normal-ordered quartic monomials",
    ))
    a = flavor_index(layout, key[1][1])
    b = flavor_index(layout, key[2][1])
    d = flavor_index(layout, key[3][1])
    c = flavor_index(layout, key[4][1])
    return a, b, c, d
end

function _antisymmetrized_vertex(interaction::Union{DensityDensityInteraction,
                                                     KanamoriInteraction})
    layout = interaction.layout
    dimension = length(flavors(layout))
    vertex = zeros(ComplexF64, dimension, dimension, dimension, dimension)
    for (key, coefficient) in _interaction_coefficients(interaction)
        a, b, c, d = _quartic_key_indices(key, layout)
        vertex[a, b, c, d] += coefficient
        vertex[b, a, c, d] -= coefficient
        vertex[a, b, d, c] -= coefficient
        vertex[b, a, d, c] += coefficient
    end
    return FullCoulombInteraction(vertex, AntisymmetrizedVertex(), layout)
end

function _rotate_coulomb_tensor(tensor::AbstractArray{<:Number,4},
                                rotation::AbstractMatrix{<:Number})
    dimension = size(tensor, 1)
    stage_i = zeros(ComplexF64, dimension, dimension, dimension, dimension)
    stage_j = similar(stage_i)
    stage_k = similar(stage_i)
    result = similar(stage_i)
    for a in 1:dimension, j in 1:dimension, k in 1:dimension, l in 1:dimension
        for i in 1:dimension
            stage_i[a, j, k, l] += conj(rotation[i, a]) * tensor[i, j, k, l]
        end
    end
    for a in 1:dimension, b in 1:dimension, k in 1:dimension, l in 1:dimension
        for j in 1:dimension
            stage_j[a, b, k, l] += conj(rotation[j, b]) * stage_i[a, j, k, l]
        end
    end
    for a in 1:dimension, b in 1:dimension, c in 1:dimension, l in 1:dimension
        for k in 1:dimension
            stage_k[a, b, c, l] += rotation[k, c] * stage_j[a, b, k, l]
        end
    end
    for a in 1:dimension, b in 1:dimension, c in 1:dimension, d in 1:dimension
        for l in 1:dimension
            result[a, b, c, d] += rotation[l, d] * stage_k[a, b, c, l]
        end
    end
    return result
end

"""
    rotate_interaction(interaction, rotation, new_layout) -> FullCoulombInteraction

Rotate an interaction with `d_old = rotation * d_new`. Full tensors retain
their declared convention. Density-density and Kanamori values are first
lowered exactly to an antisymmetrized vertex, so a general rotation never
pretends that their simplified parameterization survived.
"""
function rotate_interaction(interaction::FullCoulombInteraction,
                            rotation::AbstractMatrix{<:Number},
                            new_layout::FlavorLayout)
    matrix = _validated_basis_rotation(rotation, interaction.layout, new_layout)
    tensor = _rotate_coulomb_tensor(interaction.U, matrix)
    return FullCoulombInteraction(tensor, interaction.convention, new_layout)
end

function rotate_interaction(interaction::Union{DensityDensityInteraction,
                                                KanamoriInteraction},
                            rotation::AbstractMatrix{<:Number},
                            new_layout::FlavorLayout)
    if interaction isa DensityDensityInteraction &&
       any(value -> !iszero(value), diag(interaction.U))
        throw(ArgumentError(
            "rotate_interaction cannot represent DensityDensityInteraction diagonal terms as a four-index tensor; call split_density_density and rotate its one_body and interaction fields explicitly",
        ))
    end
    return rotate_interaction(_antisymmetrized_vertex(interaction), rotation, new_layout)
end
