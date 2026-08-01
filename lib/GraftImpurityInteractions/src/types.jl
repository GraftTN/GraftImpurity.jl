"""Explicit convention marker for `1/2 U_abcd d_a' d_b' d_d d_c`."""
struct BareCoulombTensor end

"""Explicit convention marker for `1/4 Gamma_abcd d_a' d_b' d_d d_c`."""
struct AntisymmetrizedVertex end

const AbstractCoulombConvention = Union{BareCoulombTensor,AntisymmetrizedVertex}

_finite_interaction_number(value::Number) =
    isfinite(real(value)) && isfinite(imag(value))

function _interaction_scale(values)
    scale = 0.0
    for value in values
        scale = max(scale, abs(value))
    end
    return scale
end

_interaction_tolerance(values) = 128 * eps(Float64) *
    max(1.0, _interaction_scale(values))

function _validate_interaction_matrix(matrix::AbstractMatrix{T},
                                      layout::FlavorLayout,
                                      name::AbstractString) where {T<:Number}
    nflavor = length(flavors(layout))
    size(matrix) == (nflavor, nflavor) || throw(DimensionMismatch(
        "$name must have one row and column per layout flavor",
    ))
    all(_finite_interaction_number, matrix) || throw(ArgumentError(
        "$name entries must be finite",
    ))
    return Matrix{T}(matrix)
end

function _require_hermitian_matrix(matrix::AbstractMatrix{<:Number},
                                   name::AbstractString)
    tolerance = _interaction_tolerance(matrix)
    isapprox(matrix, matrix'; atol=tolerance, rtol=0) || throw(ArgumentError(
        "$name must be Hermitian in its declared FlavorLayout basis",
    ))
    return matrix
end

function _require_real_matrix(matrix::AbstractMatrix{<:Number},
                              name::AbstractString)
    tolerance = _interaction_tolerance(matrix)
    all(value -> abs(imag(value)) <= tolerance, matrix) || throw(ArgumentError(
        "$name must be real because density products are Hermitian",
    ))
    return matrix
end

"""
    DensityDensityInteraction(U, layout)

Layout-owned density interaction with the exact convention
`sum_(a<b) U[a,b] n_a n_b + sum_a U[a,a] n_a`. The diagonal uses the fermion
identity `n_a^2 = n_a` and remains explicit rather than being silently moved
to a separate one-body value. `U` must be finite, real, symmetric, and have
one row and column per flavor. It is deliberately a separate interaction
family, not a projected `FullCoulombInteraction`.
"""
struct DensityDensityInteraction{T<:Number,M<:AbstractMatrix{T}} <:
        AbstractImpurityInteraction
    U::M
    layout::FlavorLayout

    function DensityDensityInteraction(U::M, layout::FlavorLayout,
                                       ::Val{:validated}) where {
                                           T<:Number,M<:AbstractMatrix{T}}
        new{T,M}(U, layout)
    end
end

function DensityDensityInteraction(U::AbstractMatrix{T}, layout::FlavorLayout) where {T<:Number}
    matrix = _validate_interaction_matrix(U, layout, "DensityDensityInteraction.U")
    _require_hermitian_matrix(matrix, "DensityDensityInteraction.U")
    _require_real_matrix(matrix, "DensityDensityInteraction.U")
    return DensityDensityInteraction(matrix, layout, Val(:validated))
end

Base.:(==)(left::DensityDensityInteraction, right::DensityDensityInteraction) =
    left.U == right.U && left.layout == right.layout

function _content_hash(values, seed::UInt)
    state = hash(size(values), seed)
    for value in values
        state = hash(value, state)
    end
    return state
end

Base.hash(interaction::DensityDensityInteraction, seed::UInt) = hash(
    interaction.layout,
    _content_hash(interaction.U, hash(:DensityDensityInteraction, seed)),
)

"""Selection of the two independent Kanamori exchange families."""
struct KanamoriTerms
    spin_flip::Bool
    pair_hopping::Bool
end

Base.:(==)(left::KanamoriTerms, right::KanamoriTerms) =
    left.spin_flip == right.spin_flip && left.pair_hopping == right.pair_hopping
Base.hash(terms::KanamoriTerms, seed::UInt) =
    hash((terms.spin_flip, terms.pair_hopping), hash(:KanamoriTerms, seed))

"""
    KanamoriFlavorMap(layout, orbital_pairs)

Immutable ordered assignment of every layout flavor to an explicit `(up, down)`
pair for one Kanamori orbital. No spin identity is inferred from labels, site
names, or global flavor adjacency.
"""
struct KanamoriFlavorMap
    layout::FlavorLayout
    orbital_pairs::Tuple{Vararg{NTuple{2,Symbol}}}

    function KanamoriFlavorMap(layout::FlavorLayout,
                               orbital_pairs::Tuple{Vararg{NTuple{2,Symbol}}},
                               ::Val{:validated})
        new(layout, orbital_pairs)
    end
end

function KanamoriFlavorMap(layout::FlavorLayout, orbital_pairs)
    pairs = Tuple(begin
        pair isa Tuple && length(pair) == 2 || throw(ArgumentError(
            "each Kanamori orbital must be an explicit `(up, down)` flavor pair",
        ))
        up, down = Symbol(pair[1]), Symbol(pair[2])
        up != down || throw(ArgumentError(
            "a Kanamori orbital may not repeat one flavor",
        ))
        (up, down)
    end for pair in orbital_pairs)
    isempty(pairs) && throw(ArgumentError(
        "KanamoriFlavorMap needs at least one orbital pair",
    ))
    declared = Symbol[flavor for pair in pairs for flavor in pair]
    allunique(declared) || throw(ArgumentError(
        "KanamoriFlavorMap orbital pairs must not overlap",
    ))
    Set(declared) == Set(flavors(layout)) || throw(ArgumentError(
        "KanamoriFlavorMap must cover every FlavorLayout flavor exactly once",
    ))
    return KanamoriFlavorMap(layout, pairs, Val(:validated))
end

Base.:(==)(left::KanamoriFlavorMap, right::KanamoriFlavorMap) =
    left.layout == right.layout && left.orbital_pairs == right.orbital_pairs
Base.hash(mapping::KanamoriFlavorMap, seed::UInt) = hash(
    (mapping.layout, mapping.orbital_pairs), hash(:KanamoriFlavorMap, seed),
)

"""
    KanamoriInteraction(U, Uprime, J, layout; flavor_map, spin_flip, pair_hopping)

Explicit Hubbard-Kanamori interaction. `flavor_map` fixes physical spin pairs
independently of flavor spelling. The two exchange families are selected
independently and their complete Hermitian pairs are emitted only when enabled.
"""
struct KanamoriInteraction{T<:Number,M<:KanamoriFlavorMap} <:
        AbstractImpurityInteraction
    U::T
    Uprime::T
    J::T
    terms::KanamoriTerms
    flavor_map::M
    layout::FlavorLayout
end

function _validate_kanamori_parameter(value::Number, name::AbstractString)
    _finite_interaction_number(value) || throw(ArgumentError(
        "$name must be finite",
    ))
    iszero(imag(value)) || throw(ArgumentError(
        "$name must be real to define a Hermitian Kanamori interaction",
    ))
    real(value) >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return value
end

function KanamoriInteraction(U::Number, Uprime::Number, J::Number,
                             layout::FlavorLayout;
                             flavor_map::KanamoriFlavorMap,
                             spin_flip::Bool, pair_hopping::Bool)
    flavor_map.layout == layout || throw(ArgumentError(
        "KanamoriFlavorMap FlavorLayout must match KanamoriInteraction.layout",
    ))
    values = promote(U, Uprime, J)
    _validate_kanamori_parameter(values[1], "KanamoriInteraction.U")
    _validate_kanamori_parameter(values[2], "KanamoriInteraction.Uprime")
    _validate_kanamori_parameter(values[3], "KanamoriInteraction.J")
    T = typeof(values[1])
    return KanamoriInteraction{T,typeof(flavor_map)}(
        values[1], values[2], values[3],
        KanamoriTerms(spin_flip, pair_hopping), flavor_map, layout,
    )
end

Base.:(==)(left::KanamoriInteraction, right::KanamoriInteraction) =
    left.U == right.U && left.Uprime == right.Uprime && left.J == right.J &&
    left.terms == right.terms && left.flavor_map == right.flavor_map &&
    left.layout == right.layout
Base.hash(interaction::KanamoriInteraction, seed::UInt) = hash(
    (interaction.U, interaction.Uprime, interaction.J, interaction.terms,
     interaction.flavor_map, interaction.layout),
    hash(:KanamoriInteraction, seed),
)

function _validate_coulomb_tensor(U::AbstractArray{T,4}, layout::FlavorLayout,
                                  convention::AbstractCoulombConvention) where {T<:Number}
    nflavor = length(flavors(layout))
    size(U) == (nflavor, nflavor, nflavor, nflavor) || throw(DimensionMismatch(
        "FullCoulombInteraction.U must have four FlavorLayout-sized axes",
    ))
    all(_finite_interaction_number, U) || throw(ArgumentError(
        "FullCoulombInteraction.U entries must be finite",
    ))
    tensor = Array{T,4}(U)
    tolerance = _interaction_tolerance(tensor)
    for a in axes(tensor, 1), b in axes(tensor, 2),
        c in axes(tensor, 3), d in axes(tensor, 4)
        isapprox(tensor[a, b, c, d], conj(tensor[c, d, a, b]);
                 atol=tolerance, rtol=0) || throw(ArgumentError(
            "FullCoulombInteraction.U violates its Hermiticity convention at ($a, $b, $c, $d)",
        ))
        if convention isa AntisymmetrizedVertex
            isapprox(tensor[a, b, c, d], -tensor[b, a, c, d];
                     atol=tolerance, rtol=0) || throw(ArgumentError(
                "AntisymmetrizedVertex must be antisymmetric in its creation indices",
            ))
            isapprox(tensor[a, b, c, d], -tensor[a, b, d, c];
                     atol=tolerance, rtol=0) || throw(ArgumentError(
                "AntisymmetrizedVertex must be antisymmetric in its annihilation indices",
            ))
        end
    end
    return tensor
end

"""
    FullCoulombInteraction(U, convention, layout)

Production four-index interaction with an explicit tensor convention. A bare
tensor is retained exactly and merged algebraically only during lowering; an
antisymmetrized vertex is additionally validated in both fermionic pairs.
Both conventions require Hermiticity and support complex entries.
"""
struct FullCoulombInteraction{T<:Number,A<:AbstractArray{T,4},C<:AbstractCoulombConvention} <:
        AbstractImpurityInteraction
    U::A
    convention::C
    layout::FlavorLayout

    function FullCoulombInteraction(U::A, convention::C, layout::FlavorLayout,
                                    ::Val{:validated}) where {
                                        T<:Number,A<:AbstractArray{T,4},
                                        C<:AbstractCoulombConvention}
        new{T,A,C}(U, convention, layout)
    end
end

function FullCoulombInteraction(U::AbstractArray{T,4},
                                convention::C,
                                layout::FlavorLayout) where {
                                    T<:Number,C<:AbstractCoulombConvention}
    tensor = _validate_coulomb_tensor(U, layout, convention)
    return FullCoulombInteraction(tensor, convention, layout, Val(:validated))
end

Base.:(==)(left::FullCoulombInteraction, right::FullCoulombInteraction) =
    left.U == right.U && left.convention == right.convention &&
    left.layout == right.layout
Base.hash(interaction::FullCoulombInteraction, seed::UInt) = hash(
    (interaction.convention, interaction.layout),
    _content_hash(interaction.U, hash(:FullCoulombInteraction, seed)),
)

"""
    ImpurityOneBody(matrix, layout; label=:h_loc)

Hermitian one-particle term in one explicit FlavorLayout. The same value type
owns local crystal-field and SOC matrices; `label` records semantic provenance
without weakening the basis identity.
"""
struct ImpurityOneBody{T<:Number,M<:AbstractMatrix{T}}
    matrix::M
    layout::FlavorLayout
    label::Symbol
end

function ImpurityOneBody(matrix::AbstractMatrix{T}, layout::FlavorLayout;
                         label::Symbol=:h_loc) where {T<:Number}
    isempty(String(label)) && throw(ArgumentError(
        "ImpurityOneBody label must be nonempty",
    ))
    canonical = _validate_interaction_matrix(matrix, layout, "ImpurityOneBody.matrix")
    _require_hermitian_matrix(canonical, "ImpurityOneBody.matrix")
    return ImpurityOneBody{T,typeof(canonical)}(canonical, layout, label)
end

Base.:(==)(left::ImpurityOneBody, right::ImpurityOneBody) =
    left.matrix == right.matrix && left.layout == right.layout && left.label == right.label
Base.hash(onebody::ImpurityOneBody, seed::UInt) = hash(
    (onebody.layout, onebody.label),
    _content_hash(onebody.matrix, hash(:ImpurityOneBody, seed)),
)

"""
    DensityDensityDecomposition

Exact separation of a density-density value into its explicit diagonal
one-body contribution and a strictly off-diagonal density interaction. This
is required before a general basis rotation: `n_a^2 = n_a` cannot be encoded
by a four-index fermionic interaction tensor after the basis changes.
"""
struct DensityDensityDecomposition{H<:ImpurityOneBody,I<:DensityDensityInteraction}
    one_body::H
    interaction::I
end

"""
    split_density_density(interaction) -> DensityDensityDecomposition

Preserve a valid diagonal `U[a,a] n_a` term explicitly as `ImpurityOneBody`
and return a layout-owned density interaction with an exactly zero diagonal.
No term is dropped or silently reclassified during this semantic split.
"""
function split_density_density(interaction::DensityDensityInteraction)
    matrix = Matrix{ComplexF64}(interaction.U)
    diagonal = diag(matrix)
    for index in axes(matrix, 1)
        matrix[index, index] = 0.0
    end
    return DensityDensityDecomposition(
        ImpurityOneBody(Diagonal(diagonal), interaction.layout;
                        label=:density_diagonal),
        DensityDensityInteraction(matrix, interaction.layout),
    )
end
