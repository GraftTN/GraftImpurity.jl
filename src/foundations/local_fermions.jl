"""
    FermionSiteOperators(modes)

Immutable local fZ2 Fock-space operator carrier for one physical impurity site.
`modes` fixes the canonical local fermion order.  The resulting `P`, `C`, `Cd`,
`N`, and `I` tensors use Graft's graded Backend vocabulary and therefore retain
fermionic exchange through TensorKit rather than introducing Jordan-Wigner
strings.  The occupancy convention is the declared `modes` order, with
`c_j |n> = (-1)^(sum_{i<j} n_i) n_j |n-e_j>`.
"""
struct FermionSiteOperators{P<:ElementarySpace,C<:AbstractTensorMap,
                            N<:AbstractTensorMap}
    modes::Tuple{Vararg{Symbol}}
    P::P
    C::Tuple{Vararg{C}}
    Cd::Tuple{Vararg{C}}
    N::Tuple{Vararg{N}}
    I::N
end

function _canonical_fock_states(mode_count::Int)
    mode_count > 0 || throw(ArgumentError("a fermion site needs at least one mode"))
    mode_count < 8 * sizeof(Int) ||
        throw(ArgumentError("local fermion mode count exceeds the addressable Fock basis"))
    states = collect(0:((1 << mode_count) - 1))
    sort!(states; by=state -> (isodd(count_ones(state)), state))
    return states
end

function _charged_local_tensor(matrix::Matrix{ComplexF64}, P::ElementarySpace,
                               Q::ElementarySpace)
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension || throw(DimensionMismatch(
        "local charged operator matrix must be square",
    ))
    return TensorMap(reshape(matrix, dimension, dimension, 1), P ← P ⊗ Q)
end

"""
    FermionSiteOperators(modes::AbstractVector{<:Symbol})

Construct local creation, annihilation, number, and identity operators in the
declared mode order.  The physical fZ2 space has equal even/odd degeneracies;
the explicit parity-grouped Fock basis is internal to this value and is shared
by every stored tensor.
"""
function FermionSiteOperators(modes::AbstractVector{<:Symbol})
    ordered_modes = Tuple(Symbol.(modes))
    isempty(ordered_modes) &&
        throw(ArgumentError("a fermion site needs at least one mode"))
    allunique(ordered_modes) ||
        throw(ArgumentError("local fermion modes must be unique"))

    states = _canonical_fock_states(length(ordered_modes))
    dimension = length(states)
    positions = Dict(state => position for (position, state) in enumerate(states))
    even = FermionParity(0)
    odd = FermionParity(1)
    P = Vect[FermionParity](even => dimension ÷ 2, odd => dimension ÷ 2)
    Q = Vect[FermionParity](odd => 1)

    annihilation_matrices = Matrix{ComplexF64}[]
    creation_matrices = Matrix{ComplexF64}[]
    number_matrices = Matrix{ComplexF64}[]
    for mode_index in eachindex(ordered_modes)
        mask = 1 << (mode_index - 1)
        annihilation = zeros(ComplexF64, dimension, dimension)
        creation = zeros(ComplexF64, dimension, dimension)
        number = zeros(ComplexF64, dimension, dimension)
        for state in states
            input = positions[state]
            if (state & mask) != 0
                target = state & ~mask
                sign = isodd(count_ones(state & (mask - 1))) ? -1.0 : 1.0
                annihilation[positions[target], input] = sign
                number[input, input] = 1.0
            else
                target = state | mask
                sign = isodd(count_ones(state & (mask - 1))) ? -1.0 : 1.0
                creation[positions[target], input] = sign
            end
        end
        push!(annihilation_matrices, annihilation)
        push!(creation_matrices, creation)
        push!(number_matrices, number)
    end
    identity = TensorMap(Matrix{ComplexF64}(I, dimension, dimension), P ← P)
    C = Tuple(_charged_local_tensor(matrix, P, Q)
              for matrix in annihilation_matrices)
    Cd = Tuple(_charged_local_tensor(matrix, P, Q)
               for matrix in creation_matrices)
    N = Tuple(TensorMap(matrix, P ← P) for matrix in number_matrices)
    return FermionSiteOperators(ordered_modes, P, C, Cd, N, identity)
end

"""
    FermionSiteOperators(layout, site)

Construct the local fZ2 carrier for one declared physical site of a
`FlavorLayout`.  The immutable layout's `site_modes` order is used verbatim.
"""
function FermionSiteOperators(layout::FlavorLayout, site::Symbol)
    return FermionSiteOperators(collect(site_modes(layout, site)))
end

function _local_mode_index(operators::FermionSiteOperators, mode::Symbol)
    index = findfirst(==(mode), operators.modes)
    index === nothing && throw(KeyError(mode))
    return index
end

"""Annihilation tensor for one explicitly declared local fermion mode."""
local_annihilator(operators::FermionSiteOperators, mode::Symbol) =
    operators.C[_local_mode_index(operators, mode)]

"""Creation tensor for one explicitly declared local fermion mode."""
local_creator(operators::FermionSiteOperators, mode::Symbol) =
    operators.Cd[_local_mode_index(operators, mode)]

"""Number tensor for one explicitly declared local fermion mode."""
local_number(operators::FermionSiteOperators, mode::Symbol) =
    operators.N[_local_mode_index(operators, mode)]
