"""Marker for a local fermion carrier graded only by fermion parity."""
struct FermionParitySector end

"""
Marker for a local fermion carrier graded by fZ2 × particle number `U(1)_N`.

The parity component retains TensorKit's fermionic braiding across sites; the
number component supplies the actual conserved charge sector.
"""
struct ParticleNumberSector end

const AbstractFermionSector = Union{FermionParitySector,ParticleNumberSector}

"""
    FermionSiteOperators(modes; sector=FermionParitySector())

Immutable local fermion Fock-space operator carrier for one physical impurity
site. `modes` fixes the canonical local fermion order. `FermionParitySector()`
keeps M5's fZ2 route, while `ParticleNumberSector()` constructs a fermionically
braided fZ2 × particle-number-`U(1)` carrier for number-conserving production
Hamiltonians.
The resulting `P`, `C`, `Cd`, `N`, and `I` tensors use Graft's graded Backend
vocabulary and therefore retain fermionic exchange through TensorKit rather
than introducing Jordan-Wigner strings. The occupancy convention is the
declared `modes` order, with
`c_j |n> = (-1)^(sum_{i<j} n_i) n_j |n-e_j>`.
"""
struct FermionSiteOperators{S<:AbstractFermionSector,P<:ElementarySpace,
                            C<:Tuple,D<:Tuple,N<:Tuple,I<:AbstractTensorMap}
    modes::Tuple{Vararg{Symbol}}
    sector::S
    P::P
    C::C
    Cd::D
    N::N
    I::I
end

function _canonical_fock_states(mode_count::Int, ::FermionParitySector)
    mode_count > 0 || throw(ArgumentError("a fermion site needs at least one mode"))
    mode_count < 8 * sizeof(Int) ||
        throw(ArgumentError("local fermion mode count exceeds the addressable Fock basis"))
    states = collect(0:((1 << mode_count) - 1))
    sort!(states; by=state -> (isodd(count_ones(state)), state))
    return states
end

function _canonical_fock_states(mode_count::Int, ::ParticleNumberSector)
    mode_count > 0 || throw(ArgumentError("a fermion site needs at least one mode"))
    mode_count < 8 * sizeof(Int) ||
        throw(ArgumentError("local fermion mode count exceeds the addressable Fock basis"))
    states = collect(0:((1 << mode_count) - 1))
    sort!(states; by=state -> (count_ones(state), state))
    return states
end

function _fermion_local_spaces(mode_count::Int, ::FermionParitySector)
    even = FermionParity(0)
    odd = FermionParity(1)
    P = Vect[FermionParity](even => (1 << (mode_count - 1)),
                             odd => (1 << (mode_count - 1)))
    Q = Vect[FermionParity](odd => 1)
    return P, Q, Q
end

function _fermion_local_spaces(mode_count::Int, ::ParticleNumberSector)
    sector(parity::Int, number::Int) =
        FermionParity(parity) ⊠ U1Irrep(number)
    S = typeof(sector(0, 0))
    P = Vect[S]((sector(number % 2, number) => binomial(mode_count, number)
                 for number in 0:mode_count)...)
    annihilation_charge = Vect[S](sector(1, -1) => 1)
    creation_charge = Vect[S](sector(1, 1) => 1)
    return P, annihilation_charge, creation_charge
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
    FermionSiteOperators(modes::AbstractVector{<:Symbol};
                         sector=FermionParitySector())

Construct local creation, annihilation, number, and identity operators in the
declared mode order in the requested explicit abelian sector representation.
The internal basis is grouped by parity or by particle number with its paired
fermion-parity sector accordingly and is shared by every stored tensor.
"""
function FermionSiteOperators(modes::AbstractVector{<:Symbol};
                              sector::AbstractFermionSector=FermionParitySector())
    ordered_modes = Tuple(Symbol.(modes))
    isempty(ordered_modes) &&
        throw(ArgumentError("a fermion site needs at least one mode"))
    allunique(ordered_modes) ||
        throw(ArgumentError("local fermion modes must be unique"))

    states = _canonical_fock_states(length(ordered_modes), sector)
    dimension = length(states)
    positions = Dict(state => position for (position, state) in enumerate(states))
    P, annihilation_charge, creation_charge = _fermion_local_spaces(
        length(ordered_modes), sector,
    )

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
    C = Tuple(_charged_local_tensor(matrix, P, annihilation_charge)
              for matrix in annihilation_matrices)
    Cd = Tuple(_charged_local_tensor(matrix, P, creation_charge)
               for matrix in creation_matrices)
    N = Tuple(TensorMap(matrix, P ← P) for matrix in number_matrices)
    return FermionSiteOperators(ordered_modes, sector, P, C, Cd, N, identity)
end

"""
    FermionSiteOperators(layout, site; sector=FermionParitySector())

Construct the requested explicit fermionic sector carrier for one declared
physical site of a `FlavorLayout`. The immutable layout's `site_modes` order
is used verbatim.
"""
function FermionSiteOperators(layout::FlavorLayout, site::Symbol;
                              sector::AbstractFermionSector=FermionParitySector())
    return FermionSiteOperators(collect(site_modes(layout, site)); sector)
end

"""Explicit abelian sector carried by a local fermion operator value."""
fermion_sector(operators::FermionSiteOperators) = operators.sector

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
