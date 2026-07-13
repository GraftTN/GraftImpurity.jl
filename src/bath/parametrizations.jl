"""
    BlockRealPoles(layout, partition, poles, residues, block_indices;
                   statistics)

Real-pole expansion grouped by a named Partition. Scalar residues are stored
as finite real or complex numbers; matrix residues are stored as full finite
complex matrices local to their declared block. This is an expansion, not a
Hamiltonian-realizability proof: M3 validates Hermiticity and PSD without
silently projecting or deleting off-diagonal data.
"""
struct BlockRealPoles{R} <: AbstractBathParametrization
    layout::FlavorLayout
    partition::Partition
    poles::Vector{Float64}
    residues::Vector{R}
    block_indices::Vector{Int}
    statistics::Symbol

    function BlockRealPoles(layout::FlavorLayout, partition::Partition,
                            poles::Vector{Float64}, residues::Vector{R},
                            block_indices::Vector{Int}, statistics::Symbol,
                            ::Val{:validated}) where {R}
        new{R}(layout, partition, poles, residues, block_indices, statistics)
    end
end

function _validate_pole_header(layout::FlavorLayout, partition::Partition,
                               poles::AbstractVector{<:Real}, residues,
                               block_indices::AbstractVector{<:Integer},
                               statistics::Symbol)
    validate_partition(partition, layout)
    length(poles) == length(residues) == length(block_indices) ||
        throw(DimensionMismatch("BlockRealPoles needs one residue and block index per pole"))
    statistics in (:fermion, :boson) ||
        throw(ArgumentError("BlockRealPoles statistics must be :fermion or :boson"))
    values = Float64.(poles)
    all(isfinite, values) ||
        throw(ArgumentError("BlockRealPoles poles must be finite and real"))
    blocks = Int.(block_indices)
    all(index -> 1 <= index <= length(block_names(partition)), blocks) ||
        throw(ArgumentError("BlockRealPoles block indices must reference the Partition"))
    return values, blocks
end

function BlockRealPoles(layout::FlavorLayout, partition::Partition,
                        poles::AbstractVector{<:Real},
                        residues::AbstractVector{<:Number},
                        block_indices::AbstractVector{<:Integer};
                        statistics::Symbol)
    values, blocks = _validate_pole_header(
        layout, partition, poles, residues, block_indices, statistics)
    all(index -> length(block_flavors(partition,
                                      block_names(partition)[index])) == 1,
        blocks) ||
        throw(DimensionMismatch(
            "scalar BlockRealPoles residues require one-flavor partition blocks",
        ))
    scalar_residues = if all(isreal, residues)
        Float64.(real.(residues))
    else
        ComplexF64.(residues)
    end
    all(value -> isfinite(real(value)) && isfinite(imag(value)),
        scalar_residues) ||
        throw(ArgumentError("BlockRealPoles scalar residues must be finite"))
    return BlockRealPoles(layout, partition, values, scalar_residues, blocks,
                          statistics, Val(:validated))
end

function BlockRealPoles(layout::FlavorLayout, partition::Partition,
                        poles::AbstractVector{<:Real},
                        residues::AbstractVector{<:AbstractMatrix},
                        block_indices::AbstractVector{<:Integer};
                        statistics::Symbol)
    values, blocks = _validate_pole_header(
        layout, partition, poles, residues, block_indices, statistics)
    matrices = Matrix{ComplexF64}[]
    for (index, (residue, block)) in enumerate(zip(residues, blocks))
        dimension = length(block_flavors(partition, block_names(partition)[block]))
        size(residue) == (dimension, dimension) ||
            throw(DimensionMismatch(
                "BlockRealPoles residue $index must be $dimension by $dimension for block $block",
            ))
        matrix = Matrix{ComplexF64}(residue)
        all(value -> isfinite(real(value)) && isfinite(imag(value)), matrix) ||
            throw(ArgumentError("BlockRealPoles residue $index must be finite"))
        push!(matrices, matrix)
    end
    return BlockRealPoles(layout, partition, values, matrices, blocks,
                          statistics, Val(:validated))
end

Base.length(expansion::BlockRealPoles) = length(expansion.poles)

"""
    PoleExpansion(poles; kernel, trace=(;))

Kernel-labelled output of a real-pole fitting algorithm. trace preserves
algorithm-specific evidence without imposing one common fitting loop.
"""
struct PoleExpansion{P<:BlockRealPoles,T<:NamedTuple}
    poles::P
    kernel::Symbol
    trace::T

    function PoleExpansion(poles::P, kernel::Symbol, trace::T,
                           ::Val{:validated}) where {P<:BlockRealPoles,T<:NamedTuple}
        new{P,T}(poles, kernel, trace)
    end
end

function PoleExpansion(poles::BlockRealPoles;
                       kernel::Symbol,
                       trace::NamedTuple=(;))
    isempty(String(kernel)) &&
        throw(ArgumentError("PoleExpansion kernel name must be nonempty"))
    return PoleExpansion(poles, kernel, trace, Val(:validated))
end

"""
    BathOrbitals(energies, couplings, pole_indices, block_indices,
                 associated_flavors; layout, partition)

Canonical factorized bath modes. Its five stored fields deliberately retain
only pole-basis data. The constructor receives layout and partition to validate
every block-local coupling vector and explicit owner; DiscreteBath owns the
shared FlavorLayout token for the resulting Hamiltonian bath.
"""
struct BathOrbitals
    energies::Vector{Float64}
    couplings::Vector{Vector{ComplexF64}}
    pole_indices::Vector{Int}
    block_indices::Vector{Int}
    associated_flavors::Vector{Symbol}

    function BathOrbitals(energies::Vector{Float64},
                          couplings::Vector{Vector{ComplexF64}},
                          pole_indices::Vector{Int},
                          block_indices::Vector{Int},
                          associated_flavors::Vector{Symbol},
                          ::Val{:validated})
        new(energies, couplings, pole_indices, block_indices,
            associated_flavors)
    end
end

function BathOrbitals(energies::AbstractVector{<:Real},
                      couplings::AbstractVector{<:AbstractVector{<:Number}},
                      pole_indices::AbstractVector{<:Integer},
                      block_indices::AbstractVector{<:Integer},
                      associated_flavors::AbstractVector{Symbol};
                      layout::FlavorLayout, partition::Partition)
    validate_partition(partition, layout)
    mode_count = length(energies)
    length(couplings) == mode_count &&
        length(pole_indices) == mode_count &&
        length(block_indices) == mode_count &&
        length(associated_flavors) == mode_count ||
        throw(DimensionMismatch("BathOrbitals needs one coupling, pole, block, and owner per mode"))

    epsilon = Float64.(energies)
    vectors = Vector{ComplexF64}[ComplexF64.(coupling) for coupling in couplings]
    poles = Int.(pole_indices)
    blocks = Int.(block_indices)
    owners = Symbol.(associated_flavors)
    all(isfinite, epsilon) ||
        throw(ArgumentError("BathOrbitals energies must be finite"))
    all(vector -> all(value -> isfinite(real(value)) && isfinite(imag(value)),
                      vector), vectors) ||
        throw(ArgumentError("BathOrbitals couplings must be finite"))
    all(index -> index > 0, poles) ||
        throw(ArgumentError("BathOrbitals pole indices must be positive"))
    all(index -> 1 <= index <= length(block_names(partition)), blocks) ||
        throw(ArgumentError("BathOrbitals block indices must reference the Partition"))

    for (vector, block, owner) in zip(vectors, blocks, owners)
        block_flavor_order = block_flavors(
            partition, block_names(partition)[block])
        length(vector) == length(block_flavor_order) ||
            throw(DimensionMismatch(
                "BathOrbitals coupling must match its block-local flavor dimension",
            ))
        owner in block_flavor_order ||
            throw(ArgumentError("BathOrbitals owner $owner is outside its declared block"))
        owner in flavors(layout) ||
            throw(ArgumentError("BathOrbitals owner $owner is not in the FlavorLayout"))
    end
    return BathOrbitals(epsilon, vectors, poles, blocks, owners,
                        Val(:validated))
end

Base.length(orbitals::BathOrbitals) = length(orbitals.energies)
