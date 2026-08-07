"""
    DiscreteBath(layout, partition, orbitals; statistics)

Canonical finite Hamiltonian bath in its diagonal star pole basis. BathOrbitals
provides diagonal energies and complete coupling vectors local to named blocks;
this type supplies their shared FlavorLayout and Partition. It contains no
topology, mounted site labels, hopping-basis transform, or symbolic Hamiltonian.
"""
struct DiscreteBath{O<:BathOrbitals} <: AbstractHamiltonianBath
    layout::FlavorLayout
    partition::Partition
    orbitals::O
    statistics::Symbol

    function DiscreteBath(layout::FlavorLayout, partition::Partition,
                          orbitals::O, statistics::Symbol,
                          ::Val{:validated}) where {O<:BathOrbitals}
        new{O}(layout, partition, orbitals, statistics)
    end
end

function DiscreteBath(layout::FlavorLayout, partition::Partition,
                      orbitals::BathOrbitals; statistics::Symbol)
    validate_partition(partition, layout)
    statistics in (:fermion, :boson) ||
        throw(ArgumentError("DiscreteBath statistics must be :fermion or :boson"))
    return DiscreteBath(layout, partition, orbitals, statistics,
                        Val(:validated))
end

"""FlavorLayout shared by this canonical Hamiltonian bath."""
bath_layout(bath::DiscreteBath) = bath.layout

"""Named hybridization partition shared by this canonical Hamiltonian bath."""
bath_partition(bath::DiscreteBath) = bath.partition

"""Canonical diagonal-star bath orbitals."""
bath_orbitals(bath::DiscreteBath) = bath.orbitals

"""Particle statistics declared for this canonical Hamiltonian bath."""
bath_statistics(bath::DiscreteBath) = bath.statistics

Base.length(bath::DiscreteBath) = length(bath.orbitals)

Base.:(==)(left::DiscreteBath, right::DiscreteBath) =
    left.layout == right.layout &&
    left.partition == right.partition &&
    left.orbitals == right.orbitals &&
    left.statistics == right.statistics

Base.hash(bath::DiscreteBath, seed::UInt) = hash(
    (bath.layout, bath.partition, bath.orbitals, bath.statistics),
    hash(:DiscreteBath, seed),
)

function Base.copy(bath::DiscreteBath)
    return DiscreteBath(
        bath.layout, bath.partition, copy(bath.orbitals), bath.statistics,
        Val(:validated),
    )
end
