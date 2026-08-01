"""
    Partition(:block_name => [:flavor_a, :flavor_b], ...)

Immutable named partition of one-particle Green-function or hybridization
blocks. Block names and order are part of the value identity. A partition does
not constrain the connectivity of a many-body interaction; that remains the
responsibility of the interaction type and its shared FlavorLayout.
"""
struct Partition{B<:NamedTuple}
    blocks::B

    function Partition(blocks::B, ::Val{:validated}) where {B<:NamedTuple}
        return new{B}(blocks)
    end
end

function Partition(groups::Pair...)
    isempty(groups) &&
        throw(ArgumentError("Partition needs at least one named block"))
    all(group -> group.first isa Symbol, groups) ||
        throw(ArgumentError("Partition block names must be Symbols"))

    names = Symbol[group.first for group in groups]
    allunique(names) ||
        throw(ArgumentError("Partition block names must be unique"))

    values = Tuple(begin
        group.second isa AbstractVector ||
            throw(ArgumentError("Partition block $(group.first) must be a vector of flavor Symbols"))
        flavors = Tuple(Symbol.(group.second))
        isempty(flavors) &&
            throw(ArgumentError("Partition block $(group.first) may not be empty"))
        allunique(flavors) ||
            throw(ArgumentError("Partition block $(group.first) repeats a flavor"))
        flavors
    end for group in groups)
    owned = Symbol[flavor for block in values for flavor in block]
    allunique(owned) ||
        throw(ArgumentError("Partition flavors may occur in only one named block"))

    blocks = NamedTuple{Tuple(names)}(values)
    return Partition(blocks, Val(:validated))
end

"""Names of partition blocks in deterministic declared order."""
block_names(partition::Partition) = keys(partition.blocks)

"""Ordered flavor labels owned by a named partition block."""
function block_flavors(partition::Partition, block::Symbol)
    hasproperty(partition.blocks, block) || throw(KeyError(block))
    return getproperty(partition.blocks, block)
end

"""One-based declared position of a named partition block."""
function block_index(partition::Partition, block::Symbol)
    index = findfirst(==(block), block_names(partition))
    index === nothing && throw(KeyError(block))
    return index
end

"""Flattened partition flavor sequence in declared block and within-block order."""
partition_flavors(partition::Partition) =
    Tuple(flavor for block in block_names(partition)
          for flavor in block_flavors(partition, block))

"""
    validate_partition(partition, layout) -> partition

Verify that named hybridization blocks reproduce the canonical FlavorLayout
sequence exactly. This is intentionally separate from the Partition constructor
so a declaration can be assembled before the layout-bearing physical model is
available.
"""
function validate_partition(partition::Partition, layout::FlavorLayout)
    declared = partition_flavors(partition)
    declared == flavors(layout) ||
        throw(ArgumentError("Partition flavor order must match the FlavorLayout exactly"))
    return partition
end

Base.:(==)(left::Partition, right::Partition) = left.blocks == right.blocks
Base.hash(partition::Partition, seed::UInt) =
    hash(partition.blocks, hash(:Partition, seed))
