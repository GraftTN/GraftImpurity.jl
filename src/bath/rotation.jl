function _global_bath_couplings(bath::DiscreteBath)
    layout = bath_layout(bath)
    partition = bath_partition(bath)
    orbitals = bath_orbitals(bath)
    matrix = zeros(ComplexF64, length(flavors(layout)), length(orbitals))
    for mode in eachindex(orbitals.energies)
        block = block_names(partition)[orbitals.block_indices[mode]]
        for (component, flavor) in enumerate(block_flavors(partition, block))
            matrix[flavor_index(layout, flavor), mode] = orbitals.couplings[mode][component]
        end
    end
    return matrix
end

function _rotation_mode_blocks(mode_blocks, associated_flavors,
                               bath::DiscreteBath, partition::Partition)
    modes = length(bath)
    blocks = Symbol.(mode_blocks)
    owners = Symbol.(associated_flavors)
    length(blocks) == modes || throw(DimensionMismatch(
        "rotate_bath needs one declared target block per canonical bath mode",
    ))
    length(owners) == modes || throw(DimensionMismatch(
        "rotate_bath needs one declared target owner per canonical bath mode",
    ))
    all(block -> block in block_names(partition), blocks) || throw(ArgumentError(
        "rotate_bath mode_blocks must name blocks in the new Partition",
    ))
    return blocks, owners
end

"""
    rotate_bath(bath, rotation, new_layout, new_partition;
                mode_blocks, associated_flavors) -> DiscreteBath

Rotate canonical hybridization couplings as `W_new = rotation' * W_old` while
preserving every bath energy and pole identity. A caller must explicitly choose
the new block and ownership label for every mode. If a declared target block
cannot contain the full rotated coupling vector, this throws rather than
dropping off-block components or inferring a dominant arm.
"""
function rotate_bath(bath::DiscreteBath,
                     rotation::AbstractMatrix{<:Number},
                     new_layout::FlavorLayout,
                     new_partition::Partition;
                     mode_blocks,
                     associated_flavors)
    matrix = _validated_basis_rotation(rotation, bath_layout(bath), new_layout)
    validate_partition(new_partition, new_layout)
    blocks, owners = _rotation_mode_blocks(
        mode_blocks, associated_flavors, bath, new_partition,
    )
    transformed = matrix' * _global_bath_couplings(bath)
    block_indices = Int[]
    couplings = Vector{ComplexF64}[]
    for mode in axes(transformed, 2)
        block = blocks[mode]
        flavors_in_block = block_flavors(new_partition, block)
        inside = [flavor_index(new_layout, flavor) for flavor in flavors_in_block]
        outside = setdiff(collect(axes(transformed, 1)), inside)
        scale = max(1.0, norm(@view transformed[:, mode]))
        tolerance = 128 * eps(Float64) * scale
        all(index -> abs(transformed[index, mode]) <= tolerance, outside) ||
            throw(ArgumentError(
                "rotated bath mode $mode has coupling support outside its explicitly declared target block $block",
            ))
        push!(block_indices, findfirst(==(block), block_names(new_partition)))
        push!(couplings, ComplexF64[transformed[index, mode] for index in inside])
    end
    source = bath_orbitals(bath)
    orbitals = BathOrbitals(
        source.energies, couplings, source.pole_indices, block_indices, owners;
        layout=new_layout, partition=new_partition,
    )
    return DiscreteBath(new_layout, new_partition, orbitals;
                        statistics=bath_statistics(bath))
end
