function _preparation_h_loc_layout(h_loc)
    hasproperty(h_loc, :layout) || throw(ArgumentError(
        "preparation h_loc must carry a FlavorLayout",
    ))
    layout = getproperty(h_loc, :layout)
    layout isa FlavorLayout || throw(ArgumentError(
        "preparation h_loc.layout must be a FlavorLayout",
    ))
    return layout
end

function _validate_preparation_fit_input(input::BathFitInput,
                                         partition::Partition;
                                         require_temperature::Bool=true)
    _validate_fit_input(input, partition)
    input.statistics === :fermion || throw(ArgumentError(
        "impurity preparation currently supports fermionic GreenFunc inputs only",
    ))
    !require_temperature || hasproperty(input.metadata, :temperature) ||
        throw(ArgumentError(
            "GreenFunc impurity preparation metadata must retain a temperature context",
        ))
    return input
end

function _preparation_input_from_gf(h_loc, partition::Partition,
                                    gf::GreenFunc.Gf;
                                    block::Union{Nothing,Symbol}=nothing)
    layout = _preparation_h_loc_layout(h_loc)
    validate_partition(partition, layout)
    names = block_names(partition)
    selected = if block === nothing
        length(names) == 1 || throw(ArgumentError(
            "a single GreenFunc.Gf needs an explicit block for a multi-block preparation",
        ))
        only(names)
    else
        block in names || throw(ArgumentError(
            "GreenFunc.Gf block $block is absent from the preparation Partition",
        ))
        block
    end
    return _validate_preparation_fit_input(
        BathFitInput(layout, gf, selected), partition,
    )
end

function _preparation_input_from_gf(h_loc, partition::Partition,
                                    blocks::GreenFunc.BlockGf)
    layout = _preparation_h_loc_layout(h_loc)
    validate_partition(partition, layout)
    return _validate_preparation_fit_input(
        BathFitInput(layout, blocks), partition,
    )
end

function HybridizationPreparationInput(
        delta::GreenFunc.Gf, partition::Partition;
        h_loc, block::Union{Nothing,Symbol}=nothing)
    owned_h_loc, owned_partition = deepcopy((h_loc, partition))
    input = _preparation_input_from_gf(
        owned_h_loc, owned_partition, delta; block,
    )
    return HybridizationPreparationInput(
        input, input, owned_h_loc, owned_partition, Val(:validated),
    )
end

function HybridizationPreparationInput(
        input::BathFitInput, partition::Partition; h_loc)
    owned_input, owned_h_loc, owned_partition =
        deepcopy((input, h_loc, partition))
    _preparation_h_loc_layout(owned_h_loc) == owned_input.layout || throw(ArgumentError(
        "BathFitInput FlavorLayout must match preparation h_loc",
    ))
    owned = _validate_preparation_fit_input(
        owned_input, owned_partition; require_temperature=false,
    )
    return HybridizationPreparationInput(
        owned, owned, owned_h_loc, owned_partition, Val(:validated),
    )
end

function HybridizationPreparationInput(
        delta::GreenFunc.BlockGf, partition::Partition; h_loc)
    owned_h_loc, owned_partition = deepcopy((h_loc, partition))
    input = _preparation_input_from_gf(owned_h_loc, owned_partition, delta)
    return HybridizationPreparationInput(
        input, input, owned_h_loc, owned_partition, Val(:validated),
    )
end

function _preparation_matrix_tolerance(matrix)
    scale = maximum(abs, matrix; init=0.0)
    return 128 * eps(Float64) * max(1.0, Float64(scale))
end

function _require_block_local_weiss_onebody(h_loc, partition::Partition)
    hasproperty(h_loc, :matrix) || throw(ArgumentError(
        "Weiss preparation h_loc must carry a one-body matrix",
    ))
    matrix = getproperty(h_loc, :matrix)
    layout = _preparation_h_loc_layout(h_loc)
    tolerance = _preparation_matrix_tolerance(matrix)
    names = block_names(partition)
    for left in eachindex(names), right in eachindex(names)
        left == right && continue
        left_indices = [flavor_index(layout, flavor)
                        for flavor in block_flavors(partition, names[left])]
        right_indices = [flavor_index(layout, flavor)
                         for flavor in block_flavors(partition, names[right])]
        maximum(abs, @view matrix[left_indices, right_indices]; init=0.0) <=
            tolerance || throw(ArgumentError(
                "Weiss preparation requires block-local h_loc because a BlockGf " *
                "source cannot represent cross-block hybridization",
            ))
    end
    return h_loc
end

function _weiss_hybridization_input(input::BathFitInput, h_loc,
                                    partition::Partition)
    input.domain === :matsubara || throw(ArgumentError(
        "Weiss preparation requires a Matsubara GreenFunc input",
    ))
    input.statistics === :fermion || throw(ArgumentError(
        "Weiss preparation requires fermionic GreenFunc statistics",
    ))
    _require_block_local_weiss_onebody(h_loc, partition)
    any(name -> hasproperty(input.metadata, name),
        (:weiss_conversion, :h_loc_label)) && throw(ArgumentError(
            "GreenFunc metadata keys :weiss_conversion and :h_loc_label are " *
            "reserved by Weiss preparation",
        ))
    layout = _preparation_h_loc_layout(h_loc)
    matrix = getproperty(h_loc, :matrix)
    label = hasproperty(h_loc, :label) ? getproperty(h_loc, :label) : :h_loc
    names = block_names(partition)
    converted = Tuple(map(names) do name
        indices = [flavor_index(layout, flavor)
                   for flavor in block_flavors(partition, name)]
        local_h = Matrix{ComplexF64}(matrix[indices, indices])
        dimension = length(indices)
        samples = Matrix{ComplexF64}[]
        for (index, sample) in enumerate(getproperty(input.blocks, name))
            inverse = try
                LinearAlgebra.inv(sample)
            catch error
                error isa LinearAlgebra.SingularException || rethrow()
                throw(ArgumentError(
                    "Weiss GreenFunc block $name is singular at Matsubara index $index",
                ))
            end
            candidate = ComplexF64(im * input.frequencies[index]) *
                Matrix{ComplexF64}(I, dimension, dimension) - local_h - inverse
            all(value -> isfinite(real(value)) && isfinite(imag(value)), candidate) ||
                throw(ArgumentError(
                    "Weiss conversion produced a nonfinite hybridization block " *
                    "$name at Matsubara index $index",
                ))
            push!(samples, candidate)
        end
        samples
    end)
    blocks = NamedTuple{names}(converted)
    metadata = merge(input.metadata, (; weiss_conversion=:explicit_inverse,
                                      h_loc_label=label))
    template = _reconstructed_template(input, blocks)
    hybridization = BathFitInput(
        input.layout, input.domain, input.statistics, copy(input.frequencies),
        blocks, input.target_labels, metadata, template, Val(:validated),
    )
    _validate_fit_input(hybridization, partition)
    return hybridization
end

function WeissPreparationInput(
        weiss::GreenFunc.Gf, partition::Partition;
        h_loc, block::Union{Nothing,Symbol}=nothing)
    owned_h_loc, owned_partition = deepcopy((h_loc, partition))
    source = _preparation_input_from_gf(
        owned_h_loc, owned_partition, weiss; block,
    )
    hybridization = _weiss_hybridization_input(
        source, owned_h_loc, owned_partition,
    )
    return WeissPreparationInput(
        source, hybridization, owned_h_loc, owned_partition, Val(:validated),
    )
end

function WeissPreparationInput(
        input::BathFitInput, partition::Partition; h_loc)
    owned_input, owned_h_loc, owned_partition =
        deepcopy((input, h_loc, partition))
    _preparation_h_loc_layout(owned_h_loc) == owned_input.layout || throw(ArgumentError(
        "BathFitInput FlavorLayout must match preparation h_loc",
    ))
    source = _validate_preparation_fit_input(
        owned_input, owned_partition; require_temperature=false,
    )
    hybridization = _weiss_hybridization_input(
        source, owned_h_loc, owned_partition,
    )
    return WeissPreparationInput(
        source, hybridization, owned_h_loc, owned_partition, Val(:validated),
    )
end

function WeissPreparationInput(
        weiss::GreenFunc.BlockGf, partition::Partition; h_loc)
    owned_h_loc, owned_partition = deepcopy((h_loc, partition))
    source = _preparation_input_from_gf(owned_h_loc, owned_partition, weiss)
    hybridization = _weiss_hybridization_input(
        source, owned_h_loc, owned_partition,
    )
    return WeissPreparationInput(
        source, hybridization, owned_h_loc, owned_partition, Val(:validated),
    )
end
