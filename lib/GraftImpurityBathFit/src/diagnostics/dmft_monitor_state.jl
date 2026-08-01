function _dmft_input_structure_matches(left::BathFitInput,
                                       right::BathFitInput)
    left.layout == right.layout || return false
    left.domain === right.domain || return false
    left.statistics === right.statistics || return false
    left.frequencies == right.frequencies || return false
    Tuple(keys(left.blocks)) == Tuple(keys(right.blocks)) || return false
    for block in keys(left.blocks)
        left_samples = getproperty(left.blocks, block)
        right_samples = getproperty(right.blocks, block)
        length(left_samples) == length(right_samples) || return false
        all(size(first_pair) == size(last_pair)
            for (first_pair, last_pair) in zip(left_samples, right_samples)) ||
            return false
    end
    return true
end

function _dmft_block_norm(samples)
    return sqrt(sum(sum(abs2, sample) for sample in samples))
end

function _dmft_block_difference(left, right)
    length(left) == length(right) || throw(DimensionMismatch(
        "DMFT bath update needs matching sample counts",
    ))
    return sqrt(sum(sum(abs2, left_sample - right_sample)
                    for (left_sample, right_sample) in zip(left, right)))
end

function _dmft_ratio(numerator::Float64, denominator::Float64)
    iszero(denominator) && return iszero(numerator) ? 0.0 : Inf
    return numerator / denominator
end

function _dmft_fit_report(value)
    value isa BathFitReport && return value
    for property in (:report, :fit_report)
        hasproperty(value, property) || continue
        candidate = getproperty(value, property)
        candidate isa BathFitReport && return candidate
    end
    for property in (:result, :fit_result)
        hasproperty(value, property) || continue
        candidate = _dmft_fit_report(getproperty(value, property))
        candidate === nothing || return candidate
    end
    return nothing
end

function _dmft_prediction(value, report)
    if hasproperty(value, :prediction)
        prediction = getproperty(value, :prediction)
        prediction isa BathFitInput && return prediction
    end
    if report !== nothing
        prediction = report.reconstruction
        prediction isa BathFitInput && return prediction
    end
    for property in (:result, :fit_result)
        hasproperty(value, property) || continue
        prediction = _dmft_prediction(getproperty(value, property), nothing)
        prediction === nothing || return prediction
    end
    return nothing
end

function _dmft_health(value, report)
    if report !== nothing && hasproperty(report, :health)
        health = getproperty(report, :health)
        health === nothing || return health
    end
    for property in (:health, :health_report, :bath_fit_health)
        hasproperty(value, property) || continue
        health = getproperty(value, property)
        health === nothing || return health
    end
    for property in (:result, :fit_result)
        hasproperty(value, property) || continue
        health = _dmft_health(getproperty(value, property), nothing)
        health === nothing || return health
    end
    return nothing
end

function _dmft_validate_health_source(health, target::BathFitInput)
    health === nothing && return nothing
    if hasproperty(health, :layout)
        getproperty(health, :layout) == target.layout || throw(ArgumentError(
            "DMFT bath health layout does not match target_input",
        ))
    end
    if hasproperty(health, :statistics)
        getproperty(health, :statistics) === target.statistics ||
            throw(ArgumentError(
                "DMFT bath health statistics do not match target_input",
            ))
    end
    return nothing
end

function _dmft_expansion(value)
    if value isa PoleExpansion
        return value
    elseif hasproperty(value, :expansion)
        expansion = getproperty(value, :expansion)
        expansion isa PoleExpansion && return expansion
    end
    for property in (:result, :fit_result)
        hasproperty(value, property) || continue
        expansion = _dmft_expansion(getproperty(value, property))
        expansion === nothing || return expansion
    end
    return nothing
end

function _dmft_validate_expansion_source(expansion,
                                         target::BathFitInput)
    expansion === nothing && return nothing
    expansion.poles.layout == target.layout || throw(ArgumentError(
        "DMFT bath pole expansion layout does not match target_input",
    ))
    expansion.poles.statistics === target.statistics ||
        throw(ArgumentError(
            "DMFT bath pole expansion statistics do not match target_input",
        ))
    _validate_fit_input(target, expansion.poles.partition)
    return nothing
end

function _dmft_trace_weight(residue)
    if residue isa Number
        value = ComplexF64(residue)
        tolerance = sqrt(eps(Float64)) * max(abs(value), 1.0)
        abs(imag(value)) <= tolerance && real(value) >= -tolerance ||
            return nothing
        return max(real(value), 0.0)
    end
    matrix = Matrix{ComplexF64}(residue)
    tolerance = sqrt(eps(Float64)) * max(opnorm(matrix), 1.0)
    norm(matrix - adjoint(matrix)) <= tolerance || return nothing
    hermitian = Hermitian((matrix + adjoint(matrix)) / 2)
    minimum(real.(eigvals(hermitian))) >= -tolerance || return nothing
    return max(real(tr(hermitian)), 0.0)
end

function _dmft_positive_measure(expansion::PoleExpansion, block::Symbol)
    partition = expansion.poles.partition
    block_value = block_index(partition, block)
    energies = Float64[]
    weights = Float64[]
    for pole_index in eachindex(expansion.poles.poles)
        expansion.poles.block_indices[pole_index] == block_value || continue
        weight = _dmft_trace_weight(expansion.poles.residues[pole_index])
        weight === nothing && return nothing
        iszero(weight) && continue
        push!(energies, expansion.poles.poles[pole_index])
        push!(weights, weight)
    end
    mass = sum(weights)
    isfinite(mass) && mass > 0 || return nothing
    permutation = sortperm(energies)
    return _DMFTPositiveMeasure(
        energies[permutation], weights[permutation], mass,
    )
end

function _dmft_measure_snapshot(expansion, names::Tuple)
    expansion === nothing && return nothing
    expansion_names = Tuple(block_names(expansion.poles.partition))
    expansion_names == names || return nothing
    measures = Tuple(_dmft_positive_measure(expansion, block) for block in names)
    return _DMFTMeasureSnapshot(NamedTuple{names}(measures))
end

function _dmft_wasserstein_shape(left::_DMFTPositiveMeasure,
                                 right::_DMFTPositiveMeasure)
    left_weights = left.weights ./ left.mass
    right_weights = right.weights ./ right.mass
    left_index = firstindex(left.energies)
    right_index = firstindex(right.energies)
    left_remaining = left_weights[left_index]
    right_remaining = right_weights[right_index]
    distance = 0.0
    tolerance = 16eps(Float64)
    while left_index <= lastindex(left.energies) &&
          right_index <= lastindex(right.energies)
        transported = min(left_remaining, right_remaining)
        distance += transported *
                    abs(left.energies[left_index] - right.energies[right_index])
        left_remaining -= transported
        right_remaining -= transported
        if left_remaining <= tolerance
            left_index += 1
            left_index <= lastindex(left.energies) &&
                (left_remaining = left_weights[left_index])
        end
        if right_remaining <= tolerance
            right_index += 1
            right_index <= lastindex(right.energies) &&
                (right_remaining = right_weights[right_index])
        end
    end
    return distance
end

function _dmft_measure_updates(current, previous)
    current === nothing && return (nothing, nothing)
    previous === nothing && return (nothing, nothing)
    mass_scale = max(current.mass, previous.mass)
    mass = iszero(mass_scale) ? 0.0 :
           abs(current.mass - previous.mass) / mass_scale
    return mass, _dmft_wasserstein_shape(current, previous)
end

function _dmft_maximum_optional(values)
    present = Float64[value for value in values if value !== nothing]
    isempty(present) && return nothing
    return maximum(present)
end

function _dmft_complexity_integer(value)
    value isa Integer && return Int(value)
    value isa Real && isfinite(value) && isinteger(value) && return Int(value)
    return nothing
end

function _dmft_nested_block(value, block::Symbol)
    value === nothing && return nothing
    for property in (:blocks, :block_reports, :health)
        hasproperty(value, property) || continue
        container = getproperty(value, property)
        if container isa NamedTuple && hasproperty(container, block)
            return getproperty(container, block)
        elseif container isa AbstractDict && haskey(container, block)
            return container[block]
        end
    end
    return nothing
end

function _dmft_named_integer(value, names::Tuple)
    value === nothing && return nothing
    for name in names
        hasproperty(value, name) || continue
        resolved = _dmft_complexity_integer(getproperty(value, name))
        resolved === nothing || return resolved
    end
    return nothing
end

function _dmft_selected_order_health(health)
    health === nothing && return nothing
    hasproperty(health, :orders) || return nothing
    hasproperty(health, :selected_order) || return nothing
    selected_order = _dmft_complexity_integer(
        getproperty(health, :selected_order),
    )
    selected_order === nothing && return nothing
    for candidate in getproperty(health, :orders)
        hasproperty(candidate, :order) || continue
        _dmft_complexity_integer(getproperty(candidate, :order)) ==
            selected_order && return candidate
    end
    return nothing
end

function _dmft_complexity(report, health, block::Symbol)
    selected_health = _dmft_selected_order_health(health)
    health_block = selected_health === nothing ?
                   _dmft_nested_block(health, block) : selected_health
    report_block = _dmft_nested_block(report, block)
    order = _dmft_named_integer(
        health,
        (:selected_order, :order),
    )
    order === nothing && (order = _dmft_named_integer(
        health_block === nothing ? health : health_block,
        (:order, :selected_order, :effective_order, :requested_order),
    ))
    q_train = health_block !== nothing &&
              hasproperty(health_block, :q_train) ?
              getproperty(health_block, :q_train) : nothing
    q_validation = health_block !== nothing &&
                   hasproperty(health_block, :q_validation) ?
                   getproperty(health_block, :q_validation) : nothing
    returned_poles = health_block !== nothing &&
                     hasproperty(health_block, :returned_poles) ?
                     getproperty(health_block, :returned_poles) : nothing
    mode_counts = health_block !== nothing &&
                  hasproperty(health_block, :mode_counts) ?
                  getproperty(health_block, :mode_counts) : nothing
    pole_count = _dmft_named_integer(
        health_block === nothing ? health : health_block,
        (:pole_count, :poles),
    )
    pole_count === nothing &&
        (pole_count = _dmft_named_integer(report_block, (:pole_count,)))
    mode_count = _dmft_named_integer(
        health_block === nothing ? health : health_block,
        (:mode_count, :modes, :realized_mode_count),
    )
    mode_count === nothing &&
        (mode_count = _dmft_named_integer(report_block, (:mode_count,)))
    order === nothing && (order = pole_count)
    return DMFTBathComplexity(
        order, q_train, q_validation, returned_poles, mode_counts,
        pole_count, mode_count,
    )
end

function _dmft_same_source(target::BathFitInput, source::BathFitInput)
    _dmft_input_structure_matches(target, source) || return false
    difference = 0.0
    scale = 0.0
    for block in keys(target.blocks)
        target_samples = getproperty(target.blocks, block)
        source_samples = getproperty(source.blocks, block)
        difference = hypot(
            difference, _dmft_block_difference(target_samples, source_samples),
        )
        scale = hypot(scale, _dmft_block_norm(target_samples))
    end
    return difference <= sqrt(eps(Float64)) * max(scale, 1.0)
end
