struct _BathHealthMetric
    whitening::Union{Nothing,Matrix{Float64}}
    rank::Int
    eigenvalue_floor::Float64
    mode::Symbol
end

struct _BathHealthMeasure
    energies::Vector{Float64}
    weights::Vector{Float64}
    states::Vector{Matrix{ComplexF64}}
    mass::Float64
end

function _bathhealth_complex_vector(input::BathFitInput)
    values = ComplexF64[]
    for samples in Base.values(input.blocks), sample in samples
        append!(values, vec(sample))
    end
    return values
end

function _bathhealth_dynamic_complex_mask(input::BathFitInput)
    mask = Bool[]
    for samples in Base.values(input.blocks)
        for (frequency, sample) in zip(input.frequencies, samples)
            dynamic = !(input.statistics === :boson && iszero(frequency))
            append!(mask, fill(dynamic, length(sample)))
        end
    end
    return mask
end

function _bathhealth_real_vector(input::BathFitInput; dynamic::Bool=true)
    values = _bathhealth_complex_vector(input)
    if dynamic
        values = values[_bathhealth_dynamic_complex_mask(input)]
    end
    return vcat(real.(values), imag.(values))
end

function _bathhealth_residual_vector(target::BathFitInput,
                                     prediction::BathFitInput;
                                     dynamic::Bool=true)
    _validate_residual_inputs(target, prediction)
    residual = ComplexF64[]
    for block in keys(target.blocks)
        target_samples = getproperty(target.blocks, block)
        prediction_samples = getproperty(prediction.blocks, block)
        for (expected, actual) in zip(target_samples, prediction_samples)
            append!(residual, vec(expected - actual))
        end
    end
    if dynamic
        residual = residual[_bathhealth_dynamic_complex_mask(target)]
    end
    return vcat(real.(residual), imag.(residual))
end

function _bathhealth_validate_domains(training::BathFitInput,
                                      validation::BathFitInput)
    training.layout == validation.layout || throw(ArgumentError(
        "bath-fit health training and validation layouts must match",
    ))
    training.statistics === validation.statistics || throw(ArgumentError(
        "bath-fit health training and validation statistics must match",
    ))
    Tuple(keys(training.blocks)) == Tuple(keys(validation.blocks)) ||
        throw(ArgumentError(
            "bath-fit health training and validation block names must match",
        ))
    for block in keys(training.blocks)
        training_shape = size(first(getproperty(training.blocks, block)))
        validation_shape = size(first(getproperty(validation.blocks, block)))
        training_shape == validation_shape || throw(DimensionMismatch(
            "bath-fit health block $block target shapes must match",
        ))
    end
    return nothing
end

function _bathhealth_covariance(covariance, role::Symbol, dimension::Int)
    covariance === nothing && return nothing, :none
    candidate = if covariance isa NamedTuple
        hasproperty(covariance, role) ? getproperty(covariance, role) : nothing
    else
        covariance
    end
    candidate === nothing && return nothing, :none
    matrix = Matrix{Float64}(candidate)
    size(matrix, 1) == size(matrix, 2) || throw(DimensionMismatch(
        "bath-fit health covariance must be square",
    ))
    all(isfinite, matrix) || throw(ArgumentError(
        "bath-fit health covariance must be finite",
    ))
    if size(matrix, 1) != dimension
        variance = sum(diag(matrix)) / size(matrix, 1)
        isfinite(variance) && variance > 0 || return nothing, :none
        return Matrix{Float64}(I, dimension, dimension) .* variance,
               :isotropic_transfer
    end
    return matrix, :explicit
end

function _bathhealth_metric(covariance, role::Symbol, dimension::Int)
    matrix, mode = _bathhealth_covariance(covariance, role, dimension)
    matrix === nothing && return _BathHealthMetric(nothing, dimension, 0.0, mode)
    symmetric = (matrix + transpose(matrix)) / 2
    tolerance = sqrt(eps(Float64)) * max(opnorm(symmetric), 1.0)
    norm(matrix - transpose(matrix)) <= tolerance || throw(ArgumentError(
        "bath-fit health covariance must be symmetric",
    ))
    decomposition = eigen(Hermitian(symmetric))
    minimum(decomposition.values; init=0.0) >= -tolerance ||
        throw(ArgumentError(
            "bath-fit health covariance must be positive semidefinite",
        ))
    maximum_value = maximum(decomposition.values; init=0.0)
    floor_value = max(eps(Float64), sqrt(eps(Float64)) * maximum_value)
    retained = findall(>(floor_value), decomposition.values)
    isempty(retained) && return _BathHealthMetric(
        nothing, dimension, floor_value, :rank_zero,
    )
    vectors = decomposition.vectors[:, retained]
    whitening = Diagonal(inv.(sqrt.(decomposition.values[retained]))) *
                transpose(vectors)
    return _BathHealthMetric(
        Matrix{Float64}(whitening), length(retained), floor_value, mode,
    )
end

function _bathhealth_metric(covariance, role::Symbol, input::BathFitInput)
    full_dimension = 2length(_bathhealth_complex_vector(input))
    complex_mask = _bathhealth_dynamic_complex_mask(input)
    real_mask = vcat(complex_mask, complex_mask)
    retained = findall(real_mask)
    matrix, mode = _bathhealth_covariance(
        covariance, role, full_dimension,
    )
    matrix === nothing && return _BathHealthMetric(
        nothing, length(retained), 0.0, mode,
    )
    return _bathhealth_metric(
        matrix[retained, retained], role, length(retained),
    )
end

function _bathhealth_score(residual::Vector{Float64},
                           target::Vector{Float64},
                           metric::_BathHealthMetric)
    if metric.whitening === nothing
        denominator = norm(target)
        return iszero(denominator) ?
               (iszero(norm(residual)) ? 0.0 : Inf) :
               norm(residual) / denominator
    end
    return norm(metric.whitening * residual) / sqrt(metric.rank)
end

function _bathhealth_distance(left::BathFitInput, right::BathFitInput,
                              metric::_BathHealthMetric)
    _validate_residual_inputs(left, right)
    difference = _bathhealth_residual_vector(left, right)
    if metric.whitening === nothing
        scale = max(norm(_bathhealth_real_vector(left)),
                    norm(_bathhealth_real_vector(right)))
        return iszero(scale) ? (iszero(norm(difference)) ? 0.0 : Inf) :
               norm(difference) / scale
    end
    return norm(metric.whitening * difference) / sqrt(metric.rank)
end

function _bathhealth_summary(values, confidence::Float64)
    isempty(values) && return BathFitMetricSummary(
        Float64[Inf], Inf, Inf, Inf, Inf, Inf,
    )
    samples = Float64.(values)
    if any(==(Inf), samples)
        ordered = sort(samples)
        tail = (1 - confidence) / 2
        return BathFitMetricSummary(
            samples, Inf, Inf, _bathfit_health_quantile(ordered, 0.5),
            _bathfit_health_quantile(ordered, tail),
            _bathfit_health_quantile(ordered, 1 - tail),
        )
    end
    return BathFitMetricSummary(samples; confidence)
end

function _bathhealth_zero_frequency(target::BathFitInput,
                                    prediction::BathFitInput)
    target.statistics === :boson || return nothing
    indices = findall(iszero, target.frequencies)
    isempty(indices) && return nothing
    numerator = 0.0
    denominator = 0.0
    for block in keys(target.blocks), index in indices
        expected = getproperty(target.blocks, block)[index]
        actual = getproperty(prediction.blocks, block)[index]
        numerator += sum(abs2, expected - actual)
        denominator += sum(abs2, expected)
    end
    return iszero(denominator) ?
           (iszero(numerator) ? 0.0 : Inf) :
           sqrt(numerator / denominator)
end

function _bathhealth_candidate_metrics(candidate::BathFitHealthCandidate,
                                       training::BathFitInput,
                                       validation::BathFitInput,
                                       training_metric::_BathHealthMetric,
                                       validation_metric::_BathHealthMetric)
    training_target = something(candidate.training_target, training)
    training_prediction = something(candidate.training_prediction)
    validation_prediction = something(candidate.validation_prediction)
    train_residual = _bathhealth_residual_vector(
        training_target, training_prediction,
    )
    validation_residual = _bathhealth_residual_vector(
        validation, validation_prediction,
    )
    q_train = _bathhealth_score(
        train_residual, _bathhealth_real_vector(training_target),
        training_metric,
    )
    q_validation = _bathhealth_score(
        validation_residual, _bathhealth_real_vector(validation),
        validation_metric,
    )
    factor = q_validation / max(q_train, sqrt(eps(Float64)))
    zero = _bathhealth_zero_frequency(validation, validation_prediction)
    return (; q_train, q_validation, factor, zero)
end

function _bathhealth_success(candidate::BathFitHealthCandidate)
    return candidate.status === :success &&
           candidate.training_prediction !== nothing &&
           candidate.validation_prediction !== nothing
end

function _bathhealth_best_starts(candidates, metrics,
                                 tolerance::Float64)
    grouped = Dict{Int,Vector{Int}}()
    for index in eachindex(candidates)
        candidate = candidates[index]
        _bathhealth_success(candidate) || continue
        push!(get!(grouped, candidate.replica, Int[]), index)
    end
    selected = Int[]
    rashomon = Dict{Int,Vector{Int}}()
    for replica in sort!(collect(keys(grouped)))
        indices = grouped[replica]
        best = minimum(index -> metrics[index].q_train, indices)
        near = Int[index for index in indices
                   if metrics[index].q_train <= best + tolerance]
        sort!(near; by=index -> (metrics[index].q_train,
                                candidates[index].start))
        rashomon[replica] = near
        push!(selected, first(near))
    end
    return selected, rashomon
end

function _bathhealth_pair_maps(candidates, values)
    return Dict(candidates[index].replica => values[index]
                for index in eachindex(candidates))
end

function _bathhealth_paired_difference(left_candidates, left_values,
                                       right_candidates, right_values,
                                       confidence::Float64)
    left = _bathhealth_pair_maps(left_candidates, left_values)
    right = _bathhealth_pair_maps(right_candidates, right_values)
    replicas = sort!(collect(intersect(keys(left), keys(right))))
    isempty(replicas) && return Float64[], 0.0, 0.0, -Inf, Inf
    differences = Float64[left[replica] - right[replica]
                          for replica in replicas]
    mean_value = sum(differences) / length(differences)
    standard_error = if length(differences) == 1
        0.0
    else
        deviation = sqrt(sum(abs2(value - mean_value)
                             for value in differences) /
                         (length(differences) - 1))
        deviation / sqrt(length(differences))
    end
    ordered = sort(differences)
    tail = (1 - confidence) / 2
    return differences, mean_value, standard_error,
           _bathfit_health_quantile(ordered, tail),
           _bathfit_health_quantile(ordered, 1 - tail)
end

function _bathhealth_common_order_data(order_candidates, order_values, orders)
    replica_sets = [
        Set(candidate.replica for candidate in order_candidates[order])
        for order in orders
    ]
    replicas = isempty(replica_sets) ? Int[] :
               sort!(collect(reduce(intersect, replica_sets)))
    paired_candidates = Dict{Int,Vector{BathFitHealthCandidate}}()
    paired_values = Dict{Int,Vector{Float64}}()
    for order in orders
        candidate_map = Dict(
            candidate.replica => candidate
            for candidate in order_candidates[order]
        )
        value_map = _bathhealth_pair_maps(
            order_candidates[order], order_values[order],
        )
        paired_candidates[order] = BathFitHealthCandidate[
            candidate_map[replica] for replica in replicas
        ]
        paired_values[order] = Float64[
            value_map[replica] for replica in replicas
        ]
    end
    return (;
        replicas,
        candidates=paired_candidates,
        values=paired_values,
    )
end

function _bathhealth_selection_distribution(order_candidates,
                                            order_qvalidation,
                                            replicas)
    counts = Dict{Int,Int}()
    total = 0
    for replica in replicas
        available = Tuple{Int,Float64}[]
        for order in sort!(collect(keys(order_candidates)))
            candidates = order_candidates[order]
            values = order_qvalidation[order]
            for (candidate, value) in zip(candidates, values)
                candidate.replica == replica || continue
                push!(available, (order, value))
                break
            end
        end
        isempty(available) && continue
        selected = first(sort!(available; by=item -> (item[2], item[1])))[1]
        counts[selected] = get(counts, selected, 0) + 1
        total += 1
    end
    total == 0 && return Dict{Int,Float64}()
    return Dict(order => count / total for (order, count) in counts)
end

function _bathhealth_selection_interval(distribution::Dict{Int,Float64},
                                        confidence::Float64)
    isempty(distribution) && return nothing
    ordered = sort!(collect(keys(distribution)))
    cumulative = cumsum(Float64[distribution[order] for order in ordered])
    tail = (1 - confidence) / 2
    lower_index = findfirst(>=(tail), cumulative)
    upper_index = findfirst(>=(1 - tail), cumulative)
    return (ordered[something(lower_index, 1)],
            ordered[something(upper_index, length(ordered))])
end
