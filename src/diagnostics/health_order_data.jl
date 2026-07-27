function _bathhealth_collect_order_data(
    candidates::AbstractVector{<:BathFitHealthCandidate},
    training::BathFitInput,
    validation::BathFitInput,
    covariance,
    config::BathFitDiagnosticConfig,
    training_metric::_BathHealthMetric,
    validation_metric::_BathHealthMetric,
)
    metric_cache = Dict{Int,NamedTuple}()
    for index in eachindex(candidates)
        _bathhealth_success(candidates[index]) || continue
        metric_cache[index] = _bathhealth_candidate_metrics(
            candidates[index], training, validation,
            training_metric, validation_metric,
        )
    end
    orders = sort!(unique(candidate.order for candidate in candidates))
    order_candidates = Dict{Int,Vector{BathFitHealthCandidate}}()
    order_qtrain = Dict{Int,Vector{Float64}}()
    order_qvalidation = Dict{Int,Vector{Float64}}()
    raw = Dict{Int,NamedTuple}()
    for order in orders
        indices = findall(candidate -> candidate.order == order, candidates)
        local_candidates = candidates[indices]
        local_metrics = Dict(index => metric_cache[indices[index]]
                             for index in eachindex(indices)
                             if haskey(metric_cache, indices[index]))
        aligned_metrics = Vector{Union{Nothing,NamedTuple}}(
            undef, length(local_candidates),
        )
        for index in eachindex(local_candidates)
            aligned_metrics[index] = get(local_metrics, index, nothing)
        end
        valid_metrics = Dict(index => something(aligned_metrics[index])
                             for index in eachindex(local_candidates)
                             if aligned_metrics[index] !== nothing)
        selected_indices, rashomon = _bathhealth_best_starts(
            local_candidates, valid_metrics, config.rashomon_tolerance,
        )
        selected_candidates = local_candidates[selected_indices]
        selected_metrics = NamedTuple[
            something(aligned_metrics[index]) for index in selected_indices
        ]
        qtrain = Float64[item.q_train for item in selected_metrics]
        qvalidation = Float64[item.q_validation for item in selected_metrics]
        factors = Float64[item.factor for item in selected_metrics]
        zeros = Float64[item.zero for item in selected_metrics
                        if item.zero !== nothing]
        residual_profiles = NamedTuple[]
        spectral_details = NamedTuple[]
        for candidate in selected_candidates
            training_target = something(candidate.training_target, training)
            push!(residual_profiles, (;
                replica=candidate.replica,
                training=_bathhealth_residual_profile(
                    training_target,
                    something(candidate.training_prediction),
                    covariance, :training,
                ),
                validation=_bathhealth_residual_profile(
                    validation,
                    something(candidate.validation_prediction),
                    covariance, :validation,
                ),
            ))
            candidate.expansion === nothing || push!(
                spectral_details,
                merge(
                    (; replica=candidate.replica),
                    _bathhealth_spectral_details(
                        candidate.expansion, config.energy_window,
                        validation, validation_metric,
                    ),
                ),
            )
        end
        prediction_distances = Float64[]
        mass_distances = Float64[]
        shape_distances = Float64[]
        boundary_flux = NamedTuple[]
        if !isempty(selected_candidates)
            reference = first(selected_candidates)
            for candidate in selected_candidates
                push!(prediction_distances, _bathhealth_distance(
                    something(reference.validation_prediction),
                    something(candidate.validation_prediction),
                    validation_metric,
                ))
                if reference.expansion !== nothing &&
                   candidate.expansion !== nothing
                    mass, shape = _bathhealth_measure_distance(
                        reference.expansion, candidate.expansion,
                        validation, validation_metric,
                    )
                    mass === nothing || push!(mass_distances, mass)
                    shape === nothing || push!(shape_distances, shape)
                    push!(boundary_flux, merge(
                        (; replica=candidate.replica),
                        _bathhealth_boundary_flux(
                            reference.expansion, candidate.expansion,
                            config.energy_window,
                        ),
                    ))
                end
            end
        end
        rashomon_prediction = 0.0
        rashomon_measure = Float64[]
        for near in values(rashomon)
            for first_index in near, second_index in near
                first_index < second_index || continue
                first_candidate = local_candidates[first_index]
                second_candidate = local_candidates[second_index]
                rashomon_prediction = max(
                    rashomon_prediction,
                    _bathhealth_distance(
                        something(first_candidate.validation_prediction),
                        something(second_candidate.validation_prediction),
                        validation_metric,
                    ),
                )
                if first_candidate.expansion !== nothing &&
                   second_candidate.expansion !== nothing
                    _, shape = _bathhealth_measure_distance(
                        first_candidate.expansion, second_candidate.expansion,
                        validation, validation_metric,
                    )
                    shape === nothing || push!(rashomon_measure, shape)
                end
            end
        end
        order_candidates[order] = selected_candidates
        order_qtrain[order] = qtrain
        order_qvalidation[order] = qvalidation
        raw[order] = (;
            selected_candidates, qtrain, qvalidation, factors, zeros,
            residual_profiles, spectral_details, boundary_flux,
            prediction_distances, mass_distances, shape_distances,
            rashomon_prediction,
            rashomon_measure=isempty(rashomon_measure) ? nothing :
                              maximum(rashomon_measure),
            failed=count(candidate -> candidate.status !== :success,
                         local_candidates),
        )
    end
    return (; orders, order_candidates, order_qvalidation, raw)
end
