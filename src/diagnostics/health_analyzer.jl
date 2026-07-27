"""
    analyze_bathfit(candidates, training, validation; covariance=nothing,
                    config=BathFitDiagnosticConfig())

Analyze an already computed, crossed ensemble of bath fits without refitting.
Prediction calibration, pole-measure identifiability, and optimizer scatter are
reported as separate axes. A strict predictive-overfit verdict is emitted only
for calibrated paired replicas whose training score improves while the paired
validation difference has a strictly positive empirical confidence interval.
"""
const _BATHHEALTH_VERDICT_PRIORITY = (
    :predictive_overfit,
    :underfit,
    :spectral_nonidentifiability,
    :optimizer_illness,
    :overparameterized,
)

function _bathhealth_priority_verdict(verdicts)
    for verdict in _BATHHEALTH_VERDICT_PRIORITY
        verdict in verdicts && return verdict
    end
    return :healthy
end

function analyze_bathfit(
    candidates::AbstractVector{<:BathFitHealthCandidate},
    training::BathFitInput, validation::BathFitInput;
    covariance=nothing,
    config::BathFitDiagnosticConfig=BathFitDiagnosticConfig(),
)
    isempty(candidates) && throw(ArgumentError(
        "analyze_bathfit needs at least one candidate",
    ))
    _bathhealth_validate_domains(training, validation)
    for candidate in candidates
        if _bathhealth_success(candidate)
            _validate_residual_inputs(
                training, something(candidate.training_target),
            )
        end
        candidate.expansion === nothing && continue
        candidate.expansion.poles.layout == training.layout &&
            candidate.expansion.poles.statistics === training.statistics ||
            throw(ArgumentError(
                "bath-fit health candidates must match the training layout and statistics",
            ))
        _validate_fit_input(training, candidate.expansion.poles.partition)
        _validate_fit_input(validation, candidate.expansion.poles.partition)
    end
    training_metric = _bathhealth_metric(covariance, :training, training)
    validation_metric = _bathhealth_metric(covariance, :validation, validation)
    covariance_calibrated = covariance !== nothing &&
                            training_metric.whitening !== nothing &&
                            validation_metric.whitening !== nothing
    order_data = _bathhealth_collect_order_data(
        candidates, training, validation, covariance, config,
        training_metric, validation_metric,
    )
    orders = order_data.orders
    order_candidates = order_data.order_candidates
    order_qvalidation = order_data.order_qvalidation
    raw = order_data.raw
    paired_order_data = _bathhealth_common_order_data(
        order_candidates, order_qvalidation, orders,
    )
    available_replicas = length(paired_order_data.replicas)
    sufficient_replicas = available_replicas >= config.min_replicas
    calibrated = covariance_calibrated && sufficient_replicas
    selected_order = nothing
    one_se = Dict(order => Inf for order in orders)
    if sufficient_replicas
        best_order = first(sort!(
            copy(orders);
            by=order -> (
                sum(paired_order_data.values[order]) /
                available_replicas,
                order,
            ),
        ))
        eligible = Int[]
        for order in orders
            differences, mean_difference, standard_error, _, _ =
                _bathhealth_paired_difference(
                    paired_order_data.candidates[order],
                    paired_order_data.values[order],
                    paired_order_data.candidates[best_order],
                    paired_order_data.values[best_order],
                    config.confidence,
                )
            one_se[order] = mean_difference - standard_error
            length(differences) >= config.min_replicas &&
                mean_difference <= standard_error && push!(eligible, order)
        end
        selected_order = isempty(eligible) ? best_order : minimum(eligible)
    end
    selection_distribution = sufficient_replicas ?
        _bathhealth_selection_distribution(
            paired_order_data.candidates, paired_order_data.values,
            paired_order_data.replicas,
        ) : Dict{Int,Float64}()
    selection_interval = _bathhealth_selection_interval(
        selection_distribution, config.confidence,
    )
    selection_stability = isempty(selection_distribution) ? 0.0 :
                          maximum(values(selection_distribution))
    validation_envelope = covariance_calibrated ? 1.0 : Inf
    predictive_envelope = covariance_calibrated ? 1.0 : Inf
    measure_envelope = covariance_calibrated ? 1.0 : Inf
    order_reports = BathFitOrderHealth[]
    global_verdicts = Symbol[]
    previous_order = nothing
    for order in orders
        data = raw[order]
        order_sufficient =
            length(data.selected_candidates) >= config.min_replicas
        order_calibrated = covariance_calibrated && order_sufficient
        train_summary = _bathhealth_summary(
            data.qtrain, config.confidence,
        )
        validation_summary = _bathhealth_summary(
            data.qvalidation, config.confidence,
        )
        underfit = order_calibrated && !isempty(data.qtrain) &&
                   train_summary.lower > 1.0 &&
                   validation_summary.lower > 1.0
        predictive_overfit = false
        paired_replicas = 0
        if order_calibrated && previous_order !== nothing &&
           !isempty(data.qtrain) && !isempty(raw[previous_order].qtrain)
            differences, _, _, validation_lower, _ =
                _bathhealth_paired_difference(
                data.selected_candidates, data.qvalidation,
                raw[previous_order].selected_candidates,
                raw[previous_order].qvalidation, config.confidence,
            )
            paired_replicas = length(differences)
            predictive_overfit =
                paired_replicas >= config.min_replicas &&
                sum(data.qtrain) / length(data.qtrain) <
                sum(raw[previous_order].qtrain) /
                length(raw[previous_order].qtrain) &&
                validation_lower > 0
        end
        prediction_median = isempty(data.prediction_distances) ? Inf :
                            _bathhealth_summary(
                                data.prediction_distances, config.confidence,
                            ).median
        shape_median = isempty(data.shape_distances) ? nothing :
                       _bathhealth_summary(
                           data.shape_distances, config.confidence,
                       ).median
        nonidentifiable = order_calibrated && shape_median !== nothing &&
                          prediction_median <= 1.0 &&
                          shape_median > 1.0
        optimizer_illness = order_calibrated &&
                            data.rashomon_prediction > 1.0
        overparameterized = order_calibrated && !predictive_overfit &&
                            !isempty(data.qtrain) &&
                            _bathhealth_summary(
                                data.qtrain, config.confidence,
                            ).median < 1.0 &&
                            shape_median !== nothing && shape_median > 1.0
        order_verdicts = Symbol[]
        predictive_overfit &&
            push!(order_verdicts, :predictive_overfit)
        underfit && push!(order_verdicts, :underfit)
        nonidentifiable &&
            push!(order_verdicts, :spectral_nonidentifiability)
        optimizer_illness &&
            push!(order_verdicts, :optimizer_illness)
        overparameterized &&
            push!(order_verdicts, :overparameterized)
        overall = order_calibrated ?
                  _bathhealth_priority_verdict(order_verdicts) :
                  :uncalibrated
        verdicts = (;
            overall, underfit, predictive_overfit,
            spectral_nonidentifiability=nonidentifiable,
            optimizer_illness, overparameterized, order_sufficient,
            paired_replicas,
        )
        diagnostics = (;
            residual_diagnostics=Tuple(data.residual_profiles),
            spectral_diagnostics=Tuple(data.spectral_details),
            boundary_flux=Tuple(data.boundary_flux),
        )
        append!(global_verdicts, order_verdicts)
        warnings = String[]
        data.failed > 0 && push!(
            warnings, "$(data.failed) fit attempts failed at order $order",
        )
        !order_calibrated && push!(
            warnings,
            "order $order is uncalibrated; no strict overfit verdict was made",
        )
        !isempty(data.factors) &&
            _bathhealth_summary(
                data.factors, config.confidence,
            ).median > 10.0 &&
            push!(
                warnings,
                "order $order has native-domain generalization factor O > 10; this is a warning, not a cross-method ranking",
            )
        push!(order_reports, BathFitOrderHealth(
            order, length(data.selected_candidates), data.failed,
            _bathhealth_summary(
                [candidate.returned_poles
                 for candidate in data.selected_candidates],
                config.confidence,
            ),
            _bathhealth_summary(
                [candidate.mode_count
                 for candidate in data.selected_candidates],
                config.confidence,
            ),
            _bathhealth_summary(data.qtrain, config.confidence),
            _bathhealth_summary(data.qvalidation, config.confidence),
            _bathhealth_summary(data.factors, config.confidence),
            _bathhealth_summary(data.prediction_distances, config.confidence),
            _bathhealth_summary(data.mass_distances, config.confidence),
            isempty(data.shape_distances) ? nothing :
                _bathhealth_summary(data.shape_distances, config.confidence),
            isempty(data.zeros) ? nothing :
                _bathhealth_summary(data.zeros, config.confidence),
            isempty(data.selected_candidates) ? 0.0 :
                count(candidate -> candidate.mountable,
                      data.selected_candidates) /
                length(data.selected_candidates),
            data.rashomon_prediction, data.rashomon_measure,
            get(one_se, order, Inf), verdicts, diagnostics, warnings,
        ))
        previous_order = order
    end
    energy_window = config.energy_window
    shrinkage = covariance isa NamedTuple &&
                hasproperty(covariance, :shrinkage) ?
                Float64(getproperty(covariance, :shrinkage)) : 0.0
    thresholds = BathFitHealthThresholds(
        config.confidence, sqrt(eps(Float64)), training_metric.rank,
        training_metric.eigenvalue_floor, shrinkage, validation_envelope,
        predictive_envelope, measure_envelope,
        config.noise_scale_sensitivity, energy_window,
    )
    calibration = (;
        calibrated,
        covariance_calibrated,
        sufficient_replicas,
        training_metric=training_metric.mode,
        validation_metric=validation_metric.mode,
        covariance_rank=training_metric.rank,
        min_replicas=config.min_replicas,
        available_replicas,
    )
    report_warnings = String[]
    !covariance_calibrated && push!(
        report_warnings,
        "bath-fit health is uncalibrated because no usable covariance was supplied",
    )
    calibration.available_replicas < config.min_replicas && push!(
        report_warnings,
        "bath-fit health has insufficient replicas for the configured empirical calibration",
    )
    selection_interval !== nothing &&
        first(selection_interval) != last(selection_interval) && push!(
            report_warnings,
            "bath-order selection is unstable across replicas",
        )
    overall = calibrated ?
              _bathhealth_priority_verdict(global_verdicts) :
              :uncalibrated
    base_verdicts = (;
        overall,
        underfit=:underfit in global_verdicts,
        predictive_overfit=:predictive_overfit in global_verdicts,
        spectral_nonidentifiability=
            :spectral_nonidentifiability in global_verdicts,
        optimizer_illness=:optimizer_illness in global_verdicts,
        overparameterized=:overparameterized in global_verdicts,
    )
    noise_sensitivity = _bathhealth_noise_sensitivity(order_reports, config)
    noise_sensitivity.conclusion_changes && push!(
        report_warnings,
        "bath-fit verdict changes under the configured covariance-scale sensitivity check",
    )
    verdicts = merge(base_verdicts, (;
        noise_scale_sensitive=noise_sensitivity.conclusion_changes,
    ))
    provenance = (;
        training_source=training,
        validation_source=validation,
        seed=config.seed,
        requested_replicas=config.n_replicas,
        confidence=config.confidence,
        rashomon_tolerance=config.rashomon_tolerance,
        noise_scale_sensitivity=config.noise_scale_sensitivity,
        transport=:resolution_scaled_scalar_or_greedy_bures_lifted,
        fisher_metric=validation_metric.mode,
        selection=:paired_one_standard_error,
        selection_distribution=:replica_minimum_validation,
        generalization_warning_threshold=10.0,
        noise_sensitivity,
    )
    return BathFitHealthReport(
        training.layout, training.statistics, calibration, order_reports,
        selected_order, selection_distribution, selection_interval,
        selection_stability, thresholds, verdicts, report_warnings,
        provenance,
    )
end
