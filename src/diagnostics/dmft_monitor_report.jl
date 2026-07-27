function _dmft_bound_value(value)
    value isa Real && isfinite(value) && value >= 0 && return Float64(value)
    if value isa Pair
        return _dmft_bound_value(last(value))
    elseif value isa Tuple && !isempty(value)
        return _dmft_bound_value(last(value))
    elseif value isa AbstractVector
        bounds = Float64[]
        for candidate in value
            bound = _dmft_bound_value(candidate)
            bound === nothing || push!(bounds, bound)
        end
        isempty(bounds) || return maximum(bounds)
    end
    for property in (:upper, :high, :q95, :quantile_95, :maximum)
        hasproperty(value, property) || continue
        bound = _dmft_bound_value(getproperty(value, property))
        bound === nothing || return bound
    end
    return nothing
end

function _dmft_metric_names(metric::Symbol)
    metric === :data_update &&
        return (:data_update, :target_update, :input_update)
    metric === :prediction_update &&
        return (:prediction_update, :predictive_update, :fit_update)
    metric === :fit_amplification &&
        return (:fit_amplification, :fit_gain, :noise_gain)
    metric === :loop_amplification &&
        return (:loop_amplification, :loop_gain, :feedback_gain)
    metric === :measure_mass_update &&
        return (:measure_mass_update, :mass_update, :mass_distance)
    metric === :measure_shape_update &&
        return (:measure_shape_update, :shape_update, :wasserstein,
                :wasserstein_distance)
    return (metric,)
end

function _dmft_calibrated_bound(source, metric::Symbol, depth::Int=0)
    source === nothing && return nothing
    depth <= 3 || return nothing
    for name in _dmft_metric_names(metric)
        hasproperty(source, name) || continue
        bound = _dmft_bound_value(getproperty(source, name))
        bound === nothing || return bound
    end
    # Generic BathFitHealthReport predictive/measure envelopes use whitened
    # fit-analysis units. They are intentionally not raw DMFT-update bounds.
    for property in (
        :thresholds, :bootstrap_envelope, :envelope, :bootstrap,
        :calibration, :fit_envelope,
    )
        hasproperty(source, property) || continue
        bound = _dmft_calibrated_bound(
            getproperty(source, property), metric, depth + 1,
        )
        bound === nothing || return bound
    end
    return nothing
end

function _dmft_record_metric(record::DMFTBathIterationRecord, metric::Symbol)
    return getproperty(record, metric)
end

function _dmft_record_calibration_source(monitor::DMFTBathMonitor,
                                         index::Int)
    monitor.calibration_source === nothing ||
        return monitor.calibration_source
    for record_index in index:-1:firstindex(monitor.records)
        health = monitor.records[record_index].health
        health === nothing || return health
    end
    return nothing
end

function _dmft_relative_bound(values::Vector{Float64})
    finite_values = filter(isfinite, values)
    isempty(finite_values) && return nothing
    sort!(finite_values)
    center_index = length(finite_values) ÷ 2
    center = isodd(length(finite_values)) ?
             finite_values[center_index + 1] :
             (finite_values[center_index] +
              finite_values[center_index + 1]) / 2
    deviations = sort!(abs.(finite_values .- center))
    deviation_index = length(deviations) ÷ 2
    deviation = isodd(length(deviations)) ?
                deviations[deviation_index + 1] :
                (deviations[deviation_index] +
                 deviations[deviation_index + 1]) / 2
    tolerance = sqrt(eps(Float64)) * max(abs(center), 1.0)
    return center + 6 * 1.4826 * deviation + tolerance
end

function _dmft_threshold(monitor::DMFTBathMonitor, metric::Symbol,
                         record_index::Int)
    source = _dmft_record_calibration_source(monitor, record_index)
    calibrated = _dmft_calibrated_bound(source, metric)
    calibrated === nothing ||
        return calibrated, :calibrated
    first_transition = max(2, record_index - monitor.window)
    history = Float64[]
    for index in first_transition:(record_index - 1)
        value = _dmft_record_metric(monitor.records[index], metric)
        value === nothing || push!(history, value)
    end
    relative = _dmft_relative_bound(history)
    relative === nothing && return nothing, :relative
    return relative, :relative
end

function _dmft_persistent_high(monitor::DMFTBathMonitor, metric::Symbol)
    record_count = length(monitor.records)
    first_candidate = record_count - monitor.persistence + 1
    observed = -Inf
    threshold_value = -Inf
    calibrations = Symbol[]
    for index in first_candidate:record_count
        value = _dmft_record_metric(monitor.records[index], metric)
        value === nothing && return false, 0.0, 0.0, :relative
        threshold, calibration = _dmft_threshold(monitor, metric, index)
        threshold === nothing && return false, 0.0, 0.0, :relative
        value > threshold || return false, value, threshold, calibration
        observed = max(observed, value)
        threshold_value = max(threshold_value, threshold)
        push!(calibrations, calibration)
    end
    calibration = all(==(:calibrated), calibrations) ?
                  :calibrated : :relative
    return true, observed, threshold_value, calibration
end

function _dmft_persistent_nonidentifiability(monitor::DMFTBathMonitor)
    record_count = length(monitor.records)
    first_candidate = record_count - monitor.persistence + 1
    observed = -Inf
    threshold_value = -Inf
    calibrations = Symbol[]
    for index in first_candidate:record_count
        record = monitor.records[index]
        candidates = Tuple{Symbol,Float64,Float64,Symbol}[]
        for metric in (:measure_mass_update, :measure_shape_update)
            value = _dmft_record_metric(record, metric)
            value === nothing && continue
            threshold, calibration = _dmft_threshold(monitor, metric, index)
            threshold === nothing && continue
            value > threshold &&
                push!(candidates, (metric, value, threshold, calibration))
        end
        isempty(candidates) && return false, 0.0, 0.0, :relative
        prediction = record.prediction_update
        prediction === nothing && return false, 0.0, 0.0, :relative
        prediction_threshold, prediction_calibration =
            _dmft_threshold(monitor, :prediction_update, index)
        prediction_threshold === nothing &&
            return false, 0.0, 0.0, :relative
        prediction <= prediction_threshold ||
            return false, prediction, prediction_threshold,
                         prediction_calibration
        chosen = findmax(candidate -> candidate[2] / max(candidate[3], eps()),
                         candidates)[2]
        _, value, threshold, calibration = candidates[chosen]
        observed = max(observed, value)
        threshold_value = max(threshold_value, threshold)
        push!(calibrations, calibration, prediction_calibration)
    end
    calibration = all(==(:calibrated), calibrations) ?
                  :calibrated : :relative
    return true, observed, threshold_value, calibration
end

function _dmft_evidence_message(verdict::Symbol, calibration::Symbol)
    prefix = if verdict === :noise_chasing_suspected
        "fitted-prediction updates persistently amplified smaller input updates"
    elseif verdict === :feedback_amplification_suspected
        "input updates persistently amplified the preceding input update"
    else
        "positive pole-measure mass or shape changed while fitted predictions stayed stable"
    end
    calibration === :calibrated &&
        return "$prefix relative to the supplied health/bootstrap envelope"
    return "$prefix relative only to rolling median/MAD history; this is not an absolute instability claim"
end

"""
    dmft_bath_report(monitor)

Summarize the passive history. The verdict vector is empty for a mature quiet
or tracking history. Before `window` adjacent transitions exist, it contains
only `:insufficient_history`.
"""
function dmft_bath_report(monitor::DMFTBathMonitor)
    records = copy(monitor.records)
    transitions = max(length(records) - 1, 0)
    if transitions < monitor.window
        evidence = DMFTBathVerdictEvidence[
            DMFTBathVerdictEvidence(
                :insufficient_history, :relative, Float64(transitions),
                Float64(monitor.window), monitor.persistence,
                "the monitor needs $(monitor.window) adjacent transitions before interpreting persistence",
            ),
        ]
        return DMFTBathMonitorReport(
            records, transitions, monitor.window, monitor.persistence,
            :relative, [:insufficient_history], evidence,
        )
    end

    verdicts = Symbol[]
    evidence = DMFTBathVerdictEvidence[]
    checks = (
        (:noise_chasing_suspected, :fit_amplification),
        (:feedback_amplification_suspected, :loop_amplification),
    )
    for (verdict, metric) in checks
        suspected, observed, threshold, calibration =
            _dmft_persistent_high(monitor, metric)
        suspected || continue
        push!(verdicts, verdict)
        push!(evidence, DMFTBathVerdictEvidence(
            verdict, calibration, observed, threshold, monitor.persistence,
            _dmft_evidence_message(verdict, calibration),
        ))
    end
    suspected, observed, threshold, calibration =
        _dmft_persistent_nonidentifiability(monitor)
    if suspected
        verdict = :spectral_nonidentifiability
        push!(verdicts, verdict)
        push!(evidence, DMFTBathVerdictEvidence(
            verdict, calibration, observed, threshold, monitor.persistence,
            _dmft_evidence_message(verdict, calibration),
        ))
    end
    calibration = if isempty(evidence) ||
                     all(item -> item.calibration === :relative, evidence)
        :relative
    elseif all(item -> item.calibration === :calibrated, evidence)
        :calibrated
    else
        :mixed
    end
    return DMFTBathMonitorReport(
        records, transitions, monitor.window, monitor.persistence,
        calibration, verdicts, evidence,
    )
end
