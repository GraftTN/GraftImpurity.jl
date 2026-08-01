function _bathhealth_sensitivity_baseline(order::BathFitOrderHealth)
    verdicts = order.verdicts
    return (
        underfit=getproperty(verdicts, :underfit),
        overparameterized=getproperty(verdicts, :overparameterized),
        predictive_overfit=getproperty(verdicts, :predictive_overfit),
    )
end

function _bathhealth_total_bures(reference::NamedTuple,
                                 current::NamedTuple)
    hasproperty(reference, :total_residue_matrices) &&
        hasproperty(current, :total_residue_matrices) || return nothing
    left_totals = reference.total_residue_matrices
    right_totals = current.total_residue_matrices
    keys(left_totals) == keys(right_totals) || return nothing
    weighted_distance = 0.0
    total_weight = 0.0
    for block in keys(left_totals)
        left = _bathhealth_state(
            _bathhealth_residue_matrix(getproperty(left_totals, block)),
        )
        right = _bathhealth_state(
            _bathhealth_residue_matrix(getproperty(right_totals, block)),
        )
        (left === nothing || right === nothing) && return nothing
        left_mass, left_matrix = left
        right_mass, right_matrix = right
        (iszero(left_mass) || iszero(right_mass)) && return nothing
        weight = max(left_mass, right_mass)
        weighted_distance += weight * _bathhealth_bures(
            left_matrix / left_mass, right_matrix / right_mass,
        )
        total_weight += weight
    end
    iszero(total_weight) && return nothing
    return weighted_distance / total_weight
end

function _bathhealth_scaled_shape_median(
    order::BathFitOrderHealth, scale::Float64,
    config::BathFitDiagnosticConfig,
)
    order.shape_instability === nothing && return nothing
    shape = order.shape_instability
    hasproperty(order.diagnostics, :spectral_diagnostics) ||
        return shape.median / scale
    spectral = order.diagnostics.spectral_diagnostics
    length(spectral) == length(shape.values) && !isempty(spectral) ||
        return shape.median / scale
    reference = first(spectral)
    bures = Union{Nothing,Float64}[
        _bathhealth_total_bures(reference, detail) for detail in spectral
    ]
    any(isnothing, bures) && return shape.median / scale
    scaled = Float64[]
    for (distance, invariant) in zip(shape.values, bures)
        orientation = min(distance, something(invariant))
        push!(scaled, orientation + (distance - orientation) / scale)
    end
    return _bathhealth_summary(scaled, config.confidence).median
end

function _bathhealth_noise_record(
    order::BathFitOrderHealth, scale::Float64,
    config::BathFitDiagnosticConfig,
)
    verdicts = order.verdicts
    calibrated = getproperty(verdicts, :overall) !== :uncalibrated &&
                 getproperty(verdicts, :order_sufficient)
    paired_sufficient =
        getproperty(verdicts, :paired_replicas) >= config.min_replicas
    predictive_overfit =
        calibrated && paired_sufficient &&
        getproperty(verdicts, :predictive_overfit)
    shape_median = _bathhealth_scaled_shape_median(order, scale, config)
    measure_unstable =
        calibrated && shape_median !== nothing && shape_median > 1.0
    underfit =
        calibrated && order.q_train.lower / scale > 1.0 &&
        order.q_validation.lower / scale > 1.0
    overparameterized =
        calibrated && !predictive_overfit &&
        order.q_train.median / scale < 1.0 && measure_unstable
    conclusion = (; underfit, overparameterized, predictive_overfit)
    changed = conclusion != _bathhealth_sensitivity_baseline(order)
    return (
        order=order.order,
        scale,
        q_train_median=order.q_train.median / scale,
        q_validation_median=order.q_validation.median / scale,
        predictive_distance_median=order.predictive_distance.median / scale,
        fisher_resolution_scale=scale,
        shape_instability_median=shape_median,
        underfit,
        measure_unstable,
        overparameterized,
        predictive_overfit,
        conclusion_changed=changed,
    )
end

"""
Rescore existing health summaries under covariance standard-deviation scales.
No candidate is refit; all entries are derived from the supplied order reports.
"""
function _bathhealth_noise_sensitivity(
    order_reports::Vector{BathFitOrderHealth},
    config::BathFitDiagnosticConfig,
)
    records = Tuple(
        _bathhealth_noise_record(order, scale, config)
        for scale in config.noise_scale_sensitivity
        for order in order_reports
    )
    return (
        records,
        conclusion_changes=any(record -> record.conclusion_changed, records),
        provenance=:rescore_only,
    )
end
