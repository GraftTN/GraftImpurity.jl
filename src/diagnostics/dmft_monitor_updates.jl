"""
    update!(monitor, iteration, target_input::BathFitInput, fit_result)

Append one passive observation. `fit_result` may be a `DiscretizationResult`, a
`BathFitReport`, or another typed result exposing `report`/`prediction`,
`expansion`, and optional `health` properties. The function never calls a
fitter, solver, mixer, or self-consistency update.
"""
function update!(monitor::DMFTBathMonitor, iteration::Integer,
                 target_input::BathFitInput, fit_result)
    iteration_value = Int(iteration)
    iteration_value >= 0 ||
        throw(ArgumentError("DMFT bath iteration must be nonnegative"))
    isempty(monitor.records) ||
        iteration_value > last(monitor.records).iteration ||
        throw(ArgumentError(
            "DMFT bath iterations must be appended in strictly increasing order",
        ))

    report = _dmft_fit_report(fit_result)
    prediction = _dmft_prediction(fit_result, report)
    prediction === nothing && throw(ArgumentError(
        "DMFT bath monitoring needs a fitted BathFitInput prediction",
    ))
    _dmft_input_structure_matches(target_input, prediction) ||
        throw(ArgumentError(
            "DMFT bath target and fitted prediction must share layout, domain, grid, and block shapes",
        ))
    if report !== nothing
        _dmft_same_source(target_input, report.source) || throw(ArgumentError(
            "DMFT bath fit report source does not match target_input",
        ))
    end

    names = Tuple(keys(target_input.blocks))
    health = _dmft_health(fit_result, report)
    _dmft_validate_health_source(health, target_input)
    _dmft_validate_health_source(monitor.calibration_source, target_input)
    expansion = _dmft_expansion(fit_result)
    _dmft_validate_expansion_source(expansion, target_input)
    measure = _dmft_measure_snapshot(expansion, names)
    previous_record = isempty(monitor.records) ? nothing : last(monitor.records)
    previous_measure = isempty(monitor.measures) ? nothing : last(monitor.measures)
    previous_record === nothing ||
        _dmft_input_structure_matches(target_input, previous_record.target) ||
        throw(ArgumentError(
            "DMFT bath history must preserve layout, domain, grid, block order, and sample shapes",
        ))

    block_records = Tuple(begin
        current_target = getproperty(target_input.blocks, block)
        current_prediction = getproperty(prediction.blocks, block)
        if previous_record === nothing
            DMFTBathBlockRecord(
                block, nothing, nothing, nothing, nothing, nothing, nothing,
            )
        else
            previous_target = getproperty(previous_record.target.blocks, block)
            previous_prediction =
                getproperty(previous_record.prediction.blocks, block)
            target_difference =
                _dmft_block_difference(current_target, previous_target)
            prediction_difference =
                _dmft_block_difference(current_prediction, previous_prediction)
            target_scale = max(
                _dmft_block_norm(current_target),
                _dmft_block_norm(previous_target),
            )
            data_update = iszero(target_scale) ?
                          (iszero(target_difference) ? 0.0 : Inf) :
                          target_difference / target_scale
            prediction_update = iszero(target_scale) ?
                                (iszero(prediction_difference) ? 0.0 : Inf) :
                                prediction_difference / target_scale
            fit_amplification =
                _dmft_ratio(prediction_difference, target_difference)
            loop_amplification = if length(monitor.records) < 2
                nothing
            else
                preceding_target = getproperty(
                    monitor.records[end - 1].target.blocks, block,
                )
                prior_target_difference = _dmft_block_difference(
                    previous_target, preceding_target,
                )
                _dmft_ratio(target_difference, prior_target_difference)
            end
            current_block_measure = measure === nothing ? nothing :
                                    getproperty(measure.blocks, block)
            previous_block_measure =
                previous_measure === nothing ? nothing :
                getproperty(previous_measure.blocks, block)
            mass_update, shape_update = _dmft_measure_updates(
                current_block_measure, previous_block_measure,
            )
            DMFTBathBlockRecord(
                block, data_update, prediction_update, fit_amplification,
                loop_amplification, mass_update, shape_update,
            )
        end
    end for block in names)
    blocks = NamedTuple{names}(block_records)
    complexity = NamedTuple{names}(Tuple(
        _dmft_complexity(report, health, block) for block in names
    ))

    if previous_record === nothing
        aggregate = (nothing, nothing, nothing, nothing, nothing, nothing)
        u_k = nothing
        v_k = nothing
    else
        u_k = residual_hybridization(target_input, previous_record.target)
        v_k = residual_hybridization(prediction, previous_record.prediction)
        current_target_norm = sqrt(sum(
            _dmft_block_norm(getproperty(target_input.blocks, block))^2
            for block in names
        ))
        previous_target_norm = sqrt(sum(
            _dmft_block_norm(getproperty(previous_record.target.blocks, block))^2
            for block in names
        ))
        target_difference = sqrt(sum(
            _dmft_block_difference(
                getproperty(target_input.blocks, block),
                getproperty(previous_record.target.blocks, block),
            )^2 for block in names
        ))
        prediction_difference = sqrt(sum(
            _dmft_block_difference(
                getproperty(prediction.blocks, block),
                getproperty(previous_record.prediction.blocks, block),
            )^2 for block in names
        ))
        target_scale = max(current_target_norm, previous_target_norm)
        data_update = iszero(target_scale) ?
                      (iszero(target_difference) ? 0.0 : Inf) :
                      target_difference / target_scale
        prediction_update = iszero(target_scale) ?
                            (iszero(prediction_difference) ? 0.0 : Inf) :
                            prediction_difference / target_scale
        fit_amplification =
            _dmft_ratio(prediction_difference, target_difference)
        loop_amplification = if length(monitor.records) < 2
            nothing
        else
            preceding_record = monitor.records[end - 1]
            prior_target_difference = sqrt(sum(
                _dmft_block_difference(
                    getproperty(previous_record.target.blocks, block),
                    getproperty(preceding_record.target.blocks, block),
                )^2 for block in names
            ))
            _dmft_ratio(target_difference, prior_target_difference)
        end
        mass_update = _dmft_maximum_optional(
            record.measure_mass_update for record in block_records
        )
        shape_update = _dmft_maximum_optional(
            record.measure_shape_update for record in block_records
        )
        aggregate = (
            data_update, prediction_update, fit_amplification,
            loop_amplification, mass_update, shape_update,
        )
    end

    record = DMFTBathIterationRecord(
        iteration_value, target_input, prediction, blocks, complexity, health,
        u_k, v_k, aggregate...,
    )
    push!(monitor.records, record)
    push!(monitor.measures, measure)
    return record
end
