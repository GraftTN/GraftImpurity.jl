@testset "pure bath-fit health calibration verdicts" begin
    fixture = _pure_health_fixture(1; basis=:health_calibration)
    frequencies = [-3.0, -1.0, 1.0, 3.0]
    training = _pure_health_input(
        fixture.layout, frequencies, fill(1.0 + 0im, length(frequencies)),
    )
    validation = _pure_health_input(
        fixture.layout, frequencies, fill(1.0 + 0im, length(frequencies)),
    )
    candidates = BathFitHealthCandidate[]
    for replica in 1:8
        push!(
            candidates,
            _pure_health_candidate(
                1, training,
                _pure_health_scalar_shift(validation, 0.20 + 0.005replica);
                replica,
                training_prediction=_pure_health_scalar_shift(
                    training, 0.80 - 0.01replica,
                ),
            ),
            _pure_health_candidate(
                2, training,
                _pure_health_scalar_shift(validation, 1.00 + 0.01replica);
                replica,
                training_prediction=_pure_health_scalar_shift(
                    training, 0.10 + 0.002replica,
                ),
            ),
        )
    end
    config = BathFitDiagnosticConfig(
        n_replicas=8, min_replicas=4, confidence=0.95,
    )

    uncalibrated = analyze_bathfit(
        candidates, training, validation;
        config,
    )
    @test !uncalibrated.calibration.calibrated
    @test uncalibrated.verdicts.overall === :uncalibrated
    @test !uncalibrated.verdicts.predictive_overfit
    @test all(
        order -> !order.verdicts.predictive_overfit,
        uncalibrated.orders,
    )
    @test any(
        warning -> occursin("no strict overfit verdict", warning),
        Iterators.flatten(order.warnings for order in uncalibrated.orders),
    )

    calibrated = analyze_bathfit(
        candidates, training, validation;
        covariance=_pure_health_covariance(training),
        config,
    )
    order_one = _pure_health_order(calibrated, 1)
    order_two = _pure_health_order(calibrated, 2)
    @test calibrated.calibration.calibrated
    @test order_two.q_train.mean < order_one.q_train.mean
    @test minimum(
        order_two.q_validation.values .- order_one.q_validation.values,
    ) > 0
    @test order_two.verdicts.predictive_overfit
    @test order_two.verdicts.overall === :predictive_overfit
    @test calibrated.verdicts.predictive_overfit
end

@testset "pure bath-fit health underfit" begin
    fixture = _pure_health_fixture(1; basis=:health_underfit)
    frequencies = [-2.0, -1.0, 1.0, 2.0]
    target = _pure_health_input(
        fixture.layout, frequencies, fill(1.0 + 0im, length(frequencies)),
    )
    candidates = BathFitHealthCandidate[]
    for replica in 1:2
        push!(
            candidates,
            _pure_health_candidate(
                1, target, _pure_health_scalar_shift(target, 4.0);
                replica,
                training_prediction=_pure_health_scalar_shift(target, 3.0),
            ),
        )
    end
    for replica in 1:40
        push!(
            candidates,
            _pure_health_candidate(
                2, target, _pure_health_scalar_shift(target, 0.1);
                replica,
                training_prediction=_pure_health_scalar_shift(target, 0.1),
            ),
        )
    end

    report = analyze_bathfit(
        candidates, target, target;
        covariance=_pure_health_covariance(target),
        config=BathFitDiagnosticConfig(
            n_replicas=40, min_replicas=1, confidence=0.95,
        ),
    )
    underfit = _pure_health_order(report, 1)
    @test underfit.q_train.lower > 1
    @test underfit.q_validation.lower > 1
    @test underfit.verdicts.underfit
    @test underfit.verdicts.overall === :underfit
    @test report.verdicts.underfit
end

@testset "pure bath-fit health Rashomon optimizer and measure axes" begin
    fixture = _pure_health_fixture(1; basis=:health_rashomon)
    frequencies = [-2.0, -1.0, 1.0, 2.0]
    target = _pure_health_input(
        fixture.layout, frequencies, fill(1.0 + 0im, length(frequencies)),
    )
    left = _pure_health_expansion(fixture, [-1.0], [1.0])
    right = _pure_health_expansion(fixture, [1.0], [1.0])
    candidates = BathFitHealthCandidate[
        _pure_health_candidate(
            1, target, target;
            start=1, expansion=left,
        ),
        _pure_health_candidate(
            1, target, _pure_health_scalar_shift(target, 3.0);
            start=2, expansion=right,
        ),
    ]
    report = analyze_bathfit(
        candidates, target, target;
        covariance=_pure_health_covariance(target),
        config=BathFitDiagnosticConfig(
            n_replicas=1,
            min_replicas=1,
            starts=2,
            rashomon_tolerance=0.0,
        ),
    )
    order = only(report.orders)
    @test order.rashomon_predictive_diameter > 1
    @test order.rashomon_measure_diameter !== nothing
    @test isfinite(something(order.rashomon_measure_diameter))
    @test something(order.rashomon_measure_diameter) > 0
    @test order.verdicts.optimizer_illness
    @test order.verdicts.overall === :optimizer_illness
end

@testset "pure bath-fit health selection is deterministic" begin
    fixture = _pure_health_fixture(1; basis=:health_selection)
    frequencies = [-1.0, 1.0]
    target = _pure_health_input(
        fixture.layout, frequencies, fill(1.0 + 0im, length(frequencies)),
    )
    first_errors = (0.1, 0.2, 0.8, 0.9)
    second_errors = (0.9, 0.8, 0.2, 0.1)
    candidates = BathFitHealthCandidate[]
    for replica in 1:4
        push!(
            candidates,
            _pure_health_candidate(
                2, target,
                _pure_health_scalar_shift(target, second_errors[replica]);
                replica,
            ),
            _pure_health_candidate(
                1, target,
                _pure_health_scalar_shift(target, first_errors[replica]);
                replica,
            ),
        )
    end
    config = BathFitDiagnosticConfig(
        n_replicas=4, min_replicas=1, seed=UInt64(0x1234),
    )
    first_report = analyze_bathfit(
        candidates, target, target;
        covariance=_pure_health_covariance(target),
        config,
    )
    reversed_report = analyze_bathfit(
        reverse(candidates), target, target;
        covariance=_pure_health_covariance(target),
        config,
    )
    @test first_report.selection_distribution == Dict(1 => 0.5, 2 => 0.5)
    @test reversed_report.selection_distribution ==
          first_report.selection_distribution
    @test reversed_report.selected_order == first_report.selected_order
    @test reversed_report.selection_interval ==
          first_report.selection_interval == (1, 2)
    @test reversed_report.selection_stability ==
          first_report.selection_stability == 0.5
end
