@testset "bath-fit health regressions" begin
    fixture = _pure_health_fixture(1; basis=:health_regressions)
    frequencies = [-2.0, -1.0, 1.0, 2.0]
    target = _pure_health_input(
        fixture.layout, frequencies, fill(1.0 + 0im, length(frequencies)),
    )
    covariance = _pure_health_covariance(target)

    @testset "mountable candidates retain spectral evidence" begin
        @test_throws ArgumentError BathFitHealthCandidate(
            1;
            mountable=true,
            training_target=target,
            training_prediction=target,
            validation_prediction=target,
        )
    end

    @testset "all calibrated fits underfit" begin
        candidates = BathFitHealthCandidate[
            _pure_health_candidate(
                1, target, _pure_health_scalar_shift(target, 4.0);
                replica,
                training_prediction=_pure_health_scalar_shift(target, 3.0),
            )
            for replica in 1:32
        ]
        report = analyze_bathfit(
            candidates, target, target;
            covariance,
            config=BathFitDiagnosticConfig(
                n_replicas=32, min_replicas=32,
            ),
        )
        order = only(report.orders)
        @test report.calibration.calibrated
        @test order.q_train.lower > 1
        @test order.q_validation.lower > 1
        @test order.verdicts.underfit
        @test order.verdicts.overall === :underfit
        @test report.verdicts.overall === :underfit
        @test report.selected_order == 1
        @test report.selection_distribution == Dict(1 => 1.0)
    end

    @testset "strict overfit needs enough paired replicas" begin
        candidates = BathFitHealthCandidate[]
        for replica in 1:32
            push!(
                candidates,
                _pure_health_candidate(
                    1, target, _pure_health_scalar_shift(target, 0.2);
                    replica,
                    training_prediction=_pure_health_scalar_shift(target, 0.8),
                ),
            )
        end
        for replica in 32:63
            push!(
                candidates,
                _pure_health_candidate(
                    2, target, _pure_health_scalar_shift(target, 1.0);
                    replica,
                    training_prediction=_pure_health_scalar_shift(target, 0.1),
                ),
            )
        end
        report = analyze_bathfit(
            candidates, target, target;
            covariance,
            config=BathFitDiagnosticConfig(
                n_replicas=63, min_replicas=32,
            ),
        )
        order_two = _pure_health_order(report, 2)
        @test !report.calibration.calibrated
        @test report.calibration.available_replicas == 1
        @test report.selected_order === nothing
        @test isempty(report.selection_distribution)
        @test order_two.verdicts.paired_replicas == 1
        @test !order_two.verdicts.predictive_overfit
        @test report.verdicts.overall === :uncalibrated
        @test any(
            warning -> occursin("insufficient replicas", warning),
            report.warnings,
        )
        @test all(
            warning -> !occursin("no usable covariance", warning),
            report.warnings,
        )
    end

    @testset "order selection uses only common replicas" begin
        candidates = BathFitHealthCandidate[]
        for replica in 1:2
            push!(
                candidates,
                _pure_health_candidate(
                    1, target, _pure_health_scalar_shift(target, 0.2);
                    replica,
                ),
                _pure_health_candidate(
                    2, target, _pure_health_scalar_shift(target, 0.1);
                    replica,
                ),
            )
        end
        push!(
            candidates,
            _pure_health_candidate(1, target, target; replica=3),
            _pure_health_candidate(
                2, target, _pure_health_scalar_shift(target, 10.0);
                replica=4,
            ),
        )
        report = analyze_bathfit(
            candidates, target, target;
            covariance,
            config=BathFitDiagnosticConfig(
                n_replicas=4, min_replicas=2,
            ),
        )
        order_one = _pure_health_order(report, 1)
        order_two = _pure_health_order(report, 2)
        @test order_one.q_validation.mean < order_two.q_validation.mean
        @test report.calibration.calibrated
        @test report.calibration.available_replicas == 2
        @test report.selected_order == 2
        @test report.selection_distribution == Dict(2 => 1.0)
    end

    @testset "global verdict uses fixed severity priority" begin
        candidates = BathFitHealthCandidate[]
        for replica in 1:32
            push!(
                candidates,
                _pure_health_candidate(
                    1, target, _pure_health_scalar_shift(target, 4.0);
                    replica,
                    training_prediction=_pure_health_scalar_shift(target, 3.0),
                ),
                _pure_health_candidate(
                    2, target, _pure_health_scalar_shift(target, 5.0);
                    replica,
                    training_prediction=_pure_health_scalar_shift(target, 0.1),
                ),
            )
        end
        report = analyze_bathfit(
            candidates, target, target;
            covariance,
            config=BathFitDiagnosticConfig(
                n_replicas=32, min_replicas=32,
            ),
        )
        order_one = _pure_health_order(report, 1)
        order_two = _pure_health_order(report, 2)
        @test order_one.verdicts.overall === :underfit
        @test order_two.verdicts.overall === :predictive_overfit
        @test report.verdicts.underfit
        @test report.verdicts.predictive_overfit
        @test report.verdicts.overall === :predictive_overfit
    end

    @testset "nonconverged attempts are excluded" begin
        prediction = _pure_health_scalar_shift(target, 0.1)
        candidates = BathFitHealthCandidate[
            BathFitHealthCandidate(
                1;
                replica,
                status=:nonconverged,
                returned_poles=1,
                mode_count=1,
                training_target=target,
                training_prediction=prediction,
                validation_prediction=prediction,
                message="fixture did not converge",
            )
            for replica in 1:32
        ]
        report = analyze_bathfit(
            candidates, target, target;
            covariance,
            config=BathFitDiagnosticConfig(
                n_replicas=32, min_replicas=32,
            ),
        )
        order = only(report.orders)
        @test order.successful_replicas == 0
        @test order.failed_fits == 32
        @test report.calibration.available_replicas == 0
        @test !report.calibration.calibrated
        @test order.verdicts.overall === :uncalibrated
        @test report.verdicts.overall === :uncalibrated
        @test all(
            warning -> !occursin("no usable covariance", warning),
            report.warnings,
        )
        @test any(
            warning -> occursin("32 fit attempts failed", warning),
            order.warnings,
        )
    end

    @testset "expansion partition and report schema validation" begin
        unrelated = BathFitInput(
            fixture.layout, [0.25, 0.75, 1.25, 1.75],
            :block => fill(2.0 + 0im, length(frequencies));
            domain=:real_axis, statistics=:fermion,
        )
        unrelated_candidate = _pure_health_candidate(
            1, unrelated, target;
            training_prediction=unrelated,
        )
        @test_throws ArgumentError analyze_bathfit(
            [unrelated_candidate], target, target;
            covariance,
            config=BathFitDiagnosticConfig(min_replicas=1),
        )

        perturbed_target = _pure_health_scalar_shift(target, 0.1)
        schema_compatible = _pure_health_candidate(
            1, perturbed_target, target;
            training_prediction=perturbed_target,
        )
        @test analyze_bathfit(
            [schema_compatible], target, target;
            covariance,
            config=BathFitDiagnosticConfig(min_replicas=1),
        ) isa BathFitHealthReport

        mismatched = (
            layout=fixture.layout,
            partition=Partition(:different_block => [:orbital_1]),
        )
        expansion = _pure_health_expansion(mismatched, [0.2], [1.0])
        candidate = _pure_health_candidate(
            1, target, target; expansion,
        )
        @test_throws ArgumentError analyze_bathfit(
            [candidate], target, target;
            covariance,
            config=BathFitDiagnosticConfig(min_replicas=1),
        )

        valid = analyze_bathfit(
            [_pure_health_candidate(1, target, target)],
            target, target;
            covariance,
            config=BathFitDiagnosticConfig(min_replicas=1),
        )
        report_args(calibration, verdicts) = (
            valid.layout, valid.statistics, calibration, valid.orders,
            valid.selected_order, valid.selection_distribution,
            valid.selection_interval, valid.selection_stability,
            valid.thresholds, verdicts, valid.warnings, valid.provenance,
        )
        @test_throws ArgumentError BathFitHealthReport(
            report_args((; method=:synthetic), valid.verdicts)...,
        )
        @test_throws ArgumentError BathFitHealthReport(
            report_args(valid.calibration, (; underfit=false))...,
        )

        two_order = analyze_bathfit(
            [
                _pure_health_candidate(1, target, target),
                _pure_health_candidate(2, target, target),
            ],
            target, target;
            covariance,
            config=BathFitDiagnosticConfig(min_replicas=1),
        )
        selection_args(distribution, interval) = (
            two_order.layout, two_order.statistics, two_order.calibration,
            two_order.orders, two_order.selected_order, distribution, interval,
            two_order.selection_stability, two_order.thresholds,
            two_order.verdicts, two_order.warnings, two_order.provenance,
        )
        @test_throws ArgumentError BathFitHealthReport(
            selection_args(Dict(1 => 0.5, 3 => 0.5), (1, 1))...,
        )
        @test_throws ArgumentError BathFitHealthReport(
            selection_args(Dict(1 => 0.5, 2 => 0.5), (1, 3))...,
        )
        @test_throws ArgumentError BathFitHealthReport(
            selection_args(Dict(1 => 0.5, 2 => 0.5), (2, 1))...,
        )
        @test_throws ArgumentError BathFitHealthReport(
            selection_args(Dict{Int,Float64}(), (1, 1))...,
        )
        @test_throws ArgumentError BathFitHealthReport(
            selection_args(Dict(1 => 1.0), nothing)...,
        )
        @test_throws ArgumentError BathFitHealthReport(
            selection_args(Dict(1 => 1.0), (1, 2))...,
        )
    end

    @testset "large generalization factor is warning-only" begin
        candidates = BathFitHealthCandidate[
            _pure_health_candidate(
                1, target, _pure_health_scalar_shift(target, 0.2);
                replica,
                training_prediction=_pure_health_scalar_shift(target, 0.01),
            )
            for replica in 1:32
        ]
        report = analyze_bathfit(
            candidates, target, target;
            covariance,
            config=BathFitDiagnosticConfig(
                n_replicas=32, min_replicas=32,
            ),
        )
        order = only(report.orders)
        @test order.generalization_factor.median > 10
        @test order.verdicts.overall === :healthy
        @test report.verdicts.overall === :healthy
        @test any(warning -> occursin("O > 10", warning), order.warnings)
    end
end
