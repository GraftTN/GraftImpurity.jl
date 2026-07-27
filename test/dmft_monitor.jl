using Test
using GraftImpurity

function _dmft_monitor_fixture()
    layout = FlavorLayout(
        [:imp], Dict(:imp => :impurity), Dict(:impurity => [:imp]);
        basis=:dmft_monitor,
    )
    partition = Partition(:bath => [:imp])
    return (; layout, partition, frequencies=Float64[1, 3, 5])
end

function _dmft_monitor_input(fixture, value::Real)
    samples = ComplexF64[
        complex(value * (1 + 0.1index), -0.05value * index)
        for index in eachindex(fixture.frequencies)
    ]
    return BathFitInput(
        fixture.layout, fixture.frequencies, :bath => samples;
        domain=:matsubara, statistics=:fermion,
        metadata=(; fixture=:dmft_monitor),
    )
end

function _dmft_monitor_result(fixture, prediction::Real;
                              energy::Real=0.0, mass::Real=1.0,
                              health=nothing)
    fitted = _dmft_monitor_input(fixture, prediction)
    poles = BlockRealPoles(
        fixture.layout, fixture.partition, Float64[energy],
        ComplexF64[ComplexF64(mass)], [1]; statistics=:fermion,
    )
    expansion = PoleExpansion(
        poles; kernel=:dmft_monitor_synthetic, trace=(;),
    )
    return (; prediction=fitted, expansion, health)
end

function _dmft_monitor_append!(monitor, fixture, iteration::Int,
                               target::Real, prediction::Real;
                               energy::Real=0.0, mass::Real=1.0,
                               health=nothing)
    input = _dmft_monitor_input(fixture, target)
    result = _dmft_monitor_result(
        fixture, prediction; energy, mass, health,
    )
    return update!(monitor, iteration, input, result)
end

function _dmft_monitor_health(fixture)
    summary(value) = BathFitMetricSummary([Float64(value)]; confidence=0.95)
    q_train = summary(0.95)
    q_validation = summary(1.05)
    mode_counts = summary(5)
    returned_poles = summary(4)
    order_health = BathFitOrderHealth(
        4, 32, 0, returned_poles, mode_counts, q_train, q_validation,
        summary(1.1), summary(0.2), summary(0.1), summary(0.1), nothing,
        1.0, 0.2, 0.1, 0.0, (; overall=:healthy), (;), String[],
    )
    thresholds = BathFitHealthThresholds(
        0.95, 1e-12, 6, 1e-12, 0.0, 2.0, 2.0, 2.0,
        (0.5, 1.0, 2.0), (-2.0, 2.0),
    )
    health = BathFitHealthReport(
        fixture.layout, :fermion,
        (; calibrated=true, kind=:bootstrap), [order_health], 4,
        Dict(4 => 1.0), (4, 4), 1.0, thresholds,
        (; overall=:healthy), String[],
        (; fixture=:dmft_monitor),
    )
    return (; health, q_train, q_validation, mode_counts, returned_poles)
end

@testset "passive DMFT bath monitor insufficient history and validation" begin
    fixture = _dmft_monitor_fixture()
    monitor = DMFTBathMonitor()
    first_record = _dmft_monitor_append!(monitor, fixture, 0, 1.0, 1.0)
    @test first_record isa DMFTBathIterationRecord
    @test first_record.data_update === nothing
    report = dmft_bath_report(monitor)
    @test report isa DMFTBathMonitorReport
    @test report.verdicts == [:insufficient_history]
    @test report.transitions == 0
    @test report.calibration === :relative

    second_record = _dmft_monitor_append!(monitor, fixture, 1, 1.1, 1.1)
    @test second_record.u_k isa BathFitInput
    @test second_record.v_k isa BathFitInput
    @test second_record.u_k.domain === :matsubara
    for sample in second_record.u_k.blocks.bath
        @test !iszero(sum(abs2, sample))
    end
    @test_throws ArgumentError _dmft_monitor_append!(
        monitor, fixture, 1, 1.2, 1.2,
    )
    wrong_grid = BathFitInput(
        fixture.layout, Float64[1, 2], :bath => ComplexF64[1, 1];
        domain=:matsubara, statistics=:fermion,
    )
    @test_throws ArgumentError update!(
        monitor, 2, wrong_grid, _dmft_monitor_result(fixture, 1.2),
    )

    foreign_layout = FlavorLayout(
        [:orb], Dict(:orb => :orbital), Dict(:orbital => [:orb]);
        basis=:dmft_monitor_foreign,
    )
    foreign_partition = Partition(:bath => [:orb])
    foreign_poles = BlockRealPoles(
        foreign_layout, foreign_partition, Float64[0],
        ComplexF64[1], [1]; statistics=:fermion,
    )
    foreign_result = (;
        prediction=_dmft_monitor_input(fixture, 1.2),
        expansion=PoleExpansion(
            foreign_poles; kernel=:dmft_monitor_foreign, trace=(;),
        ),
    )
    records_before = length(monitor.records)
    @test_throws ArgumentError update!(
        monitor, 2, _dmft_monitor_input(fixture, 1.2), foreign_result,
    )
    @test length(monitor.records) == records_before

    bosonic_poles = BlockRealPoles(
        fixture.layout, fixture.partition, Float64[0],
        ComplexF64[1], [1]; statistics=:boson,
    )
    bosonic_result = (;
        prediction=_dmft_monitor_input(fixture, 1.2),
        expansion=PoleExpansion(
            bosonic_poles; kernel=:dmft_monitor_bosonic, trace=(;),
        ),
    )
    @test_throws ArgumentError update!(
        monitor, 2, _dmft_monitor_input(fixture, 1.2), bosonic_result,
    )
    @test length(monitor.records) == records_before

    wide_layout = FlavorLayout(
        [:imp, :aux],
        Dict(:imp => :impurity, :aux => :auxiliary),
        Dict(:impurity => [:imp], :auxiliary => [:aux]);
        basis=:dmft_monitor_wide,
    )
    wide_fixture = (;
        layout=wide_layout,
        partition=Partition(:bath => [:imp]),
        frequencies=fixture.frequencies,
    )
    incompatible_partition = Partition(:bath => [:imp, :aux])
    incompatible_poles = BlockRealPoles(
        wide_layout, incompatible_partition, Float64[0],
        [ComplexF64[1 0; 0 1]], [1]; statistics=:fermion,
    )
    incompatible_result = (;
        prediction=_dmft_monitor_input(wide_fixture, 1.2),
        expansion=PoleExpansion(
            incompatible_poles;
            kernel=:dmft_monitor_incompatible_partition,
            trace=(;),
        ),
    )
    @test_throws DimensionMismatch update!(
        monitor, 2, _dmft_monitor_input(wide_fixture, 1.2),
        incompatible_result,
    )
    @test length(monitor.records) == records_before
end

@testset "passive DMFT bath monitor quiet and tracking histories" begin
    fixture = _dmft_monitor_fixture()
    quiet = DMFTBathMonitor()
    for iteration in 0:7
        record = _dmft_monitor_append!(
            quiet, fixture, iteration, 1.0, 1.0; energy=0.25,
        )
        @test record.health === nothing
    end
    quiet_report = dmft_bath_report(quiet)
    @test isempty(quiet_report.verdicts)
    @test quiet_report.calibration === :relative
    @test all(record -> record.data_update === nothing ||
                        record.data_update == 0,
              quiet_report.records)

    tracking = DMFTBathMonitor()
    for iteration in 0:8
        value = 1.0 + iteration
        _dmft_monitor_append!(
            tracking, fixture, iteration, value, value;
            energy=-0.5 + 0.05iteration,
        )
    end
    tracking_report = dmft_bath_report(tracking)
    @test isempty(tracking_report.verdicts)
    @test all(record -> record.fit_amplification === nothing ||
                        record.fit_amplification ≈ 1,
              tracking_report.records)
    @test all(record -> record.measure_shape_update === nothing ||
                        record.measure_shape_update ≈ 0.05,
              tracking_report.records[2:end])
end

@testset "passive DMFT bath monitor detects persistent noise chasing" begin
    fixture = _dmft_monitor_fixture()
    envelope = (;
        fit_amplification=(; upper=2.0),
        loop_amplification=(; upper=50.0),
        prediction_update=(; upper=2.0),
        measure_mass_update=(; upper=2.0),
        measure_shape_update=(; upper=2.0),
    )
    monitor = DMFTBathMonitor(; bootstrap_envelope=envelope)
    health_fixture = _dmft_monitor_health(fixture)
    health = health_fixture.health
    target = 1.0
    prediction = 1.0
    _dmft_monitor_append!(
        monitor, fixture, 0, target, prediction; health,
    )
    for iteration in 1:2
        target += 1.0
        prediction += 1.0
        _dmft_monitor_append!(
            monitor, fixture, iteration, target, prediction; health,
        )
    end
    for iteration in 3:5
        target += 0.01
        prediction += 0.2
        _dmft_monitor_append!(
            monitor, fixture, iteration, target, prediction; health,
        )
    end
    report = dmft_bath_report(monitor)
    @test :noise_chasing_suspected in report.verdicts
    @test !(:feedback_amplification_suspected in report.verdicts)
    evidence = only(filter(
        item -> item.verdict === :noise_chasing_suspected,
        report.evidence,
    ))
    @test evidence.calibration === :calibrated
    @test report.calibration === :calibrated
    complexity = last(report.records).complexity.bath
    @test complexity.order == 4
    @test complexity.q_train === health_fixture.q_train
    @test complexity.q_validation === health_fixture.q_validation
    @test complexity.returned_poles === health_fixture.returned_poles
    @test complexity.mode_counts === health_fixture.mode_counts
end

@testset "passive DMFT bath monitor detects feedback amplification" begin
    fixture = _dmft_monitor_fixture()
    formula_monitor = DMFTBathMonitor(; window=2, persistence=1)
    _dmft_monitor_append!(formula_monitor, fixture, 0, 0.0, 0.0)
    _dmft_monitor_append!(formula_monitor, fixture, 1, 1.0, 100.0)
    formula_record = _dmft_monitor_append!(
        formula_monitor, fixture, 2, 3.0, 101.0,
    )
    # The loop factor is ||u_k|| / ||u_{k-1}|| = 2, independent of the
    # deliberately different preceding fitted-prediction update.
    @test formula_record.loop_amplification ≈ 2
    @test formula_record.blocks.bath.loop_amplification ≈ 2
    @test !(formula_record.loop_amplification ≈ 0.02)

    monitor = DMFTBathMonitor()
    target = 0.0
    prediction = 0.0
    _dmft_monitor_append!(monitor, fixture, 0, target, prediction)
    increments = Float64[1, 1, 1, 1, 1, 10, 100, 1000]
    for (iteration, increment) in enumerate(increments)
        target += increment
        prediction += increment
        _dmft_monitor_append!(
            monitor, fixture, iteration, target, prediction,
        )
    end
    report = dmft_bath_report(monitor)
    @test :feedback_amplification_suspected in report.verdicts
    @test !(:noise_chasing_suspected in report.verdicts)
    @test all(record -> record.fit_amplification === nothing ||
                        record.fit_amplification ≈ 1,
              report.records)
end

@testset "passive DMFT bath monitor separates spectral nonidentifiability" begin
    fixture = _dmft_monitor_fixture()
    monitor = DMFTBathMonitor()
    target = 1.0
    prediction = 1.0
    energy = 0.0
    _dmft_monitor_append!(
        monitor, fixture, 0, target, prediction; energy,
    )
    shifts = Float64[0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1]
    for (iteration, shift) in enumerate(shifts)
        target += 0.1
        prediction += 0.1
        energy += shift
        _dmft_monitor_append!(
            monitor, fixture, iteration, target, prediction; energy,
        )
    end
    report = dmft_bath_report(monitor)
    @test :spectral_nonidentifiability in report.verdicts
    @test !(:noise_chasing_suspected in report.verdicts)
    evidence = only(filter(
        item -> item.verdict === :spectral_nonidentifiability,
        report.evidence,
    ))
    @test evidence.calibration === :relative
    @test occursin("not an absolute", evidence.message)
end

@testset "generic health envelopes do not calibrate raw DMFT updates" begin
    fixture = _dmft_monitor_fixture()
    health = _dmft_monitor_health(fixture).health
    monitor = DMFTBathMonitor(health)
    target = 1.0
    prediction = 1.0
    energy = 0.0
    _dmft_monitor_append!(
        monitor, fixture, 0, target, prediction; energy, health,
    )
    shifts = Float64[0.01, 0.01, 0.01, 0.01, 0.01, 10, 10, 10]
    for (iteration, shift) in enumerate(shifts)
        target += 0.1
        prediction += 0.1
        energy += shift
        _dmft_monitor_append!(
            monitor, fixture, iteration, target, prediction; energy, health,
        )
    end
    report = dmft_bath_report(monitor)
    evidence = only(filter(
        item -> item.verdict === :spectral_nonidentifiability,
        report.evidence,
    ))
    @test :spectral_nonidentifiability in report.verdicts
    @test evidence.calibration === :relative
    @test report.calibration === :relative
    @test evidence.threshold < health.thresholds.measure_envelope
end
