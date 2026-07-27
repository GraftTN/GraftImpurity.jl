function _pure_sensitivity_order(
    order::Integer;
    q_train::Real,
    q_validation::Real,
    predictive_distance::Real,
    shape_instability::Union{Nothing,Real},
    overall::Symbol=:healthy,
    underfit::Bool=false,
    overparameterized::Bool=false,
    predictive_overfit::Bool=false,
    order_sufficient::Bool=true,
    paired_replicas::Integer=4,
)
    summary(value) = BathFitMetricSummary(fill(Float64(value), 4))
    shape = shape_instability === nothing ? nothing :
            summary(shape_instability)
    verdicts = (;
        overall,
        underfit,
        predictive_overfit,
        spectral_nonidentifiability=false,
        optimizer_illness=false,
        overparameterized,
        order_sufficient,
        paired_replicas=Int(paired_replicas),
    )
    return BathFitOrderHealth(
        Int(order), 4, 0, summary(order), summary(order),
        summary(q_train), summary(q_validation),
        summary(q_validation / q_train), summary(predictive_distance),
        summary(0.0), shape, nothing, 1.0, 0.0, nothing, 0.0,
        verdicts, (;), String[],
    )
end

function _pure_noise_geometry_report(
    fixture, target, expansions, covariance_scale::Real,
)
    candidates = BathFitHealthCandidate[
        _pure_health_candidate(
            1, target, target;
            replica=index, expansion,
        )
        for (index, expansion) in enumerate(expansions)
    ]
    return analyze_bathfit(
        candidates, target, target;
        covariance=covariance_scale .* _pure_health_covariance(target),
        config=BathFitDiagnosticConfig(
            n_replicas=length(candidates),
            min_replicas=1,
            noise_scale_sensitivity=(1.0, 2.0),
        ),
    )
end

@testset "bath-fit noise sensitivity rescoring" begin
    orders = BathFitOrderHealth[
        _pure_sensitivity_order(
            1;
            q_train=1.5,
            q_validation=1.8,
            predictive_distance=1.2,
            shape_instability=1.4,
            overall=:underfit,
            underfit=true,
        ),
        _pure_sensitivity_order(
            2;
            q_train=0.75,
            q_validation=0.9,
            predictive_distance=0.6,
            shape_instability=1.5,
            overall=:overparameterized,
            overparameterized=true,
        ),
    ]
    config = BathFitDiagnosticConfig(
        min_replicas=4,
        noise_scale_sensitivity=(0.5, 1.0, 2.0),
    )
    sensitivity =
        GraftImpurity._bathhealth_noise_sensitivity(orders, config)

    @test sensitivity.provenance === :rescore_only
    @test length(sensitivity.records) == 6
    @test sensitivity.conclusion_changes
    high_noise_order_one = only(
        record for record in sensitivity.records
        if record.scale == 2.0 && record.order == 1
    )
    @test high_noise_order_one.q_train_median == 0.75
    @test high_noise_order_one.q_validation_median == 0.9
    @test high_noise_order_one.predictive_distance_median == 0.6
    @test high_noise_order_one.fisher_resolution_scale == 2.0
    @test high_noise_order_one.shape_instability_median == 0.7
    @test !high_noise_order_one.underfit
    @test !high_noise_order_one.measure_unstable

    nominal_order_two = only(
        record for record in sensitivity.records
        if record.scale == 1.0 && record.order == 2
    )
    @test nominal_order_two.overparameterized
    @test !nominal_order_two.conclusion_changed
end

@testset "matrix Bures shape is covariance-scale invariant" begin
    fixture = _pure_health_fixture(2; basis=:noise_matrix_bures)
    target = _pure_health_input(
        fixture.layout, [-1.0, 1.0],
        [Matrix{ComplexF64}(I, 2, 2) for _ in 1:2],
    )
    left = ComplexF64[1, 0]
    right = ComplexF64[0, 1]
    expansions = [
        _pure_health_expansion(fixture, [0.4], [left * left']),
        _pure_health_expansion(fixture, [0.4], [right * right']),
    ]
    report = _pure_noise_geometry_report(fixture, target, expansions, 1.0)
    four_covariance =
        _pure_noise_geometry_report(fixture, target, expansions, 4.0)
    order = only(report.orders)
    records = report.provenance.noise_sensitivity.records
    nominal = only(record for record in records if record.scale == 1.0)
    rescaled = only(record for record in records if record.scale == 2.0)

    @test nominal.shape_instability_median > 0
    @test rescaled.shape_instability_median ≈
          nominal.shape_instability_median
    @test rescaled.shape_instability_median ≈
          only(four_covariance.orders).shape_instability.median
end

@testset "scalar energy shape follows inverse noise scale" begin
    fixture = _pure_health_fixture(1; basis=:noise_scalar_energy)
    target = _pure_health_input(
        fixture.layout, [-1.0, 1.0],
        fill(1.0 + 0im, 2),
    )
    expansions = [
        _pure_health_expansion(fixture, [0.2], [1.0]),
        _pure_health_expansion(fixture, [0.4], [1.0]),
    ]
    report = _pure_noise_geometry_report(fixture, target, expansions, 1.0)
    four_covariance =
        _pure_noise_geometry_report(fixture, target, expansions, 4.0)
    records = report.provenance.noise_sensitivity.records
    nominal = only(record for record in records if record.scale == 1.0)
    rescaled = only(record for record in records if record.scale == 2.0)

    @test rescaled.shape_instability_median ≈
          nominal.shape_instability_median / 2
    @test rescaled.shape_instability_median ≈
          only(four_covariance.orders).shape_instability.median
end

@testset "bath-fit noise sensitivity preserves paired overfit sign" begin
    sufficient = _pure_sensitivity_order(
        1;
        q_train=0.5,
        q_validation=2.0,
        predictive_distance=1.0,
        shape_instability=nothing,
        overall=:predictive_overfit,
        predictive_overfit=true,
        paired_replicas=4,
    )
    insufficient = _pure_sensitivity_order(
        2;
        q_train=0.5,
        q_validation=2.0,
        predictive_distance=1.0,
        shape_instability=nothing,
        overall=:predictive_overfit,
        predictive_overfit=true,
        paired_replicas=3,
    )
    config = BathFitDiagnosticConfig(
        min_replicas=4,
        noise_scale_sensitivity=(0.5, 2.0),
    )
    records = GraftImpurity._bathhealth_noise_sensitivity(
        BathFitOrderHealth[sufficient, insufficient], config,
    ).records

    @test all(
        record -> record.predictive_overfit,
        (record for record in records if record.order == 1),
    )
    @test all(
        record -> !record.predictive_overfit,
        (record for record in records if record.order == 2),
    )
end
