using Test
using LinearAlgebra: I
using GraftImpurity

struct _HealthRunnerKernel <: AbstractRealPoleBathFitKernel
    order::Int
    replica::Int
    start::Int
    fail::Bool
    behavior::Symbol
end

const _HEALTH_RUNNER_CALLS = NamedTuple[]

function _health_runner_samples(input::BathFitInput)
    return copy(GraftImpurity._bathfit_realimag_sample_vector(input))
end

function GraftImpurity.real_pole_bath_fit(
    input::BathFitInput, kernel::_HealthRunnerKernel,
    partition::Partition,
)
    push!(_HEALTH_RUNNER_CALLS, (
        order=kernel.order, replica=kernel.replica, start=kernel.start,
        samples=_health_runner_samples(input),
        frequencies=copy(input.frequencies), layout=input.layout,
        metadata=input.metadata,
    ))
    kernel.fail && throw(ArgumentError("intentional health-runner fit failure"))

    poles = if kernel.order == 1
        [0.2]
    else
        additional = Float64[
            0.8 + 0.8 * (index - 1) / max(kernel.order - 2, 1)
            for index in 1:(kernel.order - 1)
        ]
        vcat(0.2, additional)
    end
    residues = if kernel.order == 1
        ComplexF64[1.0]
    elseif kernel.behavior === :overfit
        vcat(ComplexF64[1.0], fill(0.2 + 0im, kernel.order - 1))
    elseif kernel.behavior === :nonmountable
        vcat(ComplexF64[1.0], fill(-0.2 + 0im, kernel.order - 1))
    else
        vcat(ComplexF64[1.0], fill(0.01 + 0im, kernel.order - 1))
    end
    interval = SpectralInterval(-2.0, 2.0, kernel.order)
    plan = DiscretizationPlan(
        :d => BlockDiscretizationPlan([interval]); shared_grid=true,
    )
    raw = BlockRealPoles(
        input.layout, partition, poles, residues, fill(1, kernel.order);
        statistics=input.statistics,
    )
    evidence = if kernel.behavior === :nonconverged_status
        (; fits=[(; nested=(; status=:nonconverged))])
    elseif kernel.behavior === :nonconverged_diagnostics
        (; fits=[(; diagnostics=(; nested=(; converged=false)))])
    else
        (;)
    end
    trace = merge(
        (; plan, order=kernel.order, replica=kernel.replica,
         start=kernel.start),
        evidence,
    )
    return PoleExpansion(
        raw; kernel=:health_runner_fixture,
        trace,
    )
end

function _health_runner_fixture()
    layout = FlavorLayout(
        [:d], Dict(:d => :impurity), Dict(:impurity => [:d]);
        basis=:health_runner,
    )
    partition = Partition(:d => [:d])
    training_frequencies = [0.5, 1.0, 2.0, 3.0]
    validation_frequencies = [0.75, 1.25, 1.75, 2.5]
    samples(frequencies) = ComplexF64[
        inv(im * frequency - 0.2) for frequency in frequencies
    ]
    training = BathFitInput(
        layout, training_frequencies, :d => samples(training_frequencies);
        domain=:matsubara, statistics=:fermion,
        metadata=(; fixture=:health_runner),
    )
    validation = BathFitInput(
        layout, validation_frequencies, :d => samples(validation_frequencies);
        domain=:matsubara, statistics=:fermion,
        metadata=(; fixture=:health_runner_validation),
    )
    return (; layout, partition, training, validation)
end

function _health_runner_covariance(input::BathFitInput; variance=1e-6)
    dimension = length(GraftImpurity._bathfit_realimag_sample_vector(input))
    return Float64(variance) .* Matrix{Float64}(I, dimension, dimension)
end

function _health_runner_call_map(calls)
    return Dict(
        (call.order, call.replica, call.start) => call.samples
        for call in calls
    )
end

@testset "opt-in perturb-and-refit bath diagnostics runner" begin
    fixture = _health_runner_fixture()
    @test_throws InterruptException run_bathfit_diagnostics(
        fixture.training, fixture.partition, [1],
        (order, replica, start) -> throw(InterruptException());
        validation_input=fixture.validation,
        config=BathFitDiagnosticConfig(starts=1),
    )
    for behavior in (:nonconverged_status, :nonconverged_diagnostics)
        candidate = GraftImpurity._bathfit_run_candidate(
            fixture.training, fixture.training, fixture.validation,
            fixture.partition, 1, 1, 1,
            (order, replica, start) ->
                _HealthRunnerKernel(
                    order, replica, start, false, behavior,
                ),
        )
        @test candidate.status === :nonconverged
        @test occursin("nonconvergence", candidate.message)
        @test candidate.training_prediction !== nothing
        @test candidate.validation_prediction !== nothing
    end
    covariance = _health_runner_covariance(fixture.training)
    perturbation = CovariancePerturbation(covariance)
    config = BathFitDiagnosticConfig(
        n_replicas=3, min_replicas=2, starts=2, seed=UInt64(0x1234),
    )
    factory = (order, replica, start) -> _HealthRunnerKernel(
        order, replica, start, order == 2 && replica == 2 && start == 1,
        :standard,
    )

    empty!(_HEALTH_RUNNER_CALLS)
    first_report = run_bathfit_diagnostics(
        fixture.training, fixture.partition, [1, 2], factory;
        validation_input=fixture.validation, perturbation, config,
    )
    first_calls = copy(_HEALTH_RUNNER_CALLS)
    @test length(first_calls) == 2 * config.n_replicas * config.starts
    @test first_report isa BathFitHealthReport
    first_map = _health_runner_call_map(first_calls)
    for replica in 1:config.n_replicas, start in 1:config.starts
        @test first_map[(1, replica, start)] ==
              first_map[(2, replica, start)]
    end
    @test all(call -> call.frequencies == fixture.training.frequencies,
              first_calls)
    @test all(call -> call.layout === fixture.layout, first_calls)
    @test all(call -> call.metadata === fixture.training.metadata, first_calls)

    empty!(_HEALTH_RUNNER_CALLS)
    second_report = run_bathfit_diagnostics(
        fixture.training, fixture.partition, [1, 2], factory;
        validation_input=fixture.validation, perturbation, config,
    )
    @test second_report isa BathFitHealthReport
    second_map = _health_runner_call_map(_HEALTH_RUNNER_CALLS)
    @test first_map == second_map
    order_one = only(filter(item -> item.order == 1, first_report.orders))
    order_two = only(filter(item -> item.order == 2, first_report.orders))
    # The order-one prediction is exact for the unperturbed input. A strictly
    # positive Q_train therefore proves that each score uses its own noisy
    # replica target rather than silently comparing against the original.
    @test all(>(0), order_one.q_train.values)
    @test order_two.failed_fits == 1
    @test order_two.successful_replicas >= 2

    empty!(_HEALTH_RUNNER_CALLS)
    uncalibrated = run_bathfit_diagnostics(
        fixture.training, fixture.partition, [1, 2],
        (order, replica, start) ->
            _HealthRunnerKernel(order, replica, start, false, :standard);
        validation_input=fixture.validation,
        config=BathFitDiagnosticConfig(
            n_replicas=5, min_replicas=2, starts=1,
        ),
    )
    @test length(_HEALTH_RUNNER_CALLS) == 2
    @test !get(uncalibrated.calibration, :calibrated, false)
    @test any(message -> occursin("uncalibrated", lowercase(message)),
              uncalibrated.warnings)

    nonmountable = run_bathfit_diagnostics(
        fixture.training, fixture.partition, [2],
        (order, replica, start) ->
            _HealthRunnerKernel(order, replica, start, false, :nonmountable);
        validation_input=fixture.validation,
        config=BathFitDiagnosticConfig(starts=1),
    )
    nonmountable_order = only(nonmountable.orders)
    @test nonmountable_order.successful_replicas == 1
    @test nonmountable_order.admissible_fraction == 0
    @test isfinite(nonmountable_order.q_validation.mean)

    first_replica = GraftImpurity._bathfit_input_with_realimag(
        fixture.training,
        GraftImpurity._bathfit_realimag_sample_vector(fixture.training) .+ 1e-4,
    )
    second_replica = GraftImpurity._bathfit_input_with_realimag(
        fixture.training,
        GraftImpurity._bathfit_realimag_sample_vector(fixture.training) .- 2e-4,
    )
    empty!(_HEALTH_RUNNER_CALLS)
    empirical = run_bathfit_diagnostics(
        fixture.training, fixture.partition, [1, 2],
        (order, replica, start) ->
            _HealthRunnerKernel(order, replica, start, false, :standard);
        validation_input=fixture.validation,
        perturbation=EmpiricalReplicaPerturbation(
            [first_replica, second_replica],
        ),
        config=BathFitDiagnosticConfig(
            n_replicas=9, min_replicas=2, starts=1,
        ),
    )
    @test empirical isa BathFitHealthReport
    @test empirical.calibration.calibrated
    @test empirical.calibration.training_metric === :explicit
    @test empirical.calibration.validation_metric === :explicit
    @test empirical.thresholds.shrinkage > 0
    @test length(_HEALTH_RUNNER_CALLS) == 4
    empirical_map = _health_runner_call_map(_HEALTH_RUNNER_CALLS)
    for replica in 1:2
        @test empirical_map[(1, replica, 1)] ==
              empirical_map[(2, replica, 1)]
    end

    empty!(_HEALTH_RUNNER_CALLS)
    selected = run_bathfit_diagnostics(
        fixture.training, fixture.partition, [1, 2],
        (order, replica, start) ->
            _HealthRunnerKernel(order, replica, start, false, :overfit);
        validation_input=fixture.validation,
        perturbation=CovariancePerturbation(covariance),
        config=BathFitDiagnosticConfig(
            n_replicas=8, min_replicas=4, starts=1, seed=UInt64(0x9876),
        ),
    )
    @test selected.selected_order == 1
    selected_one = only(filter(item -> item.order == 1, selected.orders))
    selected_two = only(filter(item -> item.order == 2, selected.orders))
    @test selected_one.q_validation.mean < selected_two.q_validation.mean
    @test all(==(1.0), selected_one.returned_poles.values)
    @test all(==(2.0), selected_two.returned_poles.values)
    @test all(==(1.0), selected_one.mode_counts.values)
    @test all(==(2.0), selected_two.mode_counts.values)
end

@testset "complex matrix covariance embedding keeps every entry" begin
    layout = FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis=:health_runner_matrix,
    )
    matrix_input = BathFitInput(
        layout, [0.5, 1.5],
        :spin => Matrix{ComplexF64}[
            ComplexF64[1 + 2im 3 + 4im; 5 + 6im 7 + 8im],
            ComplexF64[9 + 10im 11 + 12im; 13 + 14im 15 + 16im],
        ];
        domain=:matsubara, statistics=:fermion,
    )
    embedded = GraftImpurity._bathfit_realimag_sample_vector(matrix_input)
    rebuilt = GraftImpurity._bathfit_input_with_realimag(matrix_input, embedded)
    @test GraftImpurity._bathfit_realimag_sample_vector(rebuilt) == embedded
    @test rebuilt.blocks.spin == matrix_input.blocks.spin
end
