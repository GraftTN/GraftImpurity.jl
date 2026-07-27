@testset "pure bath-fit health retains full matrix errors" begin
    fixture = _pure_health_fixture(2; basis=:health_full_matrix)
    frequency = [1.0]
    target_matrix = Matrix{ComplexF64}(I, 2, 2)
    target = _pure_health_input(
        fixture.layout, frequency, [target_matrix],
    )
    offdiagonal = copy(target_matrix)
    offdiagonal[1, 2] += 2 + im
    offdiagonal_prediction = _pure_health_input(
        fixture.layout, frequency, [offdiagonal],
    )
    candidates = BathFitHealthCandidate[
        _pure_health_candidate(1, target, target),
        _pure_health_candidate(2, target, offdiagonal_prediction),
    ]
    report = analyze_bathfit(
        candidates, target, target;
        covariance=_pure_health_covariance(target),
        config=BathFitDiagnosticConfig(
            n_replicas=1, min_replicas=1,
        ),
    )
    @test _pure_health_order(report, 1).q_validation.mean == 0
    @test _pure_health_order(report, 2).q_validation.mean ≈
          abs(2 + im) / sqrt(8)
end

function _pure_health_rotated_input(
    input::BathFitInput, layout::FlavorLayout,
    unitary::AbstractMatrix{<:Number},
)
    samples = Matrix{ComplexF64}[
        unitary * sample * unitary' for sample in input.blocks.block
    ]
    return _pure_health_input(
        layout, input.frequencies, samples;
        statistics=input.statistics,
    )
end

@testset "pure bath-fit health is invariant under a common unitary" begin
    fixture = _pure_health_fixture(2; basis=:health_unitary_original)
    rotated_fixture = _pure_health_fixture(
        2; basis=:health_unitary_rotated,
    )
    frequencies = [0.75, 1.5]
    targets = Matrix{ComplexF64}[
        [1.0 0.2 + 0.1im; 0.2 - 0.1im 0.7],
        [0.8 -0.1 + 0.2im; -0.1 - 0.2im 1.2],
    ]
    target = _pure_health_input(fixture.layout, frequencies, targets)
    first_predictions = Matrix{ComplexF64}[
        sample + [0.1 0.2im; -0.2im -0.1] for sample in targets
    ]
    second_predictions = Matrix{ComplexF64}[
        sample + [0.3 0.1; 0.1 -0.2] for sample in targets
    ]
    first_prediction = _pure_health_input(
        fixture.layout, frequencies, first_predictions,
    )
    second_prediction = _pure_health_input(
        fixture.layout, frequencies, second_predictions,
    )
    first_vector = ComplexF64[1, 0]
    second_vector = ComplexF64[1, im] / sqrt(2)
    first_residue = first_vector * first_vector'
    second_residue = second_vector * second_vector'
    first_expansion = _pure_health_expansion(
        fixture, [0.4], [first_residue],
    )
    second_expansion = _pure_health_expansion(
        fixture, [0.4], [second_residue],
    )
    candidates = BathFitHealthCandidate[
        _pure_health_candidate(
            1, target, first_prediction;
            replica=1, expansion=first_expansion,
        ),
        _pure_health_candidate(
            1, target, second_prediction;
            replica=2, expansion=second_expansion,
        ),
    ]
    report = analyze_bathfit(
        candidates, target, target;
        covariance=_pure_health_covariance(target),
        config=BathFitDiagnosticConfig(
            n_replicas=2, min_replicas=1,
        ),
    )

    unitary = ComplexF64[1 im; im 1] / sqrt(2)
    rotated_target = _pure_health_rotated_input(
        target, rotated_fixture.layout, unitary,
    )
    rotated_first_prediction = _pure_health_rotated_input(
        first_prediction, rotated_fixture.layout, unitary,
    )
    rotated_second_prediction = _pure_health_rotated_input(
        second_prediction, rotated_fixture.layout, unitary,
    )
    rotated_first_expansion = _pure_health_expansion(
        rotated_fixture, [0.4],
        [unitary * first_residue * unitary'],
    )
    rotated_second_expansion = _pure_health_expansion(
        rotated_fixture, [0.4],
        [unitary * second_residue * unitary'],
    )
    rotated_candidates = BathFitHealthCandidate[
        _pure_health_candidate(
            1, rotated_target, rotated_first_prediction;
            replica=1, expansion=rotated_first_expansion,
        ),
        _pure_health_candidate(
            1, rotated_target, rotated_second_prediction;
            replica=2, expansion=rotated_second_expansion,
        ),
    ]
    rotated_report = analyze_bathfit(
        rotated_candidates, rotated_target, rotated_target;
        covariance=_pure_health_covariance(rotated_target),
        config=BathFitDiagnosticConfig(
            n_replicas=2, min_replicas=1,
        ),
    )

    order = only(report.orders)
    rotated_order = only(rotated_report.orders)
    @test rotated_order.q_validation.values ≈ order.q_validation.values
    @test rotated_order.predictive_distance.values ≈
          order.predictive_distance.values
    @test order.shape_instability !== nothing
    @test rotated_order.shape_instability !== nothing
    # With equal pole positions and masses this transport term is the Bures
    # contribution, so equality checks its common-unitary invariance.
    @test something(rotated_order.shape_instability).values ≈
          something(order.shape_instability).values atol=1e-7
    @test rotated_order.verdicts == order.verdicts
    @test rotated_report.verdicts == report.verdicts
end

@testset "pure bath-fit health keeps prediction metrics for indefinite residues" begin
    fixture = _pure_health_fixture(2; basis=:health_indefinite)
    frequencies = [-1.0, 1.0]
    targets = Matrix{ComplexF64}[
        Matrix{ComplexF64}(I, 2, 2) for _ in frequencies
    ]
    target = _pure_health_input(fixture.layout, frequencies, targets)
    prediction = _pure_health_input(
        fixture.layout, frequencies,
        [sample + 0.1Matrix{ComplexF64}(I, 2, 2) for sample in targets],
    )
    indefinite = _pure_health_expansion(
        fixture, [0.25], [ComplexF64[1 0; 0 -0.2]],
    )
    report = analyze_bathfit(
        BathFitHealthCandidate[
            _pure_health_candidate(
                1, target, prediction;
                expansion=indefinite, mountable=false,
            ),
        ],
        target,
        target;
        covariance=_pure_health_covariance(target),
        config=BathFitDiagnosticConfig(
            n_replicas=1, min_replicas=1,
        ),
    )
    order = only(report.orders)
    @test isfinite(order.q_validation.mean)
    @test order.q_validation.mean > 0
    @test isfinite(order.predictive_distance.mean)
    @test order.admissible_fraction == 0
end
