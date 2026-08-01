@testset "bath-fit residual profiles and inactive diagonal channels" begin
    fixture = _pure_health_fixture(2; basis=:health_profiles)
    frequencies = [0.5, 1.0, 2.0]
    targets = Matrix{ComplexF64}[
        [1 / frequency 0.0; 0.0 0.0] for frequency in frequencies
    ]
    predictions = deepcopy(targets)
    predictions[1][1, 1] += 0.5
    predictions[2][1, 2] += 0.25im
    target = _pure_health_input(fixture.layout, frequencies, targets)
    prediction = _pure_health_input(
        fixture.layout, frequencies, predictions,
    )
    covariance = _pure_health_covariance(target)
    profile = GraftImpurityBathFit._bathhealth_residual_profile(
        target, prediction, covariance, :validation,
    )

    @test profile.calibrated
    @test profile.max_standardized ≈ 0.5
    @test profile.bands.low > 0
    @test profile.bands.mid > 0
    @test profile.bands.high == 0
    @test profile.band_edges == (0.5, 1.0)
    @test length(profile.diagonal_channels) == 2
    @test profile.diagonal_channels[1].active
    @test !profile.diagonal_channels[2].active
    @test profile.diagonal_aggregate > 0
end

@testset "all-zero diagonal targets stay inactive" begin
    fixture = _pure_health_fixture(2; basis=:health_profile_zero_channels)
    frequencies = [0.5, 1.0]
    targets = [zeros(ComplexF64, 2, 2) for _ in frequencies]
    predictions = deepcopy(targets)
    predictions[1][1, 1] = 1.0
    target = _pure_health_input(fixture.layout, frequencies, targets)
    prediction = _pure_health_input(
        fixture.layout, frequencies, predictions,
    )
    profile = GraftImpurityBathFit._bathhealth_residual_profile(
        target, prediction, nothing, :validation,
    )

    @test all(iszero(record.target_mass)
              for record in profile.diagonal_channels)
    @test all(!record.active for record in profile.diagonal_channels)
    @test isinf(profile.diagonal_channels[1].q)
    @test iszero(profile.diagonal_channels[2].q)
    @test ismissing(profile.diagonal_aggregate)
end

@testset "bosonic zero frequency stays outside dynamic profiles" begin
    fixture = _pure_health_fixture(1; basis=:health_profile_boson)
    frequencies = [0.0, 1.0, 2.0]
    targets = ComplexF64[1.0, 0.5, 0.25]
    predictions = copy(targets)
    predictions[1] += 10
    target = _pure_health_input(
        fixture.layout, frequencies, targets; statistics=:boson,
    )
    prediction = _pure_health_input(
        fixture.layout, frequencies, predictions; statistics=:boson,
    )
    profile = GraftImpurityBathFit._bathhealth_residual_profile(
        target, prediction, _pure_health_covariance(target), :validation,
    )
    @test profile.max_standardized == 0
    @test profile.bands.low == 0
    @test profile.bands.mid == 0
end
