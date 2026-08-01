using Test
using LinearAlgebra
using GraftImpurityFoundations
using GraftImpurityBaths
using GraftImpurityBathFit

@testset "bath-fit spectral details" begin
    fixture = _pure_health_fixture(1; basis=:health_spectral_details)
    frequencies = [-2.0, -1.0, 1.0, 2.0]
    validation = _pure_health_input(
        fixture.layout, frequencies,
        ComplexF64[1 / (im * frequency - 0.25)
                   for frequency in frequencies],
    )
    expansion = _pure_health_expansion(
        fixture, [-2.0, -0.5, 0.5, 3.0], [1e-20, 1.0, 2.0, 1.0],
    )
    covariance = _pure_health_covariance(validation)
    metric = GraftImpurityBathFit._bathhealth_metric(
        covariance, :validation, validation,
    )
    details = GraftImpurityBathFit._bathhealth_spectral_details(
        expansion, (-1.0, 1.0), validation, metric,
    )

    @test details.total_trace_mass ≈ 4.0
    @test only(details.total_residue_matrices.block) ≈ 4.0
    @test details.admissible
    @test details.residue_ranks == [1, 1, 1, 1]
    @test details.effective_ranks ≈ ones(4)
    @test details.weak_residue_fraction ≈ 2.5e-21
    @test details.minimum_pole_spacing == 1.0
    @test details.outside_mass.below ≈ 2.5e-21
    @test details.outside_mass.above ≈ 0.25
    @test length(details.fisher_resolution) == 4
    @test all(isfinite, details.fisher_resolution)
    @test details.kernel_noise_rank isa Int
    @test details.capacity_ratio >= 1

    uncalibrated = GraftImpurityBathFit._bathhealth_metric(
        nothing, :validation, validation,
    )
    uncalibrated_details = GraftImpurityBathFit._bathhealth_spectral_details(
        expansion, (-1.0, 1.0), validation, uncalibrated,
    )
    @test ismissing(uncalibrated_details.fisher_resolution)
    @test ismissing(uncalibrated_details.kernel_noise_rank)
    @test ismissing(uncalibrated_details.capacity_ratio)

    indefinite = _pure_health_expansion(fixture, [0.0], [-1.0])
    indefinite_details = GraftImpurityBathFit._bathhealth_spectral_details(
        indefinite, (-1.0, 1.0), validation, metric,
    )
    @test !indefinite_details.admissible
    @test indefinite_details.total_trace_mass == -1.0

    reference = _pure_health_expansion(fixture, [-2.0, 0.0], [1.0, 1.0])
    current = _pure_health_expansion(fixture, [-0.5, 0.0], [1.0, 1.0])
    flux = GraftImpurityBathFit._bathhealth_boundary_flux(
        reference, current, (-1.0, 1.0),
    )
    @test flux.below == -1.0
    @test flux.above == 0.0
    @test flux.total == 1.0
    @test flux.fractions.total == 0.5
end

@testset "bath-fit pole spacing is block local" begin
    layout = FlavorLayout(
        [:orbital_a, :orbital_b],
        Dict(:orbital_a => :impurity, :orbital_b => :impurity),
        Dict(:impurity => [:orbital_a, :orbital_b]);
        basis=:health_block_local_spacing,
    )
    partition = Partition(
        :block_a => [:orbital_a],
        :block_b => [:orbital_b],
    )
    expansion = PoleExpansion(
        BlockRealPoles(
            layout, partition, [0.0, 2.0, 0.0, 3.0],
            [1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2];
            statistics=:fermion,
        );
        kernel=:health_block_local_spacing,
    )
    @test GraftImpurityBathFit._bathhealth_minimum_pole_spacing(expansion) == 2.0

    no_pairs = PoleExpansion(
        BlockRealPoles(
            layout, partition, [0.0, 0.0], [1.0, 1.0], [1, 2];
            statistics=:fermion,
        );
        kernel=:health_block_local_spacing,
    )
    @test isnothing(
        GraftImpurityBathFit._bathhealth_minimum_pole_spacing(no_pairs),
    )
end

@testset "bath-fit transport uses Simpson energy integration" begin
    fixture = _pure_health_fixture(1; basis=:health_simpson_transport)
    frequencies = [-2.0, -1.0, 1.0, 2.0]
    validation = _pure_health_input(
        fixture.layout, frequencies,
        zeros(ComplexF64, length(frequencies)),
    )
    metric = GraftImpurityBathFit._bathhealth_metric(
        _pure_health_covariance(validation), :validation, validation,
    )
    left = only(GraftImpurityBathFit._bathhealth_measures(
        _pure_health_expansion(fixture, [-1.0], [1.0]),
    ))
    right = only(GraftImpurityBathFit._bathhealth_measures(
        _pure_health_expansion(fixture, [1.0], [1.0]),
    ))
    state = reshape(ComplexF64[1], 1, 1)
    endpoint_left = GraftImpurityBathFit._bathhealth_fisher_resolution(
        validation, :block, -1.0, state, metric,
    )
    midpoint = GraftImpurityBathFit._bathhealth_fisher_resolution(
        validation, :block, 0.0, state, metric,
    )
    endpoint_right = GraftImpurityBathFit._bathhealth_fisher_resolution(
        validation, :block, 1.0, state, metric,
    )
    expected = 2 / 6 * (
        inv(endpoint_left) + 4inv(midpoint) + inv(endpoint_right)
    )
    @test GraftImpurityBathFit._bathhealth_transport(
        left, right, validation, :block, metric,
    ) ≈ expected
end
