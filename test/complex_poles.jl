using Test
using LinearAlgebra: norm
using GraftImpurity

function _bcf_layout(flavors::Vector{Symbol}; basis::Symbol=:bcf_basis)
    return FlavorLayout(
        flavors, Dict(flavor => :impurity for flavor in flavors),
        Dict(:impurity => flavors); basis,
    )
end

function _bcf_samples(exponents, weights, times)
    return Matrix{ComplexF64}[
        sum(weight .* exp(-exponent * time)
            for (exponent, weight) in zip(exponents, weights))
        for time in times
    ]
end

@testset "complex BCF MiniPole" begin
    layout = _bcf_layout([:up, :down]; basis=:complex_bcf)
    partition = Partition(:spin => [:up, :down])
    times = collect(0.0:0.125:1.375)
    exponents = ComplexF64[0.35 + 0.7im, 0.8 - 0.45im]
    weights = Matrix{ComplexF64}[
        ComplexF64[0.7 + 0.1im 0.25 - 0.35im;
                   -0.15 + 0.2im 0.45 - 0.05im],
        ComplexF64[-0.1 + 0.2im 0.4 + 0.05im;
                   0.3 - 0.1im 0.2 + 0.15im],
    ]
    samples = _bcf_samples(exponents, weights, times)
    input = BCFFitInput(
        layout, times, :spin => samples; channel=:boson,
        metadata=(; source=:synthetic_bcf),
    )
    fitted = fit_complex_bcf(
        input, MiniPoleKernel(n_poles=2, rank_tolerance=1e-10), partition,
    )
    @test fitted isa ComplexPoles
    @test fitted isa AbstractBCFParametrization
    @test fitted isa AbstractBathParametrization
    @test !(fitted isa AbstractHamiltonianBath)
    @test fitted.channel === :boson
    @test length(fitted) == 2
    @test any(weight -> abs(weight[1, 2] - weight[2, 1]) > 1e-4,
              fitted.weights)
    @test maximum(norm(value - target)
                  for (value, target) in zip(
                      evaluate_bcf(fitted, times, :spin), samples,
                  )) < 1e-8
    @test evaluate_bcf(fitted, times) == evaluate_bcf(fitted, times, :spin)
    @test any(exponent -> abs(imag(exponent)) > 0.1, fitted.poles)
    @test fitted.poles ≈ exponents[sortperm(exponents; by=value ->
                                             (real(value), imag(value)))] atol=1e-8
    @test fitted.diagnostics.fits[1].engine.selected_attempt.training_error.relative_l2 < 1e-8
    hamiltonian_input = BathFitInput(
        layout, [0.5, 1.0, 1.5, 2.0],
        :spin => fill(zeros(ComplexF64, 2, 2), 4);
        domain=:matsubara, statistics=:fermion,
    )
    @test !hasmethod(realize_bath, Tuple{BathFitInput, ComplexPoles, Partition})
    @test_throws MethodError realize_bath(hamiltonian_input, fitted, partition)
    @test !hasmethod(mount_bath, Tuple{GraftImpurity.TreeTopology, ComplexPoles})
    @test_throws MethodError mount_bath(:not_a_topology, fitted)

    heldout = fit_complex_bcf(
        input,
        MiniPoleKernel(n_poles=3, rank_tolerance=1e-10, holdout_count=2),
        partition,
    )
    @test length(heldout) == 2
    @test heldout.diagnostics.fits[1].engine.training_count == length(times) - 2
    @test heldout.diagnostics.fits[1].engine.selected_attempt.holdout_error.relative_l2 < 1e-8

    alias_limit = pi / (times[2] - times[1])
    alias_exponent = ComplexF64(0.55 + 0.99 * alias_limit * im)
    alias_weight = Matrix{ComplexF64}[ComplexF64[0.4 - 0.1im 0.2im;
                                                  0.1 + 0.3im -0.2]]
    alias_samples = _bcf_samples([alias_exponent], alias_weight, times)
    alias_input = BCFFitInput(layout, times, :spin => alias_samples; channel=:boson)
    alias_fit = fit_complex_bcf(
        alias_input, MiniPoleKernel(n_poles=1, rank_tolerance=1e-10), partition,
    )
    @test alias_fit.diagnostics.fits[1].alias_warning
    @test only(alias_fit.poles) ≈ alias_exponent atol=1e-8
    @test maximum(norm(value - target)
                  for (value, target) in zip(
                      evaluate_bcf(alias_fit, times, :spin), alias_samples,
                  )) < 1e-8

    short_layout = _bcf_layout([:short]; basis=:short_bcf)
    short_partition = Partition(:short => [:short])
    short_times = collect(0.0:0.1:0.3)
    short_exponents = ComplexF64[0.25 + 0.3im, 0.7 - 0.2im]
    short_weights = Matrix{ComplexF64}[reshape(ComplexF64[0.7 - 0.1im], 1, 1),
                                       reshape(ComplexF64[-0.2 + 0.3im], 1, 1)]
    short_samples = _bcf_samples(short_exponents, short_weights, short_times)
    short_input = BCFFitInput(
        short_layout, short_times, :short => short_samples; channel=:boson,
    )
    short_fit = fit_complex_bcf(
        short_input, MiniPoleKernel(n_poles=2, rank_tolerance=1e-10),
        short_partition,
    )
    @test length(short_fit) == 2
    @test maximum(norm(value - target)
                  for (value, target) in zip(
                      evaluate_bcf(short_fit, short_times, :short), short_samples,
                  )) < 1e-8

    mixed_layout = _bcf_layout([:charge, :up, :down]; basis=:mixed_bcf)
    mixed_partition = Partition(:charge => [:charge], :spin => [:up, :down])
    charge_exponent = ComplexF64[0.4 + 0.2im]
    charge_weight = Matrix{ComplexF64}[reshape(ComplexF64[0.8 - 0.1im], 1, 1)]
    spin_exponent = ComplexF64[0.6 - 0.1im]
    spin_weight = Matrix{ComplexF64}[ComplexF64[0.2 0.4im; -0.1im 0.5]]
    charge_samples = _bcf_samples(charge_exponent, charge_weight, times)
    spin_samples = _bcf_samples(spin_exponent, spin_weight, times)
    mixed_input = BCFFitInput(
        mixed_layout, times, :charge => charge_samples, :spin => spin_samples;
        channel=:fermion_lesser,
    )
    mixed_fit = fit_complex_bcf(
        mixed_input, MiniPoleKernel(n_poles=1, rank_tolerance=1e-10),
        mixed_partition,
    )
    @test mixed_fit.block_indices == [1, 2]
    @test size(mixed_fit.weights[1]) == (1, 1)
    @test size(mixed_fit.weights[2]) == (2, 2)
    @test maximum(norm(value - target)
                  for (value, target) in zip(
                      evaluate_bcf(mixed_fit, times, :charge), charge_samples,
                  )) < 1e-8
    @test maximum(norm(value - target)
                  for (value, target) in zip(
                      evaluate_bcf(mixed_fit, times, :spin), spin_samples,
                  )) < 1e-8
    direct = ComplexPoles(
        mixed_layout, mixed_partition,
        ComplexF64[0.4 + 0.2im, 0.6 - 0.1im],
        Matrix{ComplexF64}[reshape(ComplexF64[0.8 - 0.1im], 1, 1),
                           ComplexF64[0.2 0.4im; -0.1im 0.5]],
        [1, 2]; channel=:fermion_lesser,
        diagnostics=(; source=:manual),
    )
    @test evaluate_bcf(direct, 0.0, :charge) == reshape(ComplexF64[0.8 - 0.1im], 1, 1)
    @test evaluate_bcf(direct, 0.0, :spin) == ComplexF64[0.2 0.4im; -0.1im 0.5]
    @test_throws DimensionMismatch ComplexPoles(
        mixed_layout, mixed_partition, ComplexF64[0.2 + 0.1im],
        Matrix{ComplexF64}[ones(ComplexF64, 2, 2)], [1]; channel=:boson,
    )
    @test_throws ArgumentError ComplexPoles(
        mixed_layout, mixed_partition, ComplexF64[-0.1 + 0.2im],
        Matrix{ComplexF64}[reshape(ComplexF64[1], 1, 1)], [1]; channel=:boson,
    )
    @test_throws ArgumentError BCFFitInput(
        layout, [0.0, 0.1, 0.3, 0.4], :spin => samples[1:4]; channel=:boson,
    )
    unstable_samples = _bcf_samples(
        ComplexF64[-0.2 + 0.3im], Matrix{ComplexF64}[ComplexF64[1 0; 0 1]],
        times,
    )
    unstable_input = BCFFitInput(
        layout, times, :spin => unstable_samples; channel=:boson,
    )
    @test_throws ArgumentError fit_complex_bcf(
        unstable_input, MiniPoleKernel(n_poles=1), partition,
    )
    @test_throws ArgumentError fit_complex_bcf(
        unstable_input, MiniPoleKernel(n_poles=1, rank_tolerance=0.5), partition,
    )
    fast_exponent = ComplexF64[20.0 + 0.15im]
    fast_weight = Matrix{ComplexF64}[ComplexF64[0.4 0.1im; -0.2im 0.3]]
    fast_samples = _bcf_samples(fast_exponent, fast_weight, times)
    fast_input = BCFFitInput(layout, times, :spin => fast_samples; channel=:boson)
    fast_fit = fit_complex_bcf(
        fast_input, MiniPoleKernel(n_poles=1, rank_tolerance=0.1), partition,
    )
    @test only(fast_fit.poles) ≈ only(fast_exponent) atol=1e-8
    @test maximum(norm(value - target)
                  for (value, target) in zip(
                      evaluate_bcf(fast_fit, times, :spin), fast_samples,
                  )) < 1e-8
    zero_input = BCFFitInput(
        layout, times, :spin => fill(zeros(ComplexF64, 2, 2), length(times));
        channel=:boson,
    )
    zero_fit = fit_complex_bcf(zero_input, MiniPoleKernel(n_poles=1), partition)
    @test isempty(zero_fit)
    @test zero_fit.diagnostics.fits[1].engine.selected_attempt.status === :zero_sequence
    @test all(value -> iszero(norm(value)), evaluate_bcf(zero_fit, times, :spin))
    training_zero_samples = Matrix{ComplexF64}[
        index <= length(times) - 2 ? zeros(ComplexF64, 2, 2) :
        ComplexF64[0.1 0; 0 0.2]
        for index in eachindex(times)
    ]
    training_zero_input = BCFFitInput(
        layout, times, :spin => training_zero_samples; channel=:boson,
    )
    @test_throws ArgumentError fit_complex_bcf(
        training_zero_input, MiniPoleKernel(n_poles=1, holdout_count=2), partition,
    )
end
