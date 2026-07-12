using LinearAlgebra: Hermitian, I, eigmin, norm

@testset "experimental scalar LorentzianPSD" begin
    diagnostics = (; source=:constructed)
    model = LorentzianPSD([-0.6, 0.9], [0.2, 0.4], [0.7, 0.3], diagnostics)
    grid = collect(range(-5.0, 5.0; length=1001))
    values = spectral_density(model, grid)

    @test length(model) == 2
    @test complex_poles(model) ≈ [-0.6 - 0.2im, 0.9 - 0.4im]
    @test eltype(values) == Float64
    @test all(>=(0.0), values)
    @test spectral_density(model, 0.17) >= 0
    @test_throws ArgumentError spectral_density(model, Inf)
    @test_throws ArgumentError LorentzianPSD([0.0], [0.0], [1.0], diagnostics)
    @test_throws ArgumentError LorentzianPSD([0.0], [1e-200], [1.0], diagnostics)
    @test_throws ArgumentError LorentzianPSD([0.0], [0.1], [-1.0], diagnostics)
    @test_throws ArgumentError LorentzianPSD([0.0, 1.0], [0.1], [1.0], diagnostics)

    frequencies = collect(range(-4.0, 4.0; length=401))
    exact = LorentzianPSD([0.35], [0.28], [0.8], (; source=:synthetic))
    samples = spectral_density(exact, frequencies)
    @test_logs (:warn, r"highly experimental") lorentzian_fit(
        samples, frequencies; n_poles=1, maxiter=0,
        initial_centers=[0.35], initial_widths=[0.28])

    fitted = lorentzian_fit(samples, frequencies; n_poles=1, maxiter=250,
                            initial_centers=[0.1], initial_widths=[0.5],
                            warn_experimental=false)
    reconstructed = spectral_density(fitted, frequencies)
    @test fitted.diagnostics.experimental
    @test fitted.diagnostics.proof_status == :parameterization_only
    @test !fitted.diagnostics.rigorous_completeness_proof
    @test fitted.diagnostics.positivity == :structural
    @test fitted.diagnostics.minimum_width ≈ 0.01
    @test fitted.diagnostics.relative_l2_error < 1e-6
    @test norm(reconstructed - samples) / norm(samples) < 1e-6
    @test fitted.centers[1] ≈ 0.35 atol=1e-5
    @test fitted.widths[1] ≈ 0.28 atol=1e-5
    @test fitted.weights[1] ≈ 0.8 atol=1e-5

    two_component = LorentzianPSD([-0.8, 1.1], [0.25, 0.45], [0.4, 0.7],
                                  (; source=:synthetic))
    two_samples = spectral_density(two_component, frequencies)
    fixed_poles = lorentzian_fit(
        two_samples, frequencies; n_poles=2, maxiter=0,
        initial_centers=two_component.centers,
        initial_widths=two_component.widths, warn_experimental=false)
    @test fixed_poles.weights ≈ two_component.weights atol=1e-10
    @test fixed_poles.diagnostics.relative_l2_error < 1e-10
    @test fixed_poles.diagnostics.optimizer == :initial_nnls
    @test fixed_poles.diagnostics.optimization_skip_reason == :maxiter_zero

    objective, gradient! = GraftImpurity._lorentzian_objective(
        frequencies, two_samples, 2, 0.01, norm(two_samples))
    parameters = vcat([-0.7, 1.0], sqrt.([0.24, 0.34]), sqrt.([0.35, 0.65]))
    analytic = zeros(length(parameters))
    gradient!(analytic, parameters)
    step = 1e-6
    finite_difference = map(eachindex(parameters)) do k
        plus = copy(parameters); plus[k] += step
        minus = copy(parameters); minus[k] -= step
        (objective(plus) - objective(minus)) / (2step)
    end
    @test analytic ≈ finite_difference rtol=2e-6 atol=1e-8

    zero_fit = lorentzian_fit(zeros(length(frequencies)), frequencies;
                              n_poles=3, maxiter=10,
                              warn_experimental=false)
    @test iszero(maximum(spectral_density(zero_fit, frequencies)))
    @test zero_fit.diagnostics.converged === nothing
    @test zero_fit.diagnostics.optimizer == :initial_nnls
    @test zero_fit.diagnostics.optimization_skip_reason == :zero_spectrum

    @test_throws ArgumentError lorentzian_fit(
        ComplexF64.(samples), frequencies; n_poles=1,
        warn_experimental=false)
    @test_throws ArgumentError lorentzian_fit(
        -samples, frequencies; n_poles=1, warn_experimental=false)
    @test_throws ArgumentError lorentzian_fit(
        fill(-1e-7, length(frequencies)), frequencies; n_poles=1,
        warn_experimental=false)
    @test_throws DimensionMismatch lorentzian_fit(
        samples[1:end-1], frequencies; n_poles=1,
        warn_experimental=false)
    @test_throws ArgumentError lorentzian_fit(
        samples, frequencies; n_poles=1, minimum_width=0.6,
        initial_widths=[0.5], warn_experimental=false)
    narrow = sqrt(floatmin(Float64))
    narrow_model = LorentzianPSD([0.0], [narrow], [1.0], diagnostics)
    @test isfinite(spectral_density(narrow_model, 0.0))
end

@testset "experimental matrix LorentzianPSD" begin
    centers = [-0.7, 0.9]
    widths = [0.22, 0.35]
    vectors = [ComplexF64[0.8, 0.25im],
               ComplexF64[0.3 + 0.1im, 0.7]]
    residues = [vector * vector' for vector in vectors]
    exact = MatrixLorentzianPSD(
        centers, widths, residues, (; source=:synthetic))
    frequencies = collect(range(-3.0, 3.0; length=301))
    samples = spectral_density(exact, frequencies)

    @test length(exact) == 2
    @test complex_poles(exact) ≈ centers .- im .* widths
    @test eltype(samples) == Matrix{ComplexF64}
    @test any(abs(imag(value[1, 2])) > 1e-3 for value in samples)
    @test all(norm(value - value') < 1e-13 for value in samples)
    @test all(eigmin(Hermitian(value)) >= -1e-13 for value in samples)
    reversed = MatrixLorentzianPSD(
        reverse(centers), reverse(widths), reverse(residues), (;))
    @test reversed.centers == centers
    @test reversed.widths == widths
    @test reversed.residues ≈ residues

    @test_logs (:warn, r"highly experimental") lorentzian_fit(
        samples, frequencies; n_poles=2, maxiter=0,
        initial_centers=centers, initial_widths=widths)
    fixed = lorentzian_fit(
        samples, frequencies; n_poles=2, maxiter=0,
        initial_centers=centers, initial_widths=widths,
        warn_experimental=false)
    reconstructed = spectral_density(fixed, frequencies)
    relative_error = sqrt(sum(norm(reconstructed[n] - samples[n])^2
                              for n in eachindex(samples))) /
                     sqrt(sum(norm(value)^2 for value in samples))
    @test fixed isa MatrixLorentzianPSD
    @test fixed.diagnostics.sdp_solves == 0
    @test fixed.diagnostics.matrix_dimension == 2
    @test fixed.diagnostics.positivity == :structural_residue_factorization
    @test fixed.diagnostics.optimizer == :initial_psd_projection
    @test relative_error < 5e-8
    @test all(eigmin(Hermitian(residue)) >= -1e-13
              for residue in fixed.residues)
    @test any(abs(imag(residue[1, 2])) > 1e-3
              for residue in fixed.residues)

    training = 1:2:length(frequencies)
    holdout = 2:2:length(frequencies)
    fitted = lorentzian_fit(
        samples[training], frequencies[training]; n_poles=2, maxiter=500,
        initial_centers=[-0.55, 0.75], initial_widths=[0.3, 0.45],
        warn_experimental=false)
    fitted_values = spectral_density(fitted, frequencies[holdout])
    fitted_error = sqrt(sum(norm(fitted_values[n] - samples[holdout[n]])^2
                            for n in eachindex(holdout))) /
                   sqrt(sum(norm(samples[n])^2 for n in holdout))
    @test fitted_error < 1e-5
    @test all(eigmin(Hermitian(value)) >= -1e-12 for value in fitted_values)

    factor_a = ComplexF64[0.7 0.1im; 0.2im 0.3]
    factor_b = ComplexF64[0.2 + 0.1im 0.05; -0.1im 0.6]
    factors = [factor_a, factor_b]
    gradient_centers = [-0.6, 0.8]
    gradient_widths = [0.24, 0.38]
    width_floor = 0.01
    parameters = GraftImpurity._matrix_lorentzian_pack(
        gradient_centers, gradient_widths, factors, width_floor)
    kernel = GraftImpurity._lorentzian_kernel(
        frequencies, centers, widths)
    gradient_target = GraftImpurity._matrix_lorentzian_evaluate(
        kernel, residues)
    scale = sqrt(sum(norm(value)^2 for value in gradient_target))
    objective, gradient! = GraftImpurity._matrix_lorentzian_objective(
        frequencies, gradient_target, 2, 2, width_floor, scale)
    analytic = zeros(length(parameters))
    gradient!(analytic, parameters)
    step = 1e-6
    finite_difference = map(eachindex(parameters)) do k
        plus = copy(parameters); plus[k] += step
        minus = copy(parameters); minus[k] -= step
        (objective(plus) - objective(minus)) / (2step)
    end
    @test analytic ≈ finite_difference rtol=3e-6 atol=2e-8

    zero_samples = [zeros(ComplexF64, 2, 2) for _ in frequencies]
    zero_fit = lorentzian_fit(
        zero_samples, frequencies; n_poles=2, maxiter=10,
        warn_experimental=false)
    @test all(iszero, zero_fit.residues)
    @test zero_fit.diagnostics.optimization_skip_reason == :zero_spectrum

    nonhermitian = copy(samples)
    nonhermitian[1] = ComplexF64[1 0.2im; 0.2im 1]
    indefinite = copy(samples)
    indefinite[1] = ComplexF64[1 2im; -2im 1]
    inconsistent = copy(samples)
    inconsistent[1] = Matrix{ComplexF64}(I, 3, 3)
    @test_throws ArgumentError lorentzian_fit(
        nonhermitian, frequencies; n_poles=2, warn_experimental=false)
    @test_throws ArgumentError lorentzian_fit(
        indefinite, frequencies; n_poles=2, warn_experimental=false)
    @test_throws DimensionMismatch lorentzian_fit(
        inconsistent, frequencies; n_poles=2, warn_experimental=false)
    @test_throws ArgumentError MatrixLorentzianPSD(
        [0.0], [0.2], [ComplexF64[1 2im; -2im 1]], (;))
    @test_throws ArgumentError MatrixLorentzianPSD(
        [0.0], [1e-200], [Matrix{ComplexF64}(I, 2, 2)], (;))
    narrow = sqrt(floatmin(Float64))
    narrow_model = MatrixLorentzianPSD(
        [0.0], [narrow], [Matrix{ComplexF64}(I, 2, 2)], (;))
    @test all(isfinite, spectral_density(narrow_model, 0.0))
end
