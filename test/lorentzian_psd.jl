using LinearAlgebra: Diagonal, Hermitian, I, dot, eigen, eigmin, norm

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
    @test zero_fit.diagnostics.optimization_skip_reason == :zero_psd_projection

    @test_throws ArgumentError lorentzian_fit(
        ComplexF64.(samples), frequencies; n_poles=1,
        warn_experimental=false)

    projection_basis = spectral_density(
        LorentzianPSD([0.2], [0.35], [1.0], (;)), frequencies)
    signed_samples = 0.6 .* projection_basis .- 0.02
    signed_fit = lorentzian_fit(
        signed_samples, frequencies; n_poles=1, maxiter=0,
        initial_centers=[0.2], initial_widths=[0.35],
        warn_experimental=false)
    expected_weight = max(
        dot(projection_basis, signed_samples) /
        dot(projection_basis, projection_basis),
        0.0,
    )
    @test signed_fit.weights[1] ≈ expected_weight atol=1e-12
    @test signed_fit.diagnostics.input_projection.minimum_value < 0
    @test signed_fit.diagnostics.input_projection.negative_sample_count > 0
    @test signed_fit.diagnostics.input_projection.nonnegative_cone_distance > 0
    @test signed_fit.diagnostics.projection ==
          :lorentzian_constrained_least_squares
    @test signed_fit.diagnostics.loss == :unweighted_grid_l2
    @test all(>=(0.0), spectral_density(
        signed_fit, range(-8.0, 8.0; length=801)))
    reversed_signed_fit = lorentzian_fit(
        reverse(signed_samples), reverse(frequencies); n_poles=1, maxiter=0,
        initial_centers=[0.2], initial_widths=[0.35],
        warn_experimental=false)
    @test reversed_signed_fit.weights ≈ signed_fit.weights atol=1e-12

    negative_fit = lorentzian_fit(
        -samples, frequencies; n_poles=1, warn_experimental=false)
    @test iszero(negative_fit.weights[1])
    @test negative_fit.diagnostics.optimization_skip_reason ==
          :zero_psd_projection
    @test negative_fit.diagnostics.input_projection.negative_sample_count ==
          length(samples)
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
    @test fixed.diagnostics.optimizer == :initial_psd_residue_fit
    @test fixed.diagnostics.input_projection.psd_cone_distance < 1e-12
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
    @test zero_fit.diagnostics.optimization_skip_reason == :zero_psd_projection

    nonhermitian = copy(samples)
    nonhermitian[1] = ComplexF64[1 0.2im; 0.2im 1]
    inconsistent = copy(samples)
    inconsistent[1] = Matrix{ComplexF64}(I, 3, 3)
    @test_throws ArgumentError lorentzian_fit(
        nonhermitian, frequencies; n_poles=2, warn_experimental=false)
    @test_throws DimensionMismatch lorentzian_fit(
        inconsistent, frequencies; n_poles=2, warn_experimental=false)

    projection_center = 0.15
    projection_width = 0.31
    projection_kernel = spectral_density(
        LorentzianPSD(
            [projection_center], [projection_width], [1.0], (;)),
        frequencies,
    )
    raw_residue = ComplexF64[1.0 0.2im; -0.2im -1.0]
    decomposition = eigen(Hermitian(raw_residue))
    projected_residue = decomposition.vectors *
        Diagonal(max.(decomposition.values, 0.0)) * decomposition.vectors'
    indefinite_samples = [value .* raw_residue for value in projection_kernel]
    preserved_indefinite_samples = deepcopy(indefinite_samples)
    projected_fit = lorentzian_fit(
        indefinite_samples, frequencies; n_poles=1, maxiter=0,
        initial_centers=[projection_center],
        initial_widths=[projection_width], warn_experimental=false)
    @test projected_fit.residues[1] ≈ projected_residue atol=1e-11
    @test indefinite_samples == preserved_indefinite_samples
    @test projected_fit.diagnostics.input_projection.minimum_eigenvalue < 0
    @test projected_fit.diagnostics.input_projection.negative_eigenvalue_count ==
          length(frequencies)
    @test projected_fit.diagnostics.input_projection.violating_sample_count ==
          length(frequencies)
    @test projected_fit.diagnostics.input_projection.psd_cone_distance > 0
    @test projected_fit.diagnostics.projection ==
          :lorentzian_constrained_least_squares
    @test projected_fit.diagnostics.loss == :unweighted_grid_frobenius
    @test all(eigmin(Hermitian(value)) >= -1e-12 for value in
              spectral_density(projected_fit, range(-8.0, 8.0; length=801)))
    reversed_projected_fit = lorentzian_fit(
        reverse(indefinite_samples), reverse(frequencies); n_poles=1, maxiter=0,
        initial_centers=[projection_center],
        initial_widths=[projection_width], warn_experimental=false)
    @test reversed_projected_fit.residues[1] ≈
          projected_fit.residues[1] atol=1e-11

    nearly_hermitian = deepcopy(indefinite_samples)
    nearly_hermitian[1][1, 2] += 1e-12im
    nearly_hermitian_fit = lorentzian_fit(
        nearly_hermitian, frequencies; n_poles=1, maxiter=0,
        initial_centers=[projection_center],
        initial_widths=[projection_width], warn_experimental=false)
    @test nearly_hermitian_fit.diagnostics.input_projection.hermitianization_distance > 0

    noncommuting_residue_a = ComplexF64[1.0 0.8; 0.8 -0.4]
    noncommuting_residue_b = ComplexF64[-0.3 0.6im; -0.6im 0.9]
    kernel_a = spectral_density(
        LorentzianPSD([centers[1]], [widths[1]], [1.0], (;)), frequencies)
    kernel_b = spectral_density(
        LorentzianPSD([centers[2]], [widths[2]], [1.0], (;)), frequencies)
    nonseparable_samples = [
        kernel_a[n] .* noncommuting_residue_a .+
        kernel_b[n] .* noncommuting_residue_b
        for n in eachindex(frequencies)
    ]
    _, hermitian_samples, positive_samples, _ =
        GraftImpurity._lorentzian_matrix_samples(nonseparable_samples, frequencies)
    fixed_kernel = hcat(kernel_a, kernel_b)
    positive_initialization = GraftImpurity._matrix_lorentzian_initial_residues(
        fixed_kernel, positive_samples, 0.0)
    raw_initialization = GraftImpurity._matrix_lorentzian_initial_residues(
        fixed_kernel, hermitian_samples, 0.0)
    @test sum(norm(positive_initialization[j] - raw_initialization[j])
              for j in eachindex(positive_initialization)) > 1e-3
    nonseparable_fit = lorentzian_fit(
        nonseparable_samples, frequencies; n_poles=2, maxiter=0,
        initial_centers=centers, initial_widths=widths,
        warn_experimental=false)
    @test nonseparable_fit.residues ≈ positive_initialization atol=1e-11

    rank_one_samples = [
        kernel_a[n] .* ComplexF64[1 0; 0 0] for n in eachindex(frequencies)
    ]
    activated_residue = only(GraftImpurity._matrix_lorentzian_initial_residues(
        reshape(kernel_a, :, 1), rank_one_samples, 1e-6))
    @test eigmin(Hermitian(activated_residue)) > 0

    negative_matrix_fit = lorentzian_fit(
        [-value .* Matrix{ComplexF64}(I, 2, 2) for value in projection_kernel],
        frequencies; n_poles=1, warn_experimental=false)
    @test all(iszero, negative_matrix_fit.residues[1])
    @test negative_matrix_fit.diagnostics.optimization_skip_reason ==
          :zero_psd_projection

    nonfinite = copy(samples)
    nonfinite[1] = ComplexF64[NaN 0; 0 1]
    @test_throws ArgumentError lorentzian_fit(
        nonfinite, frequencies; n_poles=2, warn_experimental=false)
    @test_throws ArgumentError MatrixLorentzianPSD(
        [0.0], [0.2], [ComplexF64[1 2im; -2im 1]], (;))
    @test_throws ArgumentError MatrixLorentzianPSD(
        [0.0], [1e-200], [Matrix{ComplexF64}(I, 2, 2)], (;))
    narrow = sqrt(floatmin(Float64))
    narrow_model = MatrixLorentzianPSD(
        [0.0], [narrow], [Matrix{ComplexF64}(I, 2, 2)], (;))
    @test all(isfinite, spectral_density(narrow_model, 0.0))
end
