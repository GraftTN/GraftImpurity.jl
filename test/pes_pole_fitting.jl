using LinearAlgebra: Hermitian, eigmin, norm

fermion_kernel(z, pole) = inv(z - pole)
boson_kernel(z, pole) = iszero(z) ? -1.0 + 0im : pole / (z - pole)

function scalar_samples(poles, weights, frequencies, kernel)
    return [sum(weights[k] * kernel(z, poles[k]) for k in eachindex(poles))
            for z in frequencies]
end

function matrix_samples(poles, weights, frequencies, kernel)
    return [sum(kernel(z, poles[k]) .* weights[k] for k in eachindex(poles))
            for z in frequencies]
end

@testset "PES pole fitting" begin
    β = 20.0
    frequencies = im .* (vcat(collect(-24:-1), collect(1:24)) .* π / β)
    holdout = im .* (vcat(collect(-48:-1), collect(1:48)) .* π / (2β))
    poles = [-1.0, 0.2, 1.3]
    scalar_weights = [0.4, 0.7, 0.2]
    values = scalar_samples(poles, scalar_weights, frequencies, fermion_kernel)

    fit = pes_fit(values, frequencies; n_poles=3, solver=:least_squares,
                  maxiter=5)
    @test fit isa PESPoleFit
    @test fit.residue_constraint == :unconstrained
    @test fit.diagnostics.sdp_solves == 0
    @test fit.diagnostics.nnls_solves == 0
    @test fit.diagnostics.conic_diagnostic === nothing
    @test length(fit) == 3
    @test issorted(fit.poles)
    @test all(isfinite, fit.poles)
    @test all(weight -> eigmin(Hermitian(weight)) >= -1e-10, fit.weights)
    expected_holdout = scalar_samples(poles, scalar_weights, holdout, fermion_kernel)
    fitted_holdout = evaluate_poles(fit, holdout)
    @test maximum(abs(expected_holdout[n] - fitted_holdout[n][1, 1])
                  for n in eachindex(holdout)) < 1e-8
    @test fit.diagnostics.max_abs_error < 1e-8

    positive_frequencies = imag.(frequencies) .> 0
    tolerance_fit = pes_fit(values[positive_frequencies],
                            frequencies[positive_frequencies];
                            tolerance=1e-8, solver=:least_squares,
                            min_support=4, max_support=8, maxiter=0)
    @test tolerance_fit.diagnostics.converged
    @test tolerance_fit.diagnostics.attempts == 1

    vectors = [ComplexF64[0.7, 0.2im],
               ComplexF64[0.3 + 0.1im, 0.8],
               ComplexF64[0.4, -0.2 + 0.3im]]
    matrix_weights = [v * adjoint(v) for v in vectors]
    matrix_values = matrix_samples(poles, matrix_weights, frequencies,
                                   fermion_kernel)
    matrix_fit = pes_fit(matrix_values, frequencies; n_poles=3,
                         solver=:sdp, maxiter=0)
    @test matrix_fit.residue_constraint == :psd
    @test all(weight -> norm(weight - weight') < 1e-10, matrix_fit.weights)
    @test all(weight -> eigmin(Hermitian(weight)) >= -1e-10,
              matrix_fit.weights)
    @test matrix_fit.diagnostics.final_solver == :sdp
    @test matrix_fit.diagnostics.sdp_solves == 1
    @test matrix_fit.diagnostics.nnls_solves == 0
    @test matrix_fit.diagnostics.max_abs_error < 2e-6
    @test evaluate_poles(matrix_fit, -frequencies[end]) ≈
          evaluate_poles(matrix_fit, frequencies[end])' atol=2e-6

    scalar_sdp_fit = pes_fit(values, frequencies; n_poles=3, solver=:sdp)
    @test scalar_sdp_fit.residue_constraint == :psd
    @test scalar_sdp_fit.diagnostics.sdp_solves == 0
    @test scalar_sdp_fit.diagnostics.nnls_solves == 1
    @test scalar_sdp_fit.diagnostics.max_abs_error < 1e-8

    orbitals = bath_orbitals(matrix_fit; atol=1e-6)
    reconstructed_weights = [zeros(ComplexF64, 2, 2)
                             for _ in eachindex(matrix_fit.poles)]
    for (coupling, pole_index) in zip(orbitals.couplings, orbitals.pole_indices)
        reconstructed_weights[pole_index] .+= coupling * coupling'
    end
    @test all(norm(reconstructed_weights[k] - matrix_fit.weights[k]) < 2e-6
              for k in eachindex(matrix_fit.weights))

    boson_frequencies = im .* (vcat(collect(-16:-1), [0], collect(1:16)) .* π / β)
    boson_values = scalar_samples(poles, scalar_weights, boson_frequencies,
                                  boson_kernel)
    boson_fit = pes_fit(boson_values, boson_frequencies; n_poles=3,
                        statistics=:boson, solver=:least_squares, maxiter=3)
    @test boson_fit.diagnostics.max_abs_error < 1e-8
    @test evaluate_poles(boson_fit, 0)[1, 1] ≈ -sum(scalar_weights) atol=1e-8

    trial_poles = [-0.8, 0.5, 1.2]
    trial_samples = [reshape(ComplexF64[value], 1, 1) for value in boson_values]
    trial_weights, residual, _ = GraftImpurity._pes_fit_weights(
        trial_poles, boson_frequencies, trial_samples, :least_squares, :boson)
    residual_norm = GraftImpurity._pes_residual_norm(residual)
    analytic = GraftImpurity._pes_pole_gradient(
        trial_poles, boson_frequencies, trial_weights, residual, :boson,
        residual_norm)
    step = 1e-6
    finite_difference = map(eachindex(trial_poles)) do k
        plus = copy(trial_poles); plus[k] += step
        minus = copy(trial_poles); minus[k] -= step
        _, plus_residual, _ = GraftImpurity._pes_fit_weights(
            plus, boson_frequencies, trial_samples, :least_squares, :boson)
        _, minus_residual, _ = GraftImpurity._pes_fit_weights(
            minus, boson_frequencies, trial_samples, :least_squares, :boson)
        (GraftImpurity._pes_residual_norm(plus_residual) -
         GraftImpurity._pes_residual_norm(minus_residual)) / (2step)
    end
    @test analytic ≈ finite_difference rtol=1e-7 atol=1e-9

    orbitals_scalar = bath_orbitals(scalar_sdp_fit)
    @test length(orbitals_scalar.energies) == length(scalar_sdp_fit)
    @test all(isfinite, orbitals_scalar.energies)
    least_squares_orbitals = bath_orbitals(fit)
    @test length(least_squares_orbitals.energies) == length(fit)

    signed_weights = [0.8, -0.35, 0.5]
    signed_values = scalar_samples(poles, signed_weights, frequencies,
                                   fermion_kernel)
    distance_fit = pes_fit(signed_values, frequencies; n_poles=3,
                           solver=:least_squares, maxiter=0,
                           conic_diagnostic=:distance)
    distance_report = distance_fit.diagnostics.conic_diagnostic
    @test distance_fit.residue_constraint == :unconstrained
    @test distance_fit.diagnostics.sdp_solves == 0
    @test distance_fit.diagnostics.nnls_solves == 0
    @test distance_report.mode == :distance
    @test distance_report.projection.minimum_eigenvalue < -0.3
    @test distance_report.projection.frobenius_distance > 0.3
    @test distance_report.projection.negative_eigenvalue_count >= 1
    @test distance_report.projection.violating_pole_count >= 1
    @test distance_report.projection.worst_pole_index in eachindex(distance_fit.poles)
    @test distance_fit.diagnostics.max_abs_error < 1e-8
    @test_throws ArgumentError bath_orbitals(distance_fit)

    mutated_fit = deepcopy(matrix_fit)
    mutated_fit.weights[1][1, 1] = -10
    @test_throws ArgumentError bath_orbitals(mutated_fit)

    @test_throws ArgumentError pes_fit(values, frequencies)
    @test_throws ArgumentError pes_fit(values, frequencies;
                                       tolerance=1e-6, n_poles=3)
    @test_throws ArgumentError pes_fit(values, frequencies;
                                       n_poles=3, statistics=:classical)
    @test_throws ArgumentError pes_fit(values, frequencies;
                                       n_poles=3, solver=:sdp,
                                       conic_diagnostic=:distance)
    @test_throws ArgumentError pes_fit(values, frequencies;
                                       n_poles=3, solver=:least_squares,
                                       conic_diagnostic=:solve)
    @test_throws ArgumentError pes_fit(values, -abs.(imag.(frequencies));
                                       n_poles=3)
end
