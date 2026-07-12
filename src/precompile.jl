import PrecompileTools

PrecompileTools.@setup_workload begin
    poles = [-0.8, 0.2, 1.1]
    frequencies = im .* (collect(1:8) .* (π / 10))
    scalar_weights = [0.4, 0.7, 0.2]
    scalar_values = [
        sum(scalar_weights[k] / (z - poles[k]) for k in eachindex(poles))
        for z in frequencies
    ]
    coupling_vectors = [
        ComplexF64[0.7, 0.2im],
        ComplexF64[0.3 + 0.1im, 0.8],
        ComplexF64[0.4, -0.2 + 0.3im],
    ]
    matrix_weights = [v * v' for v in coupling_vectors]
    matrix_values = [
        sum(matrix_weights[k] ./ (z - poles[k]) for k in eachindex(poles))
        for z in frequencies
    ]
    scalar_matrices = [reshape(ComplexF64[value], 1, 1) for value in scalar_values]

    PrecompileTools.@compile_workload begin
        scalar_fit = pes_fit(scalar_values, frequencies;
                             n_poles=3, solver=:sdp, maxiter=0)
        evaluate_poles(scalar_fit, frequencies)
        bath_orbitals(scalar_fit)

        matrix_fit = pes_fit(matrix_values, frequencies;
                             n_poles=3, solver=:sdp, maxiter=0)
        evaluate_poles(matrix_fit, frequencies)
        bath_orbitals(matrix_fit)

        signed_values = [
            sum((k == 2 ? -scalar_weights[k] : scalar_weights[k]) /
                (z - poles[k]) for k in eachindex(poles))
            for z in frequencies
        ]
        pes_fit(signed_values, frequencies;
                n_poles=3, solver=:least_squares, maxiter=0,
                conic_diagnostic=:distance)

        _pes_refine_poles(poles, frequencies, scalar_matrices,
                          :least_squares, :fermion, 1)

        real_frequencies = collect(range(-2.0, 2.0; length=64))
        lorentzian_values = [
            0.4 * 0.3 / (pi * ((omega - 0.2)^2 + 0.3^2))
            for omega in real_frequencies
        ]
        lorentzian = lorentzian_fit(
            lorentzian_values, real_frequencies; n_poles=1, maxiter=1,
            initial_centers=[0.0], initial_widths=[0.4],
            warn_experimental=false)
        spectral_density(lorentzian, real_frequencies)
        complex_poles(lorentzian)

        matrix_vectors = [ComplexF64[0.7, 0.2im],
                          ComplexF64[0.3 + 0.1im, 0.6]]
        matrix_residues = [vector * vector' for vector in matrix_vectors]
        matrix_lorentzian = MatrixLorentzianPSD(
            [-0.4, 0.6], [0.25, 0.35], matrix_residues,
            (; source=:precompile))
        matrix_lorentzian_values = spectral_density(
            matrix_lorentzian, real_frequencies)
        fitted_matrix_lorentzian = lorentzian_fit(
            matrix_lorentzian_values, real_frequencies; n_poles=2,
            maxiter=1, initial_centers=[-0.4, 0.6],
            initial_widths=[0.25, 0.35], warn_experimental=false)
        spectral_density(fitted_matrix_lorentzian, real_frequencies)
        complex_poles(fitted_matrix_lorentzian)
    end
end
