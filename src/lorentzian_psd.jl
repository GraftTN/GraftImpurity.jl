const _LORENTZIAN_MIN_WIDTH = sqrt(floatmin(Float64))

"""
    LorentzianPSD(centers, widths, weights, diagnostics)

Highly experimental scalar representation of a real, nonnegative continuous
spectrum,

```math
A(\\omega) = \\sum_j \\frac{w_j \\gamma_j / \\pi}
{(\\omega-\\epsilon_j)^2 + \\gamma_j^2},
```

with real centers `epsilon_j`, strictly positive widths `gamma_j`, and
nonnegative weights `w_j`. These parameter constraints prove pointwise
nonnegativity of the represented spectrum. They do **not** provide a theorem
that a finite Lorentzian mixture uniquely or uniformly reconstructs an
arbitrary continuous spectrum from sampled data.

This first experimental surface intentionally accepts only scalar real-axis
spectra. Matrix-valued, complex-valued, and Matsubara fitting are not part of
its current contract.
"""
struct LorentzianPSD{D<:NamedTuple}
    centers::Vector{Float64}
    widths::Vector{Float64}
    weights::Vector{Float64}
    diagnostics::D

    function LorentzianPSD(centers::AbstractVector{<:Real},
                           widths::AbstractVector{<:Real},
                           weights::AbstractVector{<:Real},
                           diagnostics::D) where {D<:NamedTuple}
        length(centers) == length(widths) == length(weights) ||
            throw(ArgumentError("LorentzianPSD needs one width and weight per center"))
        isempty(centers) && throw(ArgumentError("LorentzianPSD needs at least one component"))
        epsilon = Float64.(centers)
        gamma = Float64.(widths)
        residue = Float64.(weights)
        all(isfinite, epsilon) ||
            throw(ArgumentError("LorentzianPSD centers must be finite"))
        all(x -> isfinite(x) && x >= _LORENTZIAN_MIN_WIDTH, gamma) ||
            throw(ArgumentError("LorentzianPSD widths must be finite and at least $(_LORENTZIAN_MIN_WIDTH)"))
        all(x -> isfinite(x) && x >= 0, residue) ||
            throw(ArgumentError("LorentzianPSD weights must be finite and nonnegative"))
        order = sortperm(epsilon)
        return new{D}(epsilon[order], gamma[order], residue[order], diagnostics)
    end
end

Base.length(model::LorentzianPSD) = length(model.centers)

"""
    MatrixLorentzianPSD(centers, widths, residues, diagnostics)

Highly experimental matrix-valued real-axis spectrum

```math
A(\\omega) = \\sum_j \\frac{\\gamma_j / \\pi}
{(\\omega-\\epsilon_j)^2 + \\gamma_j^2} R_j,
\\qquad R_j \\succeq 0.
```

Each residue is a complex Hermitian positive-semidefinite matrix. Because the
Lorentzian kernels are nonnegative on the real axis, this parameterization
guarantees `A(omega)` is Hermitian positive semidefinite at every real
frequency. It does not prove uniqueness, completeness, or global optimality
of a finite-component fit.
"""
struct MatrixLorentzianPSD{D<:NamedTuple}
    centers::Vector{Float64}
    widths::Vector{Float64}
    residues::Vector{Matrix{ComplexF64}}
    diagnostics::D

    function MatrixLorentzianPSD(centers::AbstractVector{<:Real},
                                 widths::AbstractVector{<:Real}, residues,
                                 diagnostics::D) where {D<:NamedTuple}
        length(centers) == length(widths) == length(residues) ||
            throw(ArgumentError("MatrixLorentzianPSD needs one width and residue per center"))
        isempty(centers) &&
            throw(ArgumentError("MatrixLorentzianPSD needs at least one component"))
        epsilon = Float64.(centers)
        gamma = Float64.(widths)
        all(isfinite, epsilon) ||
            throw(ArgumentError("MatrixLorentzianPSD centers must be finite"))
        all(x -> isfinite(x) && x >= _LORENTZIAN_MIN_WIDTH, gamma) ||
            throw(ArgumentError("MatrixLorentzianPSD widths must be finite and at least $(_LORENTZIAN_MIN_WIDTH)"))
        first_size = size(first(residues))
        length(first_size) == 2 && first_size[1] == first_size[2] &&
            first_size[1] > 0 ||
            throw(ArgumentError("MatrixLorentzianPSD residues must be nonempty square matrices"))
        matrices = Matrix{ComplexF64}[]
        for (j, residue) in enumerate(residues)
            size(residue) == first_size ||
                throw(DimensionMismatch("MatrixLorentzianPSD residue $j has an inconsistent size"))
            push!(matrices, _lorentzian_validated_psd(
                Matrix{ComplexF64}(residue), j, "MatrixLorentzianPSD residue"))
        end
        order = sortperm(epsilon)
        return new{D}(epsilon[order], gamma[order], matrices[order], diagnostics)
    end
end

Base.length(model::MatrixLorentzianPSD) = length(model.centers)

"""Return the lower-half-plane retarded poles `epsilon_j - im * gamma_j`."""
complex_poles(model::LorentzianPSD) =
    ComplexF64.(model.centers) .- im .* model.widths
complex_poles(model::MatrixLorentzianPSD) =
    ComplexF64.(model.centers) .- im .* model.widths

"""Evaluate a scalar `LorentzianPSD` spectrum at one real frequency."""
function spectral_density(model::LorentzianPSD, frequency::Real)
    omega = Float64(frequency)
    isfinite(omega) || throw(ArgumentError("spectral frequency must be finite"))
    value = 0.0
    for j in eachindex(model.centers)
        kernel, _, _ = _lorentzian_basis(
            omega, model.centers[j], model.widths[j])
        value += model.weights[j] * kernel
    end
    isfinite(value) ||
        throw(OverflowError("Lorentzian spectral density overflowed Float64"))
    return value
end

"""Evaluate a scalar `LorentzianPSD` spectrum on real frequencies."""
spectral_density(model::LorentzianPSD,
                 frequencies::AbstractVector{<:Real}) =
    [spectral_density(model, frequency) for frequency in frequencies]

"""Evaluate a `MatrixLorentzianPSD` spectrum at one real frequency."""
function spectral_density(model::MatrixLorentzianPSD, frequency::Real)
    omega = Float64(frequency)
    isfinite(omega) || throw(ArgumentError("spectral frequency must be finite"))
    dimension = size(first(model.residues), 1)
    value = zeros(ComplexF64, dimension, dimension)
    for j in eachindex(model.centers)
        kernel, _, _ = _lorentzian_basis(
            omega, model.centers[j], model.widths[j])
        value .+= kernel .* model.residues[j]
    end
    all(z -> isfinite(real(z)) && isfinite(imag(z)), value) ||
        throw(OverflowError("matrix Lorentzian spectral density overflowed Float64"))
    return Matrix{ComplexF64}((value + value') / 2)
end

"""Evaluate a `MatrixLorentzianPSD` spectrum on real frequencies."""
spectral_density(model::MatrixLorentzianPSD,
                 frequencies::AbstractVector{<:Real}) =
    [spectral_density(model, frequency) for frequency in frequencies]

"""
    lorentzian_fit(values, frequencies; n_poles, minimum_width=nothing,
                   maxiter=300,
                   initial_centers=nothing, initial_widths=nothing,
                   warn_experimental=true) -> LorentzianPSD

Fit sampled scalar, real, nonnegative spectral data with a positive
Lorentzian mixture. Centers, square-root widths, and square-root weights are
optimized jointly with `Optim.LBFGS`; the squared parameterization guarantees
positive widths and nonnegative weights without an SDP or a global
frequency-grid positivity constraint.

!!! warning "Highly experimental"
    Pointwise nonnegativity follows exactly from the parameterization, but no
    rigorous convergence, uniqueness, finite-component completeness, or
    off-grid reconstruction-error theorem is currently claimed. The fit is a
    local nonconvex optimization and can depend on initialization.

Only real-axis scalar inputs are accepted in this first version.
"""
function lorentzian_fit(values::AbstractVector,
                        frequencies::AbstractVector;
                        n_poles::Integer,
                        minimum_width::Union{Nothing,Real}=nothing,
                        maxiter::Integer=300,
                        initial_centers::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        initial_widths::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        warn_experimental::Bool=true)
    warn_experimental && @warn(
        "lorentzian_fit is highly experimental: positivity is structural, " *
        "but convergence, uniqueness, finite-mixture completeness, and off-grid " *
        "reconstruction accuracy have no rigorous guarantee")
    n_poles > 0 || throw(ArgumentError("n_poles must be positive"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    length(values) == length(frequencies) ||
        throw(DimensionMismatch("lorentzian_fit needs one value per frequency"))
    length(values) >= 3 ||
        throw(ArgumentError("lorentzian_fit needs at least three real-axis samples"))
    all(value -> value isa Real, values) ||
        throw(ArgumentError("lorentzian_fit currently accepts only real-valued spectra"))
    all(frequency -> frequency isa Real, frequencies) ||
        throw(ArgumentError("lorentzian_fit currently accepts only real-axis frequencies"))

    started = time_ns()
    omega = Float64.(frequencies)
    target = Float64.(values)
    all(isfinite, omega) ||
        throw(ArgumentError("Lorentzian frequencies must be finite"))
    all(isfinite, target) ||
        throw(ArgumentError("Lorentzian samples must be finite"))
    length(unique(omega)) == length(omega) ||
        throw(ArgumentError("Lorentzian frequencies must be distinct"))
    scale = maximum(abs, target; init=0.0)
    negativity_tolerance = 100eps(Float64) * scale
    minimum(target; init=0.0) >= -negativity_tolerance ||
        throw(ArgumentError("LorentzianPSD fitting requires a nonnegative real spectrum"))
    clipped_negative_sample_count = count(<(0.0), target)
    target = max.(target, 0.0)
    order = sortperm(omega)
    omega = omega[order]
    target = target[order]
    span = last(omega) - first(omega)
    span > 0 || throw(ArgumentError("Lorentzian frequency interval must have positive width"))
    grid_resolution = minimum(diff(omega))
    width_floor = minimum_width === nothing ? grid_resolution / 2 :
                  Float64(minimum_width)
    isfinite(width_floor) && width_floor >= _LORENTZIAN_MIN_WIDTH ||
        throw(ArgumentError("minimum_width must be finite and at least $(_LORENTZIAN_MIN_WIDTH)"))
    width_floor <= span ||
        throw(ArgumentError("minimum_width must not exceed the fit interval"))

    centers = _lorentzian_initial_centers(
        omega, target, Int(n_poles), initial_centers)
    widths = _lorentzian_initial_widths(
        omega, centers, Int(n_poles), initial_widths, width_floor)
    kernel = _lorentzian_kernel(omega, centers, widths)
    nnls_started = time_ns()
    weights, nnls_iterations = _nnls(kernel, target)
    nnls_seconds = (time_ns() - nnls_started) / 1e9

    width_parameters = sqrt.(max.(widths .- width_floor, eps(Float64)))
    weight_scale = max(sum(target) * span / length(target), scale * span, 1.0)
    weight_parameters = iszero(scale) ? zeros(Float64, Int(n_poles)) :
        sqrt.(max.(weights, eps(Float64) * weight_scale))
    parameters = vcat(centers, width_parameters, weight_parameters)
    objective_scale = max(norm(target), eps(Float64))
    objective, gradient! = _lorentzian_objective(
        omega, target, Int(n_poles), width_floor, objective_scale)

    optimization_started = time_ns()
    result = if maxiter == 0 || iszero(scale)
        nothing
    else
        options = Optim.Options(iterations=Int(maxiter), g_tol=1e-10,
                                f_reltol=1e-12, show_trace=false,
                                store_trace=false)
        method = Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking())
        Optim.optimize(objective, gradient!, parameters, method, options)
    end
    optimization_seconds = (time_ns() - optimization_started) / 1e9
    fitted_parameters = result === nothing ? parameters : Optim.minimizer(result)
    fitted_centers, fitted_widths, fitted_weights =
        _lorentzian_unpack(fitted_parameters, Int(n_poles), width_floor)
    all(isfinite, fitted_parameters) ||
        throw(ErrorException("Optim returned non-finite Lorentzian parameters"))

    fitted = _lorentzian_kernel(omega, fitted_centers, fitted_widths) *
             fitted_weights
    residual = fitted - target
    diagnostics = (;
        experimental = true,
        proof_status = :parameterization_only,
        requested_n_poles = Int(n_poles),
        pole_count = Int(n_poles),
        converged = result === nothing ? nothing : Optim.converged(result),
        iterations = result === nothing ? 0 : Optim.iterations(result),
        objective = sum(abs2, residual) / (2objective_scale^2),
        residual_norm = norm(residual),
        relative_l2_error = norm(residual) / objective_scale,
        max_abs_error = maximum(abs, residual),
        fit_domain = (first(omega), last(omega)),
        minimum_width = width_floor,
        grid_resolution,
        spectral_weight = sum(fitted_weights),
        clipped_negative_sample_count,
        initial_nnls_iterations = nnls_iterations,
        positivity = :structural,
        rigorous_completeness_proof = false,
        optimizer = result === nothing ? :initial_nnls : :lbfgs,
        optimization_skipped = result === nothing,
        optimization_skip_reason = if result !== nothing
            nothing
        elseif maxiter == 0
            :maxiter_zero
        else
            :zero_spectrum
        end,
        timings = (;
            initial_nnls_seconds = nnls_seconds,
            optimization_seconds,
            total_seconds = (time_ns() - started) / 1e9,
        ),
    )
    return LorentzianPSD(fitted_centers, fitted_widths, fitted_weights,
                         diagnostics)
end

"""
    lorentzian_fit(values::AbstractVector{<:AbstractMatrix}, frequencies;
                   n_poles, minimum_width=nothing, maxiter=300,
                   initial_centers=nothing, initial_widths=nothing,
                   warn_experimental=true) -> MatrixLorentzianPSD

Fit a real-axis Hermitian PSD matrix spectrum with shared Lorentzian centers
and widths. The trace spectrum initializes the shared poles. Full complex
matrix residues are then represented as `R_j = B_j * B_j'` and optimized
together with the centers and widths by `Optim.LBFGS`. This factorization
guarantees matrix positivity without an SDP.

The fit is highly experimental and nonconvex. Structural positivity is exact,
but no global-optimality, uniqueness, finite-mixture completeness, or off-grid
error theorem is claimed.
"""
function lorentzian_fit(values::AbstractVector{<:AbstractMatrix},
                        frequencies::AbstractVector;
                        n_poles::Integer,
                        minimum_width::Union{Nothing,Real}=nothing,
                        maxiter::Integer=300,
                        initial_centers::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        initial_widths::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        warn_experimental::Bool=true)
    warn_experimental && @warn(
        "matrix lorentzian_fit is highly experimental: PSD is structural, " *
        "but convergence, uniqueness, finite-mixture completeness, and off-grid " *
        "reconstruction accuracy have no rigorous guarantee")
    n_poles > 0 || throw(ArgumentError("n_poles must be positive"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    started = time_ns()
    omega, samples, clipped_sample_count =
        _lorentzian_matrix_samples(values, frequencies)
    dimension = size(first(samples), 1)
    trace_target = [sum(real(samples[n][i, i]) for i in 1:dimension)
                    for n in eachindex(samples)]

    trace_fit = lorentzian_fit(
        trace_target, omega; n_poles, minimum_width, maxiter,
        initial_centers, initial_widths, warn_experimental=false)
    centers = copy(trace_fit.centers)
    widths = copy(trace_fit.widths)
    width_floor = trace_fit.diagnostics.minimum_width
    kernel = _lorentzian_kernel(omega, centers, widths)
    initial_residues = _matrix_lorentzian_initial_residues(
        kernel, samples, trace_fit.weights)
    factors = _matrix_lorentzian_factors(initial_residues)
    parameters = _matrix_lorentzian_pack(
        centers, widths, factors, width_floor)
    objective_scale = max(_matrix_lorentzian_norm(samples), eps(Float64))
    objective, gradient! = _matrix_lorentzian_objective(
        omega, samples, Int(n_poles), dimension, width_floor,
        objective_scale)

    optimization_started = time_ns()
    zero_spectrum = iszero(maximum(trace_target; init=0.0))
    result = if maxiter == 0 || zero_spectrum
        nothing
    else
        options = Optim.Options(iterations=Int(maxiter), g_tol=1e-10,
                                f_reltol=1e-12, show_trace=false,
                                store_trace=false)
        method = Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking())
        Optim.optimize(objective, gradient!, parameters, method, options)
    end
    optimization_seconds = (time_ns() - optimization_started) / 1e9
    fitted_parameters = result === nothing ? parameters : Optim.minimizer(result)
    all(isfinite, fitted_parameters) ||
        throw(ErrorException("Optim returned non-finite matrix Lorentzian parameters"))
    fitted_centers, fitted_widths, fitted_factors =
        _matrix_lorentzian_unpack(fitted_parameters, Int(n_poles), dimension,
                                  width_floor)
    fitted_residues = [factor * factor' for factor in fitted_factors]
    fitted_kernel = _lorentzian_kernel(omega, fitted_centers, fitted_widths)
    reconstructed = _matrix_lorentzian_evaluate(
        fitted_kernel, fitted_residues)
    residual_norm = sqrt(sum(sum(abs2, reconstructed[n] - samples[n])
                             for n in eachindex(samples)))
    max_abs_error = maximum(maximum(abs, reconstructed[n] - samples[n])
                            for n in eachindex(samples))
    minimum_eigenvalue = minimum(minimum(eigvals(Hermitian(value)))
                                 for value in reconstructed)
    diagnostics = (;
        experimental = true,
        proof_status = :parameterization_only,
        rigorous_completeness_proof = false,
        requested_n_poles = Int(n_poles),
        pole_count = Int(n_poles),
        matrix_dimension = dimension,
        converged = result === nothing ? nothing : Optim.converged(result),
        iterations = result === nothing ? 0 : Optim.iterations(result),
        objective = residual_norm^2 / (2objective_scale^2),
        residual_norm,
        relative_l2_error = residual_norm / objective_scale,
        max_abs_error,
        fit_domain = (first(omega), last(omega)),
        minimum_width = width_floor,
        grid_resolution = trace_fit.diagnostics.grid_resolution,
        residue_traces = real.(tr.(fitted_residues)),
        total_spectral_weight = sum(fitted_residues),
        minimum_reconstructed_eigenvalue = minimum_eigenvalue,
        clipped_input_sample_count = clipped_sample_count,
        positivity = :structural_residue_factorization,
        optimizer = result === nothing ? :initial_psd_projection : :lbfgs,
        optimization_skipped = result === nothing,
        optimization_skip_reason = if result !== nothing
            nothing
        elseif maxiter == 0
            :maxiter_zero
        else
            :zero_spectrum
        end,
        trace_initialization = trace_fit.diagnostics,
        sdp_solves = 0,
        timings = (;
            trace_fit_seconds = trace_fit.diagnostics.timings.total_seconds,
            matrix_optimization_seconds = optimization_seconds,
            total_seconds = (time_ns() - started) / 1e9,
        ),
    )
    return MatrixLorentzianPSD(
        fitted_centers, fitted_widths, fitted_residues, diagnostics)
end

function _lorentzian_initial_centers(omega, target, n_poles::Int, supplied)
    if supplied !== nothing
        length(supplied) == n_poles ||
            throw(DimensionMismatch("initial_centers must contain n_poles entries"))
        centers = Float64.(supplied)
        all(isfinite, centers) ||
            throw(ArgumentError("initial Lorentzian centers must be finite"))
        return centers
    end
    quadrature = similar(omega)
    quadrature[1] = (omega[2] - omega[1]) / 2
    quadrature[end] = (omega[end] - omega[end - 1]) / 2
    for n in 2:length(omega) - 1
        quadrature[n] = (omega[n + 1] - omega[n - 1]) / 2
    end
    cumulative = cumsum(target .* quadrature)
    total = last(cumulative)
    if iszero(total)
        edges = range(first(omega), last(omega); length=n_poles + 1)
        return [(edges[j] + edges[j + 1]) / 2 for j in 1:n_poles]
    end
    return [omega[searchsortedfirst(cumulative, (j - 0.5) * total / n_poles)]
            for j in 1:n_poles]
end

function _lorentzian_initial_widths(omega, centers, n_poles::Int, supplied,
                                    minimum_width::Float64)
    if supplied !== nothing
        length(supplied) == n_poles ||
            throw(DimensionMismatch("initial_widths must contain n_poles entries"))
        widths = Float64.(supplied)
        all(x -> isfinite(x) && x > 0, widths) ||
            throw(ArgumentError("initial Lorentzian widths must be finite and positive"))
        all(>=(minimum_width), widths) ||
            throw(ArgumentError("initial Lorentzian widths must be at least minimum_width"))
        return widths
    end
    if n_poles == 1
        return [max((last(omega) - first(omega)) / 10, minimum_width)]
    end
    order = sortperm(centers)
    sorted = centers[order]
    sorted_widths = map(eachindex(sorted)) do j
        left = j == 1 ? Inf : sorted[j] - sorted[j - 1]
        right = j == length(sorted) ? Inf : sorted[j + 1] - sorted[j]
        max(min(left, right) / 2, minimum_width)
    end
    widths = similar(sorted_widths)
    widths[order] = sorted_widths
    return widths
end

function _lorentzian_kernel(omega, centers, widths)
    return [first(_lorentzian_basis(omega[n], centers[j], widths[j]))
            for n in eachindex(omega), j in eachindex(centers)]
end

function _lorentzian_basis(omega::Real, center::Real, gamma::Real)
    offset = Float64(omega) - Float64(center)
    isfinite(offset) || return (0.0, 0.0, 0.0)
    width = Float64(gamma)
    scale = max(abs(offset), width)
    scaled_offset = offset / scale
    scaled_width = width / scale
    denominator = scaled_offset^2 + scaled_width^2
    kernel = scaled_width / (pi * scale * denominator)
    derivative_scale = inv(scale * scale) / (pi * denominator^2)
    center_derivative =
        2scaled_width * scaled_offset * derivative_scale
    width_derivative =
        (scaled_offset^2 - scaled_width^2) * derivative_scale
    return kernel, center_derivative, width_derivative
end

function _lorentzian_unpack(parameters, n_poles::Int, width_floor::Float64)
    centers = collect(@view parameters[1:n_poles])
    width_parameters = @view parameters[n_poles + 1:2n_poles]
    weight_parameters = @view parameters[2n_poles + 1:3n_poles]
    widths = width_floor .+ width_parameters .^ 2
    weights = weight_parameters .^ 2
    return centers, widths, weights
end

function _lorentzian_objective(omega, target, n_poles::Int,
                               width_floor::Float64, scale::Float64)
    inverse_scale_squared = inv(scale * scale)
    function objective(parameters)
        centers, widths, weights =
            _lorentzian_unpack(parameters, n_poles, width_floor)
        fitted = _lorentzian_kernel(omega, centers, widths) * weights
        return sum(abs2, fitted - target) * inverse_scale_squared / 2
    end
    function gradient!(storage, parameters)
        centers, widths, weights =
            _lorentzian_unpack(parameters, n_poles, width_floor)
        width_parameters = @view parameters[n_poles + 1:2n_poles]
        weight_parameters = @view parameters[2n_poles + 1:3n_poles]
        kernel = _lorentzian_kernel(omega, centers, widths)
        residual = kernel * weights - target
        fill!(storage, 0.0)
        for j in 1:n_poles
            center_derivative = 0.0
            width_derivative = 0.0
            weight_derivative = 0.0
            center = centers[j]
            gamma = widths[j]
            weight = weights[j]
            for n in eachindex(omega)
                _, kernel_center_derivative, kernel_width_derivative =
                    _lorentzian_basis(omega[n], center, gamma)
                common = residual[n] * inverse_scale_squared
                center_derivative +=
                    common * weight * kernel_center_derivative
                width_derivative +=
                    common * weight * kernel_width_derivative
                weight_derivative += residual[n] * inverse_scale_squared *
                    kernel[n, j]
            end
            storage[j] = center_derivative
            storage[n_poles + j] =
                2width_parameters[j] * width_derivative
            storage[2n_poles + j] =
                2weight_parameters[j] * weight_derivative
        end
        return storage
    end
    return objective, gradient!
end

function _lorentzian_validated_psd(matrix::Matrix{ComplexF64}, index::Int,
                                   label::AbstractString)
    all(z -> isfinite(real(z)) && isfinite(imag(z)), matrix) ||
        throw(ArgumentError("$label $index must be finite"))
    scale = max(norm(matrix), 1.0)
    norm(matrix - matrix') <= 100sqrt(eps(Float64)) * scale ||
        throw(ArgumentError("$label $index must be Hermitian"))
    hermitian = Matrix{ComplexF64}((matrix + matrix') / 2)
    decomposition = eigen(Hermitian(hermitian))
    minimum(decomposition.values) >= -100eps(Float64) * scale ||
        throw(ArgumentError("$label $index must be positive semidefinite"))
    eigenvalues = max.(Float64.(decomposition.values), 0.0)
    projected = decomposition.vectors *
                Diagonal(eigenvalues) * decomposition.vectors'
    return Matrix{ComplexF64}((projected + projected') / 2)
end

function _lorentzian_matrix_samples(values, frequencies)
    length(values) == length(frequencies) ||
        throw(DimensionMismatch("matrix lorentzian_fit needs one value per frequency"))
    length(values) >= 3 ||
        throw(ArgumentError("matrix lorentzian_fit needs at least three real-axis samples"))
    all(frequency -> frequency isa Real, frequencies) ||
        throw(ArgumentError("matrix lorentzian_fit currently accepts only real-axis frequencies"))
    omega = Float64.(frequencies)
    all(isfinite, omega) ||
        throw(ArgumentError("Lorentzian frequencies must be finite"))
    length(unique(omega)) == length(omega) ||
        throw(ArgumentError("Lorentzian frequencies must be distinct"))
    isempty(values) && throw(ArgumentError("matrix Lorentzian samples may not be empty"))
    first_size = size(first(values))
    length(first_size) == 2 && first_size[1] == first_size[2] &&
        first_size[1] > 0 ||
        throw(ArgumentError("matrix Lorentzian samples must be nonempty square matrices"))
    samples = Matrix{ComplexF64}[]
    clipped = 0
    for (n, value) in enumerate(values)
        size(value) == first_size ||
            throw(DimensionMismatch("matrix Lorentzian sample $n has an inconsistent size"))
        matrix = Matrix{ComplexF64}(value)
        scale = max(norm(matrix), 1.0)
        norm(matrix - matrix') <= 100sqrt(eps(Float64)) * scale ||
            throw(ArgumentError("matrix Lorentzian sample $n must be Hermitian"))
        hermitian = Matrix{ComplexF64}((matrix + matrix') / 2)
        decomposition = eigen(Hermitian(hermitian))
        minimum(decomposition.values) >= -100eps(Float64) * scale ||
            throw(ArgumentError("matrix Lorentzian sample $n must be positive semidefinite"))
        clipped += count(<(0.0), decomposition.values)
        eigenvalues = max.(Float64.(decomposition.values), 0.0)
        projected = decomposition.vectors *
                    Diagonal(eigenvalues) * decomposition.vectors'
        push!(samples, Matrix{ComplexF64}((projected + projected') / 2))
    end
    order = sortperm(omega)
    return omega[order], samples[order], clipped
end

function _matrix_lorentzian_initial_residues(kernel, samples, trace_weights)
    nfrequency, n_poles = size(kernel)
    dimension = size(first(samples), 1)
    flattened = Matrix{ComplexF64}(undef, nfrequency, dimension^2)
    for n in 1:nfrequency
        flattened[n, :] .= vec(samples[n])
    end
    coefficients = kernel \ flattened
    total_trace = sum(trace_weights)
    if iszero(total_trace)
        return [zeros(ComplexF64, dimension, dimension) for _ in 1:n_poles]
    end
    activation = sqrt(eps(Float64)) * max(total_trace / n_poles, eps(Float64))
    residues = Matrix{ComplexF64}[]
    for j in 1:n_poles
        raw = reshape(collect(@view coefficients[j, :]), dimension, dimension)
        decomposition = eigen(Hermitian((raw + raw') / 2))
        eigenvalues = max.(Float64.(decomposition.values), 0.0)
        desired_trace = max(trace_weights[j], activation)
        if sum(eigenvalues) > eps(Float64)
            eigenvalues .*= desired_trace / sum(eigenvalues)
            eigenvalues .= max.(eigenvalues,
                                sqrt(eps(Float64)) * desired_trace / dimension)
            eigenvalues .*= desired_trace / sum(eigenvalues)
            residue = decomposition.vectors *
                      Diagonal(eigenvalues) * decomposition.vectors'
        else
            residue = Matrix{ComplexF64}(
                Diagonal(fill(desired_trace / dimension, dimension)))
        end
        push!(residues, Matrix{ComplexF64}((residue + residue') / 2))
    end
    return residues
end

function _matrix_lorentzian_factors(residues)
    return map(residues) do residue
        decomposition = eigen(Hermitian(residue))
        decomposition.vectors *
        Diagonal(sqrt.(max.(Float64.(decomposition.values), 0.0)))
    end
end

function _matrix_lorentzian_pack(centers, widths, factors,
                                 width_floor::Float64)
    parameters = Float64[]
    append!(parameters, centers)
    append!(parameters,
            sqrt.(max.(widths .- width_floor, eps(Float64))))
    for factor in factors
        append!(parameters, vec(real(factor)))
        append!(parameters, vec(imag(factor)))
    end
    return parameters
end

function _matrix_lorentzian_unpack(parameters, n_poles::Int, dimension::Int,
                                   width_floor::Float64)
    centers = collect(@view parameters[1:n_poles])
    width_parameters = @view parameters[n_poles + 1:2n_poles]
    widths = width_floor .+ width_parameters .^ 2
    factors = Matrix{ComplexF64}[]
    block_size = dimension^2
    offset = 2n_poles
    for _ in 1:n_poles
        real_part = reshape(
            collect(@view parameters[offset + 1:offset + block_size]),
            dimension, dimension)
        offset += block_size
        imaginary_part = reshape(
            collect(@view parameters[offset + 1:offset + block_size]),
            dimension, dimension)
        offset += block_size
        push!(factors, ComplexF64.(real_part, imaginary_part))
    end
    return centers, widths, factors
end

function _matrix_lorentzian_evaluate(kernel, residues)
    nfrequency, n_poles = size(kernel)
    dimension = size(first(residues), 1)
    return [begin
        value = zeros(ComplexF64, dimension, dimension)
        for j in 1:n_poles
            value .+= kernel[n, j] .* residues[j]
        end
        Matrix{ComplexF64}((value + value') / 2)
    end for n in 1:nfrequency]
end

_matrix_lorentzian_norm(samples) =
    sqrt(sum(sum(abs2, sample) for sample in samples))

function _matrix_lorentzian_objective(omega, target, n_poles::Int,
                                      dimension::Int,
                                      width_floor::Float64, scale::Float64)
    inverse_scale_squared = inv(scale * scale)
    block_size = dimension^2
    function objective(parameters)
        centers, widths, factors = _matrix_lorentzian_unpack(
            parameters, n_poles, dimension, width_floor)
        residues = [factor * factor' for factor in factors]
        kernel = _lorentzian_kernel(omega, centers, widths)
        reconstructed = _matrix_lorentzian_evaluate(kernel, residues)
        return sum(sum(abs2, reconstructed[n] - target[n])
                   for n in eachindex(target)) * inverse_scale_squared / 2
    end
    function gradient!(storage, parameters)
        centers, widths, factors = _matrix_lorentzian_unpack(
            parameters, n_poles, dimension, width_floor)
        width_parameters = @view parameters[n_poles + 1:2n_poles]
        residues = [factor * factor' for factor in factors]
        kernel = _lorentzian_kernel(omega, centers, widths)
        reconstructed = _matrix_lorentzian_evaluate(kernel, residues)
        residual = [reconstructed[n] - target[n] for n in eachindex(target)]
        fill!(storage, 0.0)
        factor_offset = 2n_poles
        for j in 1:n_poles
            center_derivative = 0.0
            width_derivative = 0.0
            factor_gradient = zeros(ComplexF64, dimension, dimension)
            center = centers[j]
            gamma = widths[j]
            residue = residues[j]
            factor = factors[j]
            for n in eachindex(omega)
                overlap = real(sum(conj.(residual[n]) .* residue)) *
                          inverse_scale_squared
                _, kernel_center_derivative, kernel_width_derivative =
                    _lorentzian_basis(omega[n], center, gamma)
                center_derivative += overlap * kernel_center_derivative
                width_derivative += overlap * kernel_width_derivative
                factor_gradient .+=
                    2kernel[n, j] * inverse_scale_squared .* residual[n] * factor
            end
            storage[j] = center_derivative
            storage[n_poles + j] =
                2width_parameters[j] * width_derivative
            real_range = factor_offset + 1:factor_offset + block_size
            storage[real_range] .= vec(real(factor_gradient))
            factor_offset += block_size
            imaginary_range = factor_offset + 1:factor_offset + block_size
            storage[imaginary_range] .= vec(imag(factor_gradient))
            factor_offset += block_size
        end
        return storage
    end
    return objective, gradient!
end
