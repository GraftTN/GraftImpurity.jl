import Clarabel
import JuMP
import Optim

"""
    PESPoleFit

A real-pole expansion produced by [`pes_fit`](@ref). Its matrix weights are
Hermitian; they are guaranteed positive semidefinite only when
`residue_constraint == :psd`. The expansion represents

```math
F(z) = \\sum_j K(z, \\epsilon_j) R_j,
```

with `K(z, ε) = 1 / (z - ε)` for fermions and
`K(z, ε) = ε / (z - ε)` for bosons.
"""
struct PESPoleFit{D<:NamedTuple}
    poles::Vector{Float64}
    weights::Vector{Matrix{ComplexF64}}
    statistics::Symbol
    residue_constraint::Symbol
    diagnostics::D

    function PESPoleFit(poles::AbstractVector{<:Real}, weights,
                        statistics::Symbol, diagnostics::D;
                        residue_constraint::Symbol=:psd) where {D<:NamedTuple}
        statistics in (:fermion, :boson) ||
            throw(ArgumentError("statistics must be :fermion or :boson"))
        residue_constraint in (:psd, :unconstrained) ||
            throw(ArgumentError("residue_constraint must be :psd or :unconstrained"))
        length(poles) == length(weights) ||
            throw(ArgumentError("PESPoleFit needs one weight matrix per pole"))
        isempty(poles) && throw(ArgumentError("PESPoleFit needs at least one pole"))
        p = Float64.(poles)
        all(isfinite, p) || throw(ArgumentError("PES poles must be finite and real"))
        first_size = size(first(weights))
        first_size[1] == first_size[2] ||
            throw(ArgumentError("PES weight matrices must be square"))
        matrices = Matrix{ComplexF64}[]
        for (j, weight) in enumerate(weights)
            size(weight) == first_size ||
                throw(ArgumentError("PES weight $j has an inconsistent size"))
            matrix = Matrix{ComplexF64}(weight)
            validated = residue_constraint == :psd ?
                _pes_validated_psd(matrix, j) :
                _pes_validated_hermitian(matrix, j)
            push!(matrices, validated)
        end
        order = sortperm(p)
        return new{D}(p[order], matrices[order], statistics,
                      residue_constraint, diagnostics)
    end
end

Base.length(fit::PESPoleFit) = length(fit.poles)

"""
    pes_fit(values, frequencies; tolerance=nothing, n_poles=nothing,
            statistics=:fermion, solver=:sdp, maxiter=0,
            min_support=4, max_support=50, aaa_tolerance=1e-13,
            residue_tolerance=1e-5, conic_diagnostic=:none)

Fit scalar or matrix-valued Matsubara samples with the pole-estimation and
semidefinite-relaxation pipeline used by ADAPOL.  Exactly one of `tolerance`
and `n_poles` must be supplied.  Pole locations are initialized by a shared,
matrix-valued AAA approximation constrained to real poles, then refined with
`Optim.LBFGS`.  `solver=:sdp` fits every residue in a Hermitian PSD cone using
`JuMP` and `Clarabel`; `solver=:least_squares` performs the unconstrained
Hermitian fit and never invokes a conic solver. For least squares,
`conic_diagnostic=:distance` reports the exact Frobenius distance to the
product PSD cone without solving an SDP. The default `:none` adds no
diagnostic cost.

`frequencies` may contain pure-imaginary Matsubara points or real Matsubara
frequency magnitudes.  `values` may be a scalar vector, a vector of square
matrices, or an `n_frequency × n_orbital × n_orbital` array.
"""
function pes_fit(values, frequencies;
                 tolerance::Union{Nothing,Real}=nothing,
                 n_poles::Union{Nothing,Integer}=nothing,
                 statistics::Symbol=:fermion,
                 solver::Symbol=:sdp,
                 maxiter::Integer=0,
                 min_support::Integer=4,
                 max_support::Integer=50,
                 aaa_tolerance::Real=1e-13,
                 residue_tolerance::Real=1e-5,
                 conic_diagnostic::Symbol=:none)
    xor(tolerance === nothing, n_poles === nothing) ||
        throw(ArgumentError("specify exactly one of tolerance and n_poles"))
    statistics in (:fermion, :boson) ||
        throw(ArgumentError("statistics must be :fermion or :boson"))
    solver in (:sdp, :least_squares, :lstsq) ||
        throw(ArgumentError("solver must be :sdp or :least_squares"))
    conic_diagnostic in (:none, :distance) ||
        throw(ArgumentError("conic_diagnostic must be :none or :distance"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    min_support >= 2 || throw(ArgumentError("min_support must be at least 2"))
    max_support >= min_support ||
        throw(ArgumentError("max_support must not be smaller than min_support"))
    aaa_tolerance > 0 || throw(ArgumentError("aaa_tolerance must be positive"))
    residue_tolerance >= 0 ||
        throw(ArgumentError("residue_tolerance must be nonnegative"))
    tolerance === nothing || tolerance > 0 ||
        throw(ArgumentError("tolerance must be positive"))
    n_poles === nothing || n_poles > 0 ||
        throw(ArgumentError("n_poles must be positive"))

    z = _pes_frequencies(frequencies)
    samples = _pes_samples(values, length(z))
    positive_count = count(x -> imag(x) > 0, z)
    positive_count >= 3 ||
        throw(ArgumentError("PES fitting needs at least three positive Matsubara frequencies"))

    canonical_solver = solver == :lstsq ? :least_squares : solver
    canonical_solver == :sdp && conic_diagnostic != :none &&
        throw(ArgumentError("conic_diagnostic is only valid with solver=:least_squares"))
    min_s = _pes_even_at_least(Int(min_support))
    maximum_usable = 2positive_count - 2
    max_s = min(_pes_even_at_most(Int(max_support)), maximum_usable)
    max_s >= min_s ||
        throw(ArgumentError("frequency grid is too small for min_support=$min_support"))
    supports = if n_poles === nothing
        collect(min_s:2:max_s)
    else
        requested = _pes_even_at_least(Int(n_poles) + 1)
        requested <= maximum_usable ||
            throw(ArgumentError("n_poles=$n_poles requires more positive frequencies"))
        [requested]
    end

    started = time_ns()
    best = nothing
    best_error = Inf
    total_aaa_seconds = 0.0
    total_ls_seconds = 0.0
    total_sdp_seconds = 0.0
    total_nnls_seconds = 0.0
    total_constrained_refinement_seconds = 0.0
    sdp_solves = 0
    nnls_solves = 0
    attempts = 0

    for support in supports
        attempts += 1
        aaa_started = time_ns()
        estimated, aaa_info = _pes_aaa_real(samples, z;
                                             max_support=support,
                                             tolerance=Float64(aaa_tolerance))
        total_aaa_seconds += (time_ns() - aaa_started) / 1e9
        isempty(estimated) && continue
        poles = sort!(Float64.(real.(estimated)))

        preliminary, _, solve_info = _pes_fit_weights(
            poles, z, samples, canonical_solver, statistics)
        total_sdp_seconds += solve_info.sdp_seconds
        total_nnls_seconds += solve_info.nnls_seconds
        sdp_solves += solve_info.sdp_solves
        nnls_solves += solve_info.nnls_solves
        preliminary_error = _pes_max_error(poles, preliminary, z, samples, statistics)
        poles_changed = false
        if tolerance !== nothing
            keep = [norm(weight) > Float64(residue_tolerance) for weight in preliminary]
            any(keep) || (keep[argmax(norm.(preliminary))] = true)
            poles_changed = !all(keep)
            poles = poles[keep]
        end

        least_squares_refinement = nothing
        sdp_refinement = nothing
        if maxiter > 0
            ls_started = time_ns()
            poles, least_squares_refinement = _pes_refine_poles(
                poles, z, samples, :least_squares, statistics, Int(maxiter))
            total_ls_seconds += (time_ns() - ls_started) / 1e9
            if canonical_solver == :sdp
                sdp_started = time_ns()
                sdp_counter = Ref(0)
                nnls_counter = Ref(0)
                sdp_seconds_counter = Ref(0.0)
                nnls_seconds_counter = Ref(0.0)
                poles, sdp_refinement = _pes_refine_poles(
                    poles, z, samples, :sdp, statistics, Int(maxiter);
                    sdp_counter, nnls_counter, sdp_seconds_counter,
                    nnls_seconds_counter)
                total_constrained_refinement_seconds +=
                    (time_ns() - sdp_started) / 1e9
                total_sdp_seconds += sdp_seconds_counter[]
                total_nnls_seconds += nnls_seconds_counter[]
                sdp_solves += sdp_counter[]
                nnls_solves += nnls_counter[]
            end
        end

        weights, solve_info = if maxiter == 0 && !poles_changed
            preliminary, (; sdp_seconds=0.0, nnls_seconds=0.0,
                           sdp_solves=0, nnls_solves=0, backend=:reused)
        else
            fitted, _, info = _pes_fit_weights(
                poles, z, samples, canonical_solver, statistics)
            fitted, info
        end
        total_sdp_seconds += solve_info.sdp_seconds
        total_nnls_seconds += solve_info.nnls_seconds
        sdp_solves += solve_info.sdp_solves
        nnls_solves += solve_info.nnls_solves
        final_solver = canonical_solver
        error = _pes_max_error(poles, weights, z, samples, statistics)
        if error < best_error
            best_error = error
            best = (; poles=copy(poles), weights=copy.(weights), final_solver,
                    support, aaa_info, preliminary_error,
                    least_squares_refinement, sdp_refinement)
        end
        tolerance !== nothing && error <= tolerance && break
    end

    best === nothing && throw(ErrorException("PES pole estimation failed for every support budget"))
    conic_report = canonical_solver == :least_squares ?
        _pes_conic_diagnostic(best.poles, best.weights,
                              conic_diagnostic) : nothing
    residue_constraint = canonical_solver == :sdp ? :psd : :unconstrained
    diagnostics = (;
        requested_tolerance = tolerance === nothing ? nothing : Float64(tolerance),
        requested_n_poles = n_poles === nothing ? nothing : Int(n_poles),
        converged = tolerance === nothing ? nothing : best_error <= tolerance,
        max_abs_error = best_error,
        support_points = best.support,
        pole_count = length(best.poles),
        statistics,
        requested_solver = canonical_solver,
        final_solver = best.final_solver,
        residue_constraint,
        conic_diagnostic = conic_report,
        attempts,
        aaa = best.aaa_info,
        least_squares_refinement = best.least_squares_refinement,
        sdp_refinement = best.sdp_refinement,
        preliminary_max_abs_error = best.preliminary_error,
        timings = (;
            aaa_seconds = total_aaa_seconds,
            least_squares_refinement_seconds = total_ls_seconds,
            constrained_refinement_seconds = total_constrained_refinement_seconds,
            sdp_seconds = total_sdp_seconds,
            nnls_seconds = total_nnls_seconds,
            total_seconds = (time_ns() - started) / 1e9,
        ),
        sdp_solves,
        nnls_solves,
    )
    return PESPoleFit(best.poles, best.weights, statistics, diagnostics;
                      residue_constraint)
end

"""Evaluate a `PESPoleFit` at one complex frequency, returning a matrix."""
function evaluate_poles(fit::PESPoleFit, z::Number)
    point = ComplexF64(z)
    d = size(first(fit.weights), 1)
    value = zeros(ComplexF64, d, d)
    for j in eachindex(fit.poles)
        value .+= _pes_kernel(point, fit.poles[j], fit.statistics) .* fit.weights[j]
    end
    return value
end

"""Evaluate a `PESPoleFit` on a frequency collection."""
evaluate_poles(fit::PESPoleFit, frequencies) =
    [evaluate_poles(fit, z) for z in frequencies]

"""
    bath_orbitals(fit::PESPoleFit; atol=0, rtol=sqrt(eps()))

Factor each PSD pole weight as `R = sum(v * v')`.  Degenerate energies are
repeated once per retained eigenvector; coupling-vector phases are arbitrary.
"""
function bath_orbitals(fit::PESPoleFit; atol::Real=0.0,
                       rtol::Real=sqrt(eps(Float64)))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    energies = Float64[]
    couplings = Vector{ComplexF64}[]
    pole_indices = Int[]
    for (j, weight) in enumerate(fit.weights)
        psd_weight = _pes_validated_psd(weight, j)
        decomposition = eigen(Hermitian(psd_weight))
        scale = maximum(decomposition.values; init=0.0)
        cutoff = max(Float64(atol), Float64(rtol) * scale)
        for k in eachindex(decomposition.values)
            λ = decomposition.values[k]
            λ > cutoff || continue
            push!(energies, fit.poles[j])
            push!(couplings, sqrt(λ) .* ComplexF64.(decomposition.vectors[:, k]))
            push!(pole_indices, j)
        end
    end
    return (; energies, couplings, pole_indices)
end

function _pes_frequencies(frequencies)
    points = ComplexF64[]
    for frequency in frequencies
        z = complex(frequency)
        isfinite(real(z)) && isfinite(imag(z)) ||
            throw(ArgumentError("PES frequencies must be finite"))
        scale = max(abs(z), 1.0)
        if abs(imag(z)) <= 100eps(Float64) * scale
            push!(points, im * Float64(real(z)))
        elseif abs(real(z)) <= 100eps(Float64) * scale
            push!(points, ComplexF64(z))
        else
            throw(ArgumentError("PES frequencies must be real magnitudes or pure-imaginary points"))
        end
    end
    isempty(points) && throw(ArgumentError("PES frequency grid may not be empty"))
    length(unique(points)) == length(points) ||
        throw(ArgumentError("PES frequencies must be distinct"))
    return points
end

function _pes_samples(values, nfrequency::Int)
    samples = Matrix{ComplexF64}[]
    if values isa AbstractVector && all(x -> x isa Number, values)
        length(values) == nfrequency ||
            throw(DimensionMismatch("PES needs one scalar value per frequency"))
        for value in values
            push!(samples, reshape(ComplexF64[value], 1, 1))
        end
    elseif values isa AbstractVector && all(x -> x isa AbstractMatrix, values)
        length(values) == nfrequency ||
            throw(DimensionMismatch("PES needs one matrix value per frequency"))
        isempty(values) && throw(ArgumentError("PES samples may not be empty"))
        expected = size(first(values))
        expected[1] == expected[2] || throw(ArgumentError("PES samples must be square"))
        for (n, value) in enumerate(values)
            size(value) == expected ||
                throw(DimensionMismatch("PES sample $n has an inconsistent size"))
            push!(samples, Matrix{ComplexF64}(value))
        end
    elseif values isa AbstractArray && ndims(values) == 3
        size(values, 1) == nfrequency ||
            throw(DimensionMismatch("the first PES array dimension must index frequencies"))
        size(values, 2) == size(values, 3) ||
            throw(ArgumentError("PES matrix samples must be square"))
        for n in axes(values, 1)
            push!(samples, Matrix{ComplexF64}(@view values[n, :, :]))
        end
    else
        throw(ArgumentError("unsupported PES sample container"))
    end
    all(M -> all(x -> isfinite(real(x)) && isfinite(imag(x)), M), samples) ||
        throw(ArgumentError("PES samples must be finite"))
    return samples
end

function _pes_aaa_real(samples, points; max_support::Int, tolerance::Float64)
    positive = findall(z -> imag(z) > 0, points)
    zhalf = points[positive]
    fhalf = samples[positive]
    nhalf = length(zhalf)
    d = size(first(samples), 1)
    d2 = d^2
    z = vcat(zhalf, conj.(zhalf))
    mirrored = [Matrix{ComplexF64}(adjoint(value)) for value in fhalf]
    full_samples = vcat(fhalf, mirrored)
    npoint = length(z)
    flat = reduce(vcat, [permutedims(vec(value)) for value in full_samples])
    remaining = collect(1:npoint)
    support_z = ComplexF64[]
    support_f = Matrix{ComplexF64}(undef, 0, d2)
    mean_value = sum(flat) / (npoint * d2)
    approximation = fill(mean_value, npoint, d2)
    barycentric = ComplexF64[]
    achieved = Inf

    used_support = 0
    for count in 2:2:max_support
        residuals = [sum(abs2, view(flat, k, :) .- view(approximation, k, :))
                     for k in 1:npoint]
        first_index = argmax(residuals)
        partner = mod1(first_index + nhalf, npoint)
        first_index in remaining ||
            (first_index = remaining[argmax(residuals[remaining])];
             partner = mod1(first_index + nhalf, npoint))
        push!(support_z, z[first_index], z[partner])
        support_f = vcat(support_f, flat[first_index:first_index, :],
                         flat[partner:partner, :])
        filter!(index -> index != first_index && index != partner, remaining)
        used_support = count
        isempty(remaining) && break

        cauchy = 1.0 ./ (z[remaining] .- permutedims(support_z))
        loewner = Matrix{ComplexF64}(undef, length(remaining) * d2, count)
        for (row_index, sample_index) in enumerate(remaining), component in 1:d2
            row = (row_index - 1) * d2 + component
            for j in 1:count
                loewner[row, j] =
                    (flat[sample_index, component] - support_f[j, component]) *
                    cauchy[row_index, j]
            end
        end
        left = loewner[:, 1:2:count]
        right = loewner[:, 2:2:count]
        constrained = hcat(left + right, im .* (left - right))
        real_system = vcat(real.(constrained), imag.(constrained))
        singular = svd(real_system; full=false)
        real_weight = singular.V[:, end]
        half = count ÷ 2
        pair_weight = real_weight[1:half] .+ im .* real_weight[half + 1:end]
        barycentric = ComplexF64[]
        for weight in pair_weight
            push!(barycentric, weight, conj(weight))
        end

        approximation .= flat
        denominator = cauchy * barycentric
        for component in 1:d2
            numerator = cauchy * (barycentric .* support_f[:, component])
            approximation[remaining, component] .= numerator ./ denominator
        end
        achieved = maximum(abs.(approximation .- flat))
        achieved <= tolerance && break
    end
    isempty(barycentric) && throw(ErrorException("AAA failed before constructing a rational approximant"))
    poles, max_imaginary = _pes_barycentric_poles(support_z, barycentric)
    info = (; support_points=used_support, aaa_max_abs_error=achieved,
            max_discarded_pole_imaginary=max_imaginary)
    return poles, info
end

function _pes_barycentric_poles(support, weights)
    n = length(weights)
    pencil_a = zeros(ComplexF64, n + 1, n + 1)
    pencil_b = zeros(ComplexF64, n + 1, n + 1)
    for j in 1:n
        pencil_a[1, j + 1] = weights[j]
        pencil_a[j + 1, 1] = 1
        pencil_a[j + 1, j + 1] = support[j]
        pencil_b[j + 1, j + 1] = 1
    end
    eigenvalues = eigvals(pencil_a, pencil_b)
    finite = filter(z -> isfinite(real(z)) && isfinite(imag(z)), eigenvalues)
    max_imaginary = maximum(abs ∘ imag, finite; init=0.0)
    return real.(finite), max_imaginary
end

function _pes_fit_weights(poles, z, samples, solver::Symbol, statistics::Symbol)
    kernel = _pes_kernel_matrix(z, poles, statistics)
    if solver == :least_squares
        weights = _pes_hermitian_least_squares(kernel, samples)
        residual = _pes_residual(kernel, weights, samples)
        return weights, residual, (; sdp_seconds=0.0, nnls_seconds=0.0,
                                    sdp_solves=0, nnls_solves=0,
                                    backend=:least_squares)
    end
    solver == :sdp || throw(ArgumentError("unknown PES residue solver: $solver"))
    started = time_ns()
    weights, backend = _pes_sdp_weights(kernel, samples)
    weights = [_pes_validated_psd(weight, k) for (k, weight) in enumerate(weights)]
    elapsed = (time_ns() - started) / 1e9
    residual = _pes_residual(kernel, weights, samples)
    is_nnls = backend == :nnls
    return weights, residual, (; sdp_seconds=is_nnls ? 0.0 : elapsed,
                                nnls_seconds=is_nnls ? elapsed : 0.0,
                                sdp_solves=is_nnls ? 0 : 1,
                                nnls_solves=is_nnls ? 1 : 0,
                                backend)
end

function _pes_hermitian_least_squares(kernel, samples)
    nfrequency, npole = size(kernel)
    d = size(first(samples), 1)
    real_kernel = vcat(real.(kernel), imag.(kernel))
    weights = [zeros(ComplexF64, d, d) for _ in 1:npole]
    for i in 1:d
        rhs = vcat([real(samples[n][i, i]) for n in 1:nfrequency],
                   [imag(samples[n][i, i]) for n in 1:nfrequency])
        coefficient = real_kernel \ rhs
        for k in 1:npole
            weights[k][i, i] = coefficient[k]
        end
        for j in i + 1:d
            symmetric = [(samples[n][j, i] + samples[n][i, j]) / 2
                         for n in 1:nfrequency]
            antisymmetric = [(samples[n][i, j] - samples[n][j, i]) / 2
                             for n in 1:nfrequency]
            rhs_real = vcat(real.(symmetric), imag.(symmetric))
            rhs_imag = vcat(imag.(antisymmetric), -real.(antisymmetric))
            real_part = real_kernel \ rhs_real
            imaginary_part = real_kernel \ rhs_imag
            for k in 1:npole
                weights[k][i, j] = real_part[k] + im * imaginary_part[k]
                weights[k][j, i] = real_part[k] - im * imaginary_part[k]
            end
        end
    end
    return weights
end

function _pes_sdp_weights(kernel, samples)
    nfrequency, npole = size(kernel)
    d = size(first(samples), 1)
    if d == 1
        real_kernel = vcat(real.(kernel), imag.(kernel))
        rhs = vcat([real(samples[n][1, 1]) for n in 1:nfrequency],
                   [imag(samples[n][1, 1]) for n in 1:nfrequency])
        coefficients, _ = _nnls(real_kernel, rhs)
        weights = [reshape(ComplexF64[coefficient], 1, 1)
                   for coefficient in coefficients]
        return weights, :nnls
    end
    model = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_optimizer_attribute(model, "tol_gap_abs", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_gap_rel", 1e-12)
    JuMP.set_optimizer_attribute(model, "tol_feas", 1e-12)
    JuMP.set_optimizer_attribute(model, "max_iter", 500)
    real_weights = [JuMP.@variable(model, [1:d, 1:d], Symmetric)
                    for _ in 1:npole]
    imag_weights = [JuMP.@variable(model, [1:d, 1:d]) for _ in 1:npole]
    for k in 1:npole
        for i in 1:d, j in i:d
            JuMP.@constraint(model,
                imag_weights[k][i, j] + imag_weights[k][j, i] == 0)
        end
        block = [real_weights[k] -imag_weights[k];
                 imag_weights[k] real_weights[k]]
        JuMP.@constraint(model, block in JuMP.PSDCone())
    end
    JuMP.@objective(model, Min, sum(
        (sum(real(kernel[n, k]) * real_weights[k][i, j] -
             imag(kernel[n, k]) * imag_weights[k][i, j] for k in 1:npole) -
         real(samples[n][i, j]))^2 +
        (sum(real(kernel[n, k]) * imag_weights[k][i, j] +
             imag(kernel[n, k]) * real_weights[k][i, j] for k in 1:npole) -
         imag(samples[n][i, j]))^2
        for n in 1:nfrequency, i in 1:d, j in 1:d))
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    status in (JuMP.MOI.OPTIMAL, JuMP.MOI.ALMOST_OPTIMAL) ||
        throw(ErrorException("Clarabel failed to solve the PES SDP: $status"))
    result = Matrix{ComplexF64}[]
    for k in 1:npole
        value = ComplexF64.(JuMP.value.(real_weights[k])) .+
                im .* JuMP.value.(imag_weights[k])
        push!(result, (value + value') / 2)
    end
    return result, :real_block_sdp
end

function _pes_refine_poles(initial, z, samples, solver::Symbol,
                           statistics::Symbol, maxiter::Int;
                           sdp_counter::Union{Nothing,Base.RefValue{Int}}=nothing,
                           nnls_counter::Union{Nothing,Base.RefValue{Int}}=nothing,
                           sdp_seconds_counter::Union{Nothing,Base.RefValue{Float64}}=nothing,
                           nnls_seconds_counter::Union{Nothing,Base.RefValue{Float64}}=nothing)
    cached_poles = Ref(Float64[])
    cached_value = Ref(Inf)
    cached_gradient = Ref(Float64[])
    function evaluate(poles)
        if length(cached_poles[]) == length(poles) && cached_poles[] == poles
            return cached_value[], cached_gradient[]
        end
        weights, residual, info = _pes_fit_weights(poles, z, samples, solver, statistics)
        sdp_counter === nothing || (sdp_counter[] += info.sdp_solves)
        nnls_counter === nothing || (nnls_counter[] += info.nnls_solves)
        sdp_seconds_counter === nothing ||
            (sdp_seconds_counter[] += info.sdp_seconds)
        nnls_seconds_counter === nothing ||
            (nnls_seconds_counter[] += info.nnls_seconds)
        value = _pes_residual_norm(residual)
        gradient = _pes_pole_gradient(poles, z, weights, residual, statistics, value)
        cached_poles[] = copy(poles)
        cached_value[] = value
        cached_gradient[] = gradient
        return value, gradient
    end
    objective(poles) = first(evaluate(poles))
    function gradient!(storage, poles)
        storage .= last(evaluate(poles))
        return storage
    end
    options = Optim.Options(iterations=maxiter, g_tol=1e-10, f_reltol=1e-10,
                            show_trace=false, store_trace=false)
    method = Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking())
    result = Optim.optimize(objective, gradient!, Float64.(initial), method, options)
    poles = Optim.minimizer(result)
    all(isfinite, poles) || throw(ErrorException("Optim returned non-finite PES poles"))
    refinement = (;
        converged = Optim.converged(result),
        iterations = Optim.iterations(result),
        residual_norm = Optim.minimum(result),
    )
    return sort!(collect(poles)), refinement
end

function _pes_pole_gradient(poles, z, weights, residual, statistics, value)
    value > eps(Float64) || return zeros(Float64, length(poles))
    gradient = zeros(Float64, length(poles))
    for k in eachindex(poles), n in eachindex(z)
        derivative = _pes_kernel_derivative(z[n], poles[k], statistics)
        gradient[k] -= real(sum(conj.(residual[n]) .* (derivative .* weights[k]))) / value
    end
    return gradient
end

function _pes_kernel_matrix(z, poles, statistics)
    return [_pes_kernel(z[n], poles[k], statistics)
            for n in eachindex(z), k in eachindex(poles)]
end

_pes_kernel(z::Complex, pole::Real, ::Val{:fermion}) = inv(z - pole)
_pes_kernel(z::Complex, pole::Real, ::Val{:boson}) = iszero(z) ? -1.0 + 0im : pole / (z - pole)
_pes_kernel(z::Complex, pole::Real, statistics::Symbol) =
    _pes_kernel(z, pole, Val(statistics))

_pes_kernel_derivative(z::Complex, pole::Real, ::Val{:fermion}) = inv((z - pole)^2)
_pes_kernel_derivative(z::Complex, pole::Real, ::Val{:boson}) =
    iszero(z) ? 0.0 + 0im : z / (z - pole)^2
_pes_kernel_derivative(z::Complex, pole::Real, statistics::Symbol) =
    _pes_kernel_derivative(z, pole, Val(statistics))

function _pes_residual(kernel, weights, samples)
    return [samples[n] .- sum(kernel[n, k] .* weights[k] for k in axes(kernel, 2))
            for n in axes(kernel, 1)]
end

_pes_residual_norm(residual) = sqrt(sum(sum(abs2, value) for value in residual))

function _pes_max_error(poles, weights, z, samples, statistics)
    residual = _pes_residual(_pes_kernel_matrix(z, poles, statistics), weights, samples)
    return maximum(maximum(abs, value) for value in residual)
end

function _pes_conic_diagnostic(poles, weights, mode::Symbol)
    mode == :none && return nothing
    started = time_ns()
    projection = _pes_psd_cone_distance(poles, weights)
    return (;
        mode,
        projection,
        diagnostic_seconds = (time_ns() - started) / 1e9,
    )
end

function _pes_psd_cone_distance(poles, weights)
    per_pole = map(enumerate(weights)) do (index, weight)
        values = Float64.(eigvals(Hermitian((weight + weight') / 2)))
        negative = min.(values, 0.0)
        distance = norm(negative)
        weight_norm = norm(weight)
        tolerance = 100sqrt(eps(Float64)) * max(weight_norm, 1.0)
        operator_distance = max(0.0, -minimum(values))
        (;
            pole_index = index,
            pole = poles[index],
            frobenius_norm = weight_norm,
            minimum_eigenvalue = minimum(values),
            operator_distance,
            frobenius_distance = distance,
            relative_frobenius_distance =
                _pes_relative_distance(distance, weight_norm),
            raw_negative_eigenvalue_count = count(<(0.0), values),
            negative_eigenvalue_count = count(<(-tolerance), values),
            within_psd_tolerance = minimum(values) >= -tolerance,
            tolerance,
        )
    end
    distance = sqrt(sum(item.frobenius_distance^2 for item in per_pole))
    total_norm = sqrt(sum(sum(abs2, weight) for weight in weights))
    worst_pole_index = argmax(item.operator_distance for item in per_pole)
    return (;
        frobenius_distance = distance,
        relative_frobenius_distance = _pes_relative_distance(distance, total_norm),
        operator_distance = per_pole[worst_pole_index].operator_distance,
        minimum_eigenvalue = minimum(item.minimum_eigenvalue for item in per_pole),
        worst_pole_index,
        worst_pole = poles[worst_pole_index],
        raw_negative_eigenvalue_count =
            sum(item.raw_negative_eigenvalue_count for item in per_pole),
        negative_eigenvalue_count =
            sum(item.negative_eigenvalue_count for item in per_pole),
        raw_violating_pole_count =
            count(item -> item.frobenius_distance > 0, per_pole),
        violating_pole_count = count(item -> !item.within_psd_tolerance, per_pole),
        per_pole,
    )
end

_pes_relative_distance(distance, scale) =
    iszero(scale) ? (iszero(distance) ? 0.0 : Inf) : distance / scale

function _pes_validated_hermitian(weight::Matrix{ComplexF64}, index::Int)
    all(z -> isfinite(real(z)) && isfinite(imag(z)), weight) ||
        throw(ArgumentError("PES weight $index must be finite"))
    scale = max(norm(weight), 1.0)
    norm(weight - weight') <= 1e-6 * scale ||
        throw(ArgumentError("PES weight $index must be Hermitian"))
    return Matrix{ComplexF64}((weight + weight') / 2)
end

function _pes_validated_psd(weight::Matrix{ComplexF64}, index::Int)
    hermitian = _pes_validated_hermitian(weight, index)
    scale = max(norm(hermitian), 1.0)
    decomposition = eigen(Hermitian(hermitian))
    minimum(decomposition.values) >= -1e-6 * scale ||
        throw(ArgumentError("PES weight $index must be positive semidefinite"))
    eigenvalues = max.(Float64.(decomposition.values), 0.0)
    projected = decomposition.vectors * (eigenvalues .* decomposition.vectors')
    return Matrix{ComplexF64}((projected + projected') / 2)
end

_pes_even_at_least(value::Int) = iseven(value) ? value : value + 1
_pes_even_at_most(value::Int) = iseven(value) ? value : value - 1
