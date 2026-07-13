function _with_fit_timing(expansion::PoleExpansion, started::Integer)
    elapsed = (time_ns() - started) / 1e9
    isfinite(elapsed) && elapsed >= 0 || throw(ErrorException(
        "bath-fit elapsed time is not finite",
    ))
    return PoleExpansion(
        expansion.poles;
        kernel=expansion.kernel,
        trace=merge(expansion.trace, (; fit_seconds=elapsed)),
    )
end

function _bathfit_trace_broadening(expansion::PoleExpansion)
    hasproperty(expansion.trace, :broadening) || return nothing
    return getproperty(expansion.trace, :broadening)
end

function _bathfit_report_broadening(input::BathFitInput,
                                    expansion::PoleExpansion, broadening)
    input.domain === :matsubara && return _reconstruction_broadening(
        input, broadening,
    )
    candidate = broadening
    if candidate === nothing
        candidate = _bathfit_trace_broadening(expansion)
    end
    return _reconstruction_broadening(input, candidate)
end

function _validate_realization_broadening(input::BathFitInput,
                                          expansion::PoleExpansion,
                                          broadening)
    broadening !== nothing && return _bathfit_report_broadening(
        input, expansion, broadening,
    )
    input.domain === :matsubara && return nothing
    _bathfit_trace_broadening(expansion) === nothing && return nothing
    return _bathfit_report_broadening(input, expansion, nothing)
end

function _bathfit_block_residual(source::Vector{Matrix{ComplexF64}},
                                 reconstruction::Vector{Matrix{ComplexF64}})
    length(source) == length(reconstruction) || throw(DimensionMismatch(
        "bath-fit residual needs source and reconstruction samples on one grid",
    ))
    absolute = 0.0
    maximum_error = 0.0
    l2_squared = 0.0
    source_l2_squared = 0.0
    for (target, value) in zip(source, reconstruction)
        size(target) == size(value) || throw(DimensionMismatch(
            "bath-fit residual requires matching block sample shapes",
        ))
        difference = value - target
        absolute += norm(difference)
        maximum_error = max(maximum_error, maximum(abs, difference; init=0.0))
        l2_squared += sum(abs2, difference)
        source_l2_squared += sum(abs2, target)
    end
    l2 = sqrt(l2_squared)
    relative = if iszero(source_l2_squared)
        iszero(l2) ? 0.0 : Inf
    else
        l2 / sqrt(source_l2_squared)
    end
    return BathFitResidual(absolute, maximum_error, l2, relative)
end

function _bathfit_spectral_samples(samples::Vector{Matrix{ComplexF64}},
                                   component::Symbol)
    component === :spectral && return samples
    component === :retarded || throw(ArgumentError(
        "real-axis spectral diagnostics need component=:spectral or :retarded",
    ))
    return Matrix{ComplexF64}[
        (adjoint(sample) - sample) / (2pi * im) for sample in samples
    ]
end

function _bathfit_spectral_weight_error(frequencies::Vector{Float64},
                                         source::Vector{Matrix{ComplexF64}},
                                         reconstruction::Vector{Matrix{ComplexF64}},
                                         component::Symbol)
    target_samples = _bathfit_spectral_samples(source, component)
    reconstruction_samples = _bathfit_spectral_samples(reconstruction, component)
    length(frequencies) >= 2 || return norm(
        only(reconstruction_samples) - only(target_samples),
    )
    permutation = sortperm(frequencies)
    ordered_frequencies = frequencies[permutation]
    target = target_samples[permutation]
    value = reconstruction_samples[permutation]
    source_weight = _integrate_linear_matrix(
        ordered_frequencies, target, first(ordered_frequencies),
        last(ordered_frequencies),
    )
    reconstruction_weight = _integrate_linear_matrix(
        ordered_frequencies, value, first(ordered_frequencies),
        last(ordered_frequencies),
    )
    return norm(reconstruction_weight - source_weight)
end

function _bathfit_residue_cone(residue::Number)
    value = ComplexF64(residue)
    projected = max(real(value), 0.0) + 0im
    return (; minimum_eigenvalue=real(value), distance=abs(value - projected),
            squared_norm=abs2(value))
end

function _bathfit_residue_cone(residue::AbstractMatrix)
    matrix = Matrix{ComplexF64}(residue)
    hermitian = (matrix + adjoint(matrix)) / 2
    decomposition = eigen(Hermitian(hermitian))
    values = real.(decomposition.values)
    projected = decomposition.vectors *
                Diagonal(max.(values, 0.0)) * adjoint(decomposition.vectors)
    return (; minimum_eigenvalue=minimum(values), distance=norm(matrix - projected),
            squared_norm=sum(abs2, matrix))
end

function _bathfit_residue_diagnostics(expansion::PoleExpansion,
                                      block_index_value::Int)
    indices = findall(==(block_index_value), expansion.poles.block_indices)
    isempty(indices) && return (; minimum_eigenvalue=Inf, cone_distance=0.0,
                                 relative_cone_distance=0.0)
    per_residue = [_bathfit_residue_cone(expansion.poles.residues[index])
                   for index in indices]
    distance = sqrt(sum(item.distance^2 for item in per_residue))
    norm_value = sqrt(sum(item.squared_norm for item in per_residue))
    return (; minimum_eigenvalue=minimum(item.minimum_eigenvalue for item in per_residue),
            cone_distance=distance,
            relative_cone_distance=iszero(norm_value) ? 0.0 : distance / norm_value)
end

function _bathfit_block_spacing(expansion::PoleExpansion,
                                bath::Union{Nothing,DiscreteBath},
                                block_index_value::Int)
    pole_indices = findall(==(block_index_value), expansion.poles.block_indices)
    raw_poles = expansion.poles.poles[pole_indices]
    energies = if bath === nothing
        raw_poles
    else
        mode_indices = findall(==(block_index_value), bath.orbitals.block_indices)
        bath.orbitals.energies[mode_indices]
    end
    levels = sort!(unique!(Float64.(energies)))
    bandwidth = length(levels) <= 1 ? 0.0 : last(levels) - first(levels)
    max_spacing = length(levels) <= 1 ? nothing : maximum(diff(levels))
    revival = max_spacing === nothing ? nothing : 2pi / max_spacing
    mode_count = bath === nothing ? 0 : count(==(block_index_value),
                                               bath.orbitals.block_indices)
    return (; pole_count=length(pole_indices), mode_count, bandwidth,
            max_spacing, revival_time=revival)
end

function _bathfit_boundary_curve(trace::NamedTuple)
    hasproperty(trace, :boundary_curve) || return ()
    curve = getproperty(trace, :boundary_curve)
    return Tuple(curve)
end

function _bathfit_trace_warnings(trace::NamedTuple)
    warnings = BathFitWarning[]
    if hasproperty(trace, :boundary_curve)
        for candidate in getproperty(trace, :boundary_curve)
            hasproperty(candidate, :status) || continue
            getproperty(candidate, :status) === :invalid || continue
            message = hasproperty(candidate, :message) &&
                      getproperty(candidate, :message) !== nothing ?
                      String(getproperty(candidate, :message)) :
                      "BoundaryFit candidate was invalid"
            push!(warnings, BathFitWarning(:boundary_candidate_invalid, nothing,
                                           message))
        end
    end
    if hasproperty(trace, :fits)
        for fit in getproperty(trace, :fits)
            hasproperty(fit, :status) || continue
            getproperty(fit, :status) === :nonconverged || continue
            block = hasproperty(fit, :block) ? getproperty(fit, :block) : nothing
            push!(warnings, BathFitWarning(
                :fit_nonconverged, block,
                "the fitter returned a finite, nonconverged result",
            ))
        end
    end
    return warnings
end

function _bathfit_diagnostic_warnings(diagnostics::Vector{PoleBinDiagnostic})
    warnings = BathFitWarning[]
    for diagnostic in diagnostics
        diagnostic.status === :valid && continue
        message = if diagnostic.status in (:numerical_zero,
                                           :numerical_symmetrization,
                                           :numerical_symmetrization_and_zero)
            "residue passed the realization tolerance with status $(diagnostic.status)"
        else
            "residue could not be realized with status $(diagnostic.status)"
        end
        push!(warnings, BathFitWarning(
            Symbol("residue_", diagnostic.status), diagnostic.block, message,
        ))
    end
    return warnings
end

function _bathfit_fit_seconds(trace::NamedTuple)
    hasproperty(trace, :fit_seconds) || return nothing
    value = Float64(getproperty(trace, :fit_seconds))
    isfinite(value) && value >= 0 || throw(ArgumentError(
        "PoleExpansion trace.fit_seconds must be finite and nonnegative",
    ))
    return value
end

function _bathfit_report(expansion::PoleExpansion, input::BathFitInput,
                         plan::DiscretizationPlan,
                         bath::Union{Nothing,DiscreteBath},
                         diagnostics::Vector{PoleBinDiagnostic},
                         resolved_order::NamedTuple,
                         realization_seconds::Real;
                         broadening=nothing)
    trace = merge(expansion.trace, (; realization_orbital_order=resolved_order))
    warnings = vcat(_bathfit_diagnostic_warnings(diagnostics),
                    _bathfit_trace_warnings(trace))
    reconstruction = nothing
    reconstruction_seconds = nothing
    eta = if input.domain === :real_axis && broadening === nothing &&
             _bathfit_trace_broadening(expansion) === nothing
        push!(warnings, BathFitWarning(
            :reconstruction_unavailable, nothing,
            "real-axis reconstruction requires an explicit positive broadening",
        ))
        nothing
    else
        _bathfit_report_broadening(input, expansion, broadening)
    end
    if bath === nothing
        push!(warnings, BathFitWarning(
            :nonmountable, nothing,
            "the raw pole expansion is not Hamiltonian-mountable; no bath reconstruction exists",
        ))
    else
        can_reconstruct = input.domain === :matsubara || eta !== nothing
        if can_reconstruct
            started = time_ns()
            reconstruction = reconstruct_hybridization(bath, input; broadening=eta)
            reconstruction_seconds = (time_ns() - started) / 1e9
        end
    end
    component = _bathfit_component(input)
    names = block_names(expansion.poles.partition)
    # The realization boundary already validates matching layout and partition;
    # use its expansion-owned ordering to preserve the canonical report order.
    reports = Tuple(begin
        block_index_value = block_index(expansion.poles.partition, block)
        source_samples = getproperty(input.blocks, block)
        reconstructed_samples = reconstruction === nothing ? nothing :
                                getproperty(reconstruction.blocks, block)
        residual = reconstructed_samples === nothing ? nothing :
                   _bathfit_block_residual(source_samples, reconstructed_samples)
        spectral = reconstructed_samples === nothing || input.domain !== :real_axis ?
                   nothing : _bathfit_spectral_weight_error(
            input.frequencies, source_samples, reconstructed_samples, component,
        )
        plan_block_value = plan_block(plan, block)
        spacing = _bathfit_block_spacing(expansion, bath, block_index_value)
        residue = _bathfit_residue_diagnostics(expansion, block_index_value)
        BathFitBlockReport(
            block; residual, spectral_weight_error=spectral,
            discarded_weight=plan_block_value.discarded_weight,
            weight_measure=plan_block_value.weight_measure,
            pole_count=spacing.pole_count, mode_count=spacing.mode_count,
            bandwidth=spacing.bandwidth, max_spacing=spacing.max_spacing,
            revival_time=spacing.revival_time,
            minimum_residue_eigenvalue=residue.minimum_eigenvalue,
            psd_cone_distance=residue.cone_distance,
            relative_psd_cone_distance=residue.relative_cone_distance,
            boundary_curve=_bathfit_boundary_curve(trace),
        )
    end for block in names)
    blocks = NamedTuple{names}(reports)
    timing = BathFitTiming(
        fit_seconds=_bathfit_fit_seconds(trace),
        realization_seconds=realization_seconds,
        reconstruction_seconds=reconstruction_seconds,
    )
    return BathFitReport(
        input, reconstruction, blocks, plan, expansion.kernel, bath !== nothing,
        eta, copy(diagnostics), warnings, timing, trace,
    )
end
