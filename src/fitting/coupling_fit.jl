"""
Direct coupling-space Matsubara fitting for a finite Hamiltonian bath.

This is an independent Julia implementation of the finite-bath form
`Delta(iw) = sum(V_k * V_k' / (iw - epsilon_k))`. It is informed by the
weighted objective in Cao--Stoudenmire--Parcollet (2024), but does not translate
the reference forkTPS implementation. The optimization owns only numerical
parameters; `BathFitInput` continues to own source data and `PoleExpansion`
continues to carry the raw full matrix residues.
"""

function _coupling_validate_ties(kernel::CouplingFitKernel, partition::Partition)
    names = block_names(partition)
    known = Set(names)
    for tie in kernel.block_ties
        tie.source in known || throw(ArgumentError(
            "CouplingBlockTie source $(tie.source) is not a Partition block",
        ))
        tie.target in known || throw(ArgumentError(
            "CouplingBlockTie target $(tie.target) is not a Partition block",
        ))
        length(block_flavors(partition, tie.source)) ==
            length(block_flavors(partition, tie.target)) || throw(DimensionMismatch(
                "CouplingBlockTie $(tie.source) => $(tie.target) needs equal block dimensions",
            ))
    end
    return kernel
end

function _coupling_selected_grid(input::BathFitInput, kernel::CouplingFitKernel,
                                 partition::Partition)
    _validate_matsubara_fit(input, partition; kernel_name="CouplingFitKernel")
    input.statistics === :fermion || throw(ArgumentError(
        "CouplingFitKernel currently supports fermionic Matsubara input only",
    ))
    _coupling_validate_ties(kernel, partition)
    order = sortperm(input.frequencies)
    frequencies = input.frequencies[order]
    all(frequency -> frequency > 0, frequencies) || throw(ArgumentError(
        "CouplingFitKernel requires strictly positive Matsubara frequencies",
    ))
    window = kernel.frequency_window
    selected = window === nothing ? eachindex(frequencies) :
        findall(frequency -> window[1] <= frequency <= window[2], frequencies)
    length(selected) >= max(4, 2 * kernel.n_modes) || throw(ArgumentError(
        "CouplingFitKernel frequency window leaves too few Matsubara samples",
    ))
    selected_frequencies = frequencies[selected]
    all(frequency -> isfinite(frequency^(-kernel.alpha)), selected_frequencies) ||
        throw(ArgumentError(
            "CouplingFitKernel selected Matsubara weights must be finite",
        ))
    return order, Int.(collect(selected)), selected_frequencies
end

function _coupling_energy_bounds(kernel::CouplingFitKernel,
                                 frequencies::Vector{Float64})
    bounds = kernel.energy_bounds
    if bounds === nothing
        scale = min(maximum(abs, frequencies), floatmax(Float64) / 2)
        return (-scale, scale)
    end
    return bounds
end

function _coupling_mode_bounds(bounds::Tuple{Float64,Float64},
                               kernel::CouplingFitKernel)
    lower, upper = bounds
    modes = kernel.n_modes
    if kernel.allocation isa FreeModeAllocation
        return fill(lower, modes), fill(upper, modes)
    end
    negative = (kernel.allocation::SignedModeAllocation).n_negative
    scale = max(abs(lower), abs(upper))
    zero_guard = max(nextfloat(0.0), sqrt(eps(Float64)) * scale)
    mode_lower = Vector{Float64}(undef, modes)
    mode_upper = Vector{Float64}(undef, modes)
    for mode in 1:modes
        if mode <= negative
            mode_lower[mode] = lower
            mode_upper[mode] = min(upper, -zero_guard)
            mode_lower[mode] < mode_upper[mode] || throw(ArgumentError(
                "CouplingFitKernel energy_bounds leave no negative interval for SignedModeAllocation",
            ))
        else
            mode_lower[mode] = max(lower, zero_guard)
            mode_upper[mode] = upper
            mode_lower[mode] < mode_upper[mode] || throw(ArgumentError(
                "CouplingFitKernel energy_bounds leave no positive interval for SignedModeAllocation",
            ))
        end
    end
    return mode_lower, mode_upper
end

function _coupling_initial_energies(lower::Vector{Float64}, upper::Vector{Float64})
    energies = similar(lower)
    handled = falses(length(lower))
    for mode in eachindex(lower)
        handled[mode] && continue
        group = findall(index -> lower[index] == lower[mode] &&
                                 upper[index] == upper[mode], eachindex(lower))
        for (position, index) in enumerate(group)
            fraction = position / (length(group) + 1)
            energies[index] = lower[index] + fraction * (upper[index] - lower[index])
            handled[index] = true
        end
    end
    return energies
end

function _coupling_initial_couplings(samples::Vector{Matrix{ComplexF64}},
                                     frequencies::Vector{Float64}, modes::Int,
                                     components::AbstractCouplingComponents)
    dimension = size(first(samples), 1)
    moment = zeros(ComplexF64, dimension, dimension)
    for (sample, frequency) in zip(samples, frequencies)
        contribution = im * frequency .* sample
        all(entry -> isfinite(real(entry)) && isfinite(imag(entry)), contribution) ||
            throw(ArgumentError(
                "CouplingFitKernel selected high-frequency moment is not representable in Float64",
            ))
        moment .+= contribution
        all(entry -> isfinite(real(entry)) && isfinite(imag(entry)), moment) ||
            throw(ArgumentError(
                "CouplingFitKernel selected high-frequency moment accumulation overflows Float64",
            ))
    end
    moment ./= length(samples)
    hermitian_moment = Hermitian((moment + moment') ./ 2)
    decomposition = eigen(hermitian_moment)
    values = max.(real.(decomposition.values), 0.0)
    component_order = sortperm(values; rev=true)
    component_indices = [component_order[mod1(mode, dimension)] for mode in 1:modes]
    multiplicities = [count(==(component), component_indices)
                      for component in 1:dimension]
    couplings = Vector{Vector{ComplexF64}}(undef, modes)
    for mode in 1:modes
        component = component_indices[mode]
        amplitude = sqrt(values[component] / multiplicities[component])
        vector = ComplexF64.(amplitude .* decomposition.vectors[:, component])
        if iszero(norm(vector))
            vector[component] = sqrt(eps(Float64))
        end
        components isa RealComponents && (vector = ComplexF64.(real.(vector)))
        couplings[mode] = vector
    end
    return couplings
end

function _coupling_logistic(value::Float64)
    if value >= 0
        return inv(1 + exp(-value))
    end
    exponent = exp(value)
    return exponent / (1 + exponent)
end

function _coupling_logit(value::Float64)
    clipped = clamp(value, sqrt(eps(Float64)), 1 - sqrt(eps(Float64)))
    return log(clipped / (1 - clipped))
end

function _coupling_encode(energies::Vector{Float64},
                          couplings::Vector{Vector{ComplexF64}},
                          lower::Vector{Float64}, upper::Vector{Float64},
                          components::AbstractCouplingComponents)
    parameters = Float64[]
    for (energy, low, high) in zip(energies, lower, upper)
        push!(parameters, _coupling_logit((energy - low) / (high - low)))
    end
    append!(parameters, _coupling_encode_components(couplings, components))
    return parameters
end

function _coupling_encode_components(couplings::Vector{Vector{ComplexF64}},
                                     components::AbstractCouplingComponents)
    parameters = Float64[]
    for vector in couplings
        append!(parameters, real.(vector))
        components isa ComplexComponents && append!(parameters, imag.(vector))
    end
    return parameters
end

function _coupling_decode(parameters::AbstractVector{<:Real}, modes::Int,
                          dimension::Int, lower::Vector{Float64},
                          upper::Vector{Float64},
                          components::AbstractCouplingComponents)
    expected = modes + modes * dimension *
        (components isa ComplexComponents ? 2 : 1)
    length(parameters) == expected || throw(DimensionMismatch(
        "CouplingFitKernel parameter vector has an inconsistent length",
    ))
    energies = Float64[
        lower[mode] + (upper[mode] - lower[mode]) *
            _coupling_logistic(Float64(parameters[mode]))
        for mode in 1:modes
    ]
    return energies, _coupling_decode_components(
        @view(parameters[(modes + 1):end]), modes, dimension, components,
    )
end

function _coupling_decode_components(parameters::AbstractVector{<:Real}, modes::Int,
                                     dimension::Int,
                                     components::AbstractCouplingComponents)
    expected = modes * dimension * (components isa ComplexComponents ? 2 : 1)
    length(parameters) == expected || throw(DimensionMismatch(
        "CouplingFitKernel coupling-component vector has an inconsistent length",
    ))
    index = 1
    couplings = Vector{Vector{ComplexF64}}(undef, modes)
    for mode in 1:modes
        real_part = Float64.(parameters[index:(index + dimension - 1)])
        index += dimension
        imaginary_part = if components isa ComplexComponents
            values = Float64.(parameters[index:(index + dimension - 1)])
            index += dimension
            values
        else
            zeros(Float64, dimension)
        end
        couplings[mode] = ComplexF64.(real_part .+ im .* imaginary_part)
    end
    return couplings
end

function _coupling_values(energies::Vector{Float64},
                          couplings::Vector{Vector{ComplexF64}},
                          frequencies::Vector{Float64})
    dimension = length(first(couplings))
    values = Matrix{ComplexF64}[]
    for frequency in frequencies
        value = zeros(ComplexF64, dimension, dimension)
        for (energy, coupling) in zip(energies, couplings)
            value .+= coupling * coupling' ./ (im * frequency - energy)
        end
        push!(values, value)
    end
    return values
end

function _coupling_error(values::Vector{Matrix{ComplexF64}},
                         samples::Vector{Matrix{ComplexF64}},
                         frequencies::Vector{Float64}, alpha::Float64)
    weighted_squared = 0.0
    target_squared = 0.0
    maximum_error = 0.0
    for (value, sample, frequency) in zip(values, samples, frequencies)
        weight = frequency^(-alpha)
        difference = norm(value - sample)
        maximum_error = max(maximum_error, difference)
        weighted_squared += weight * difference^2
        target_squared += weight * norm(sample)^2
    end
    l2 = sqrt(weighted_squared)
    return (; maximum=maximum_error, weighted_l2=l2,
            target_weighted_l2=sqrt(target_squared),
            relative_l2=target_squared == 0 ? l2 : l2 / sqrt(target_squared))
end

function _coupling_related_vectors(couplings::Vector{Vector{ComplexF64}}, ::EqualTie)
    return copy.(couplings)
end

function _coupling_related_vectors(couplings::Vector{Vector{ComplexF64}}, ::ConjugateTie)
    return Vector{ComplexF64}[ComplexF64.(conj.(coupling))
                               for coupling in couplings]
end

function _coupling_combined_error(energies::Vector{Float64},
                                  couplings::Vector{Vector{ComplexF64}},
                                  samples::Vector{Matrix{ComplexF64}},
                                  frequencies::Vector{Float64}, alpha::Float64,
                                  followers)
    source_values = _coupling_values(energies, couplings, frequencies)
    source_error = _coupling_error(source_values, samples, frequencies, alpha)
    weighted_squared = source_error.weighted_l2^2
    target_squared = source_error.target_weighted_l2^2
    maximum_error = source_error.maximum
    follower_errors = NamedTuple[]
    for follower in followers
        follower_couplings = _coupling_related_vectors(
            couplings, follower.tie.relation,
        )
        follower_values = _coupling_values(energies, follower_couplings, frequencies)
        error = _coupling_error(
            follower_values, follower.samples, frequencies, alpha,
        )
        weighted_squared += error.weighted_l2^2
        target_squared += error.target_weighted_l2^2
        maximum_error = max(maximum_error, error.maximum)
        push!(follower_errors, (; block=follower.tie.target,
                                relation=follower.tie.relation, error))
    end
    weighted_l2 = sqrt(weighted_squared)
    aggregate = (; maximum=maximum_error, weighted_l2,
                 target_weighted_l2=sqrt(target_squared),
                 relative_l2=target_squared == 0 ? weighted_l2 :
                             weighted_l2 / sqrt(target_squared))
    return (; source_error, follower_errors, aggregate)
end

function _coupling_objective(parameters::AbstractVector{<:Real}, modes::Int,
                             dimension::Int, lower::Vector{Float64},
                             upper::Vector{Float64},
                             components::AbstractCouplingComponents,
                             samples::Vector{Matrix{ComplexF64}},
                             frequencies::Vector{Float64}, alpha::Float64,
                             followers)
    energies, couplings = _coupling_decode(
        parameters, modes, dimension, lower, upper, components,
    )
    error = _coupling_combined_error(
        energies, couplings, samples, frequencies, alpha, followers,
    )
    return error.aggregate.weighted_l2^2 / (length(frequencies) * (length(followers) + 1))
end

function _coupling_zero_block(samples::Vector{Matrix{ComplexF64}})
    return all(sample -> iszero(norm(sample)), samples)
end

function _coupling_require_tolerance(errors, tolerance::Union{Nothing,Float64},
                                     source::Symbol)
    tolerance === nothing && return nothing
    errors.source_error.relative_l2 <= tolerance || throw(ArgumentError(
        "CouplingFitKernel relative fit error exceeds fit_tolerance for block $source",
    ))
    for follower in errors.follower_errors
        follower.error.relative_l2 <= tolerance || throw(ArgumentError(
            "CouplingFitKernel tied block $(follower.block) exceeds fit_tolerance",
        ))
    end
    return nothing
end

function _coupling_refine_fixed_energies(
    energies::Vector{Float64}, couplings::Vector{Vector{ComplexF64}},
    samples::Vector{Matrix{ComplexF64}}, frequencies::Vector{Float64},
    kernel::CouplingFitKernel, followers,
)
    dimension = size(first(samples), 1)
    initial = _coupling_encode_components(couplings, kernel.components)
    objective = parameters -> begin
        trial_couplings = _coupling_decode_components(
            parameters, length(energies), dimension, kernel.components,
        )
        error = _coupling_combined_error(
            energies, trial_couplings, samples, frequencies, kernel.alpha, followers,
        )
        return error.aggregate.weighted_l2^2 /
               (length(frequencies) * (length(followers) + 1))
    end
    result = Optim.optimize(
        objective, initial, Optim.BFGS(),
        Optim.Options(iterations=min(kernel.maxiter, 250),
                      f_abstol=kernel.optimizer_tolerance,
                      g_abstol=kernel.optimizer_tolerance,
                      show_trace=false, store_trace=false),
    )
    isfinite(Optim.minimum(result)) || return couplings
    refined = _coupling_decode_components(
        Optim.minimizer(result), length(energies), dimension, kernel.components,
    )
    all(coupling -> all(value -> isfinite(real(value)) && isfinite(imag(value)),
                        coupling), refined) || return couplings
    return refined
end

function _coupling_snap_declared_boundaries(
    energies::Vector{Float64}, couplings::Vector{Vector{ComplexF64}},
    mode_lower::Vector{Float64}, mode_upper::Vector{Float64},
    declared_bounds::Tuple{Float64,Float64}, samples::Vector{Matrix{ComplexF64}},
    frequencies::Vector{Float64}, kernel::CouplingFitKernel, followers,
)
    current = _coupling_combined_error(
        energies, couplings, samples, frequencies, kernel.alpha, followers,
    )
    snapped = falses(length(energies))
    for mode in eachindex(energies)
        candidates = Float64[]
        mode_lower[mode] == declared_bounds[1] &&
            push!(candidates, mode_lower[mode])
        mode_upper[mode] == declared_bounds[2] &&
            push!(candidates, mode_upper[mode])
        for candidate in candidates
            trial = copy(energies)
            trial[mode] = candidate
            trial_couplings = _coupling_refine_fixed_energies(
                trial, couplings, samples, frequencies, kernel, followers,
            )
            trial_error = _coupling_combined_error(
                trial, trial_couplings, samples, frequencies, kernel.alpha, followers,
            )
            if trial_error.aggregate.weighted_l2 < current.aggregate.weighted_l2
                energies = trial
                couplings = trial_couplings
                current = trial_error
                snapped[mode] = true
            end
        end
    end
    return energies, couplings, current, snapped
end

function _coupling_fit_block(samples::Vector{Matrix{ComplexF64}},
                             frequencies::Vector{Float64}, kernel::CouplingFitKernel,
                             block::Symbol, energy_bounds::Tuple{Float64,Float64};
                             followers=NamedTuple[])
    dimension = size(first(samples), 1)
    if _coupling_zero_block(samples) &&
       all(follower -> _coupling_zero_block(follower.samples), followers)
        diagnostic = (; block, status=:zero_sequence, selected_modes=0,
                       energies=Float64[], couplings=Vector{ComplexF64}[],
                       error=(; maximum=0.0, weighted_l2=0.0,
                              target_weighted_l2=0.0, relative_l2=0.0),
                       optimizer=(; converged=true, iterations=0, objective=0.0))
        return (; energies=Float64[], couplings=Vector{Vector{ComplexF64}}[],
                residues=Matrix{ComplexF64}[], diagnostic)
    end
    lower, upper = _coupling_mode_bounds(energy_bounds, kernel)
    initial_energies = _coupling_initial_energies(lower, upper)
    initial_samples = if _coupling_zero_block(samples) && !isempty(followers)
        follower = first(followers)
        follower.tie.relation isa EqualTie ? follower.samples :
            Matrix{ComplexF64}[ComplexF64.(conj.(sample)) for sample in follower.samples]
    else
        samples
    end
    initial_couplings = _coupling_initial_couplings(
        initial_samples, frequencies, kernel.n_modes, kernel.components,
    )
    initial = _coupling_encode(
        initial_energies, initial_couplings, lower, upper,
        kernel.components,
    )
    objective = parameters -> _coupling_objective(
        parameters, kernel.n_modes, dimension, lower, upper,
        kernel.components, samples, frequencies, kernel.alpha, followers,
    )
    result = Optim.optimize(
        objective, initial, Optim.BFGS(),
        Optim.Options(iterations=kernel.maxiter,
                      f_abstol=kernel.optimizer_tolerance,
                      g_abstol=kernel.optimizer_tolerance,
                      show_trace=false, store_trace=false),
    )
    isfinite(Optim.minimum(result)) || throw(ArgumentError(
        "CouplingFitKernel optimizer produced a nonfinite objective for block $block",
    ))
    energies, couplings = _coupling_decode(
        Optim.minimizer(result), kernel.n_modes, dimension, lower, upper,
        kernel.components,
    )
    order = sortperm(eachindex(energies); by=index -> (energies[index], index))
    energies = energies[order]
    couplings = couplings[order]
    lower = lower[order]
    upper = upper[order]
    energies, couplings, errors, boundary_snaps = _coupling_snap_declared_boundaries(
        energies, couplings, lower, upper, energy_bounds, samples, frequencies,
        kernel, followers,
    )
    error = errors.aggregate
    all(isfinite, energies) &&
        all(coupling -> all(value -> isfinite(real(value)) && isfinite(imag(value)),
                            coupling), couplings) || throw(ArgumentError(
            "CouplingFitKernel optimizer produced nonfinite bath parameters for block $block",
        ))
    _coupling_require_tolerance(errors, kernel.fit_tolerance, block)
    residues = Matrix{ComplexF64}[coupling * coupling' for coupling in couplings]
    converged = Optim.converged(result)
    diagnostic = (; block, status=converged ? :fitted : :nonconverged,
                   selected_modes=length(energies),
                   energies, couplings, error, source_error=errors.source_error,
                   follower_errors=errors.follower_errors,
                   optimizer=(; converged,
                              iterations=Optim.iterations(result),
                              objective=Optim.minimum(result),
                              postprocess_objective=error.weighted_l2^2 /
                                                    (length(frequencies) *
                                                     (length(followers) + 1))),
                   frequency_count=length(frequencies), alpha=kernel.alpha,
                   objective=:weighted_squared_frobenius,
                   components=kernel.components,
                   energy_bounds, boundary_snaps)
    return (; energies, couplings, residues, diagnostic)
end

function _coupling_tied_block(source_fit, tie::CouplingBlockTie,
                              samples::Vector{Matrix{ComplexF64}},
                              frequencies::Vector{Float64}, kernel::CouplingFitKernel)
    couplings = _coupling_related_vectors(source_fit.couplings, tie.relation)
    energies = copy(source_fit.energies)
    values = isempty(couplings) ?
        Matrix{ComplexF64}[zeros(ComplexF64, size(first(samples))...)
                            for _ in frequencies] :
        _coupling_values(energies, couplings, frequencies)
    error = _coupling_error(values, samples, frequencies, kernel.alpha)
    residues = Matrix{ComplexF64}[coupling * coupling' for coupling in couplings]
    diagnostic = (; block=tie.target, status=:tied, source=tie.source,
                   relation=tie.relation, selected_modes=length(energies),
                   energies, couplings, error,
                   optimizer=(; converged=true, iterations=0, objective=error.weighted_l2^2),
                   frequency_count=length(frequencies), alpha=kernel.alpha,
                   components=kernel.components)
    return (; energies, couplings, residues, diagnostic)
end

"""
    real_pole_bath_fit(input, kernel::CouplingFitKernel, partition) -> PoleExpansion

Fit each independent named fermionic Matsubara block directly in its real
energies and complex coupling vectors. Every returned residue is formed as
`V * V'`, so off-diagonal matrix data is retained and Hamiltonian
realizability is PSD-by-construction. `realize_bath` remains the shared
canonical factorization and ownership gate.
"""
function real_pole_bath_fit(input::BathFitInput, kernel::CouplingFitKernel,
                            partition::Partition)
    order, selected, frequencies = _coupling_selected_grid(input, kernel, partition)
    bounds = _coupling_energy_bounds(kernel, frequencies)
    target_blocks = Set(tie.target for tie in kernel.block_ties)
    ties_by_source = Dict{Symbol,Vector{CouplingBlockTie}}()
    for tie in kernel.block_ties
        push!(get!(ties_by_source, tie.source, CouplingBlockTie[]), tie)
    end
    fits = Dict{Symbol,NamedTuple}()
    for block in block_names(partition)
        block in target_blocks && continue
        samples = _fit_block_samples(input, block)[order][selected]
        followers = NamedTuple[]
        for tie in get(ties_by_source, block, CouplingBlockTie[])
            follower_samples = _fit_block_samples(input, tie.target)[order][selected]
            push!(followers, (; tie, samples=follower_samples))
        end
        fits[block] = _coupling_fit_block(
            samples, frequencies, kernel, block, bounds; followers,
        )
    end
    for tie in kernel.block_ties
        haskey(fits, tie.source) || throw(ArgumentError(
            "CouplingBlockTie source $(tie.source) was not independently fitted",
        ))
        samples = _fit_block_samples(input, tie.target)[order][selected]
        fits[tie.target] = _coupling_tied_block(
            fits[tie.source], tie, samples, frequencies, kernel,
        )
    end
    poles = Float64[]
    residues = Matrix{ComplexF64}[]
    block_indices = Int[]
    diagnostics = NamedTuple[]
    for (block_index_value, block) in enumerate(block_names(partition))
        fit = get(fits, block, nothing)
        fit === nothing && throw(ArgumentError(
            "CouplingFitKernel has no result for block $block",
        ))
        append!(poles, fit.energies)
        append!(residues, fit.residues)
        append!(block_indices, fill(block_index_value, length(fit.energies)))
        push!(diagnostics, fit.diagnostic)
    end
    raw = BlockRealPoles(input.layout, partition, poles, residues, block_indices;
                         statistics=input.statistics)
    return PoleExpansion(
        raw;
        kernel=:coupling_fit,
        trace=(; plan=DiscretizationPlan(partition), fits=diagnostics,
               alpha=kernel.alpha, frequency_window=kernel.frequency_window,
               energy_bounds=bounds, allocation=kernel.allocation,
               components=kernel.components,
               block_ties=kernel.block_ties, source_metadata=input.metadata),
    )
end
