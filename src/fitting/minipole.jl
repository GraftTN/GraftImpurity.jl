"""
Real-Hamiltonian MiniPole adapter.

This file adapts a conformal rational Matsubara interpolant into a finite
matrix moment sequence and then invokes the shared matrix-ESPRIT engine. It
does not implement or expose a complex BCF result: use `fit_complex_bcf` with
`BCFFitInput` for that distinct typed contract.
"""

function _minipole_scale(frequencies::Vector{Float64}, kernel::MiniPoleKernel)
    kernel.conformal_scale === nothing || return kernel.conformal_scale
    nonzero = sort(Float64[abs(frequency) for frequency in frequencies
                            if !iszero(frequency)])
    isempty(nonzero) && throw(ArgumentError(
        "MiniPole real-pole fitting needs a nonzero Matsubara frequency to choose a conformal scale",
    ))
    return sqrt(first(nonzero) * last(nonzero))
end

function _minipole_scale_candidates(frequencies::Vector{Float64},
                                    kernel::MiniPoleKernel)
    preferred = _minipole_scale(frequencies, kernel)
    factors = (1.0, sqrt(2.0), inv(sqrt(2.0)), 2.0, 0.5)
    candidates = Float64[]
    for factor in factors
        candidate = preferred * factor
        isfinite(candidate) && candidate > 0 || continue
        candidate in candidates || push!(candidates, candidate)
    end
    return candidates
end

function _minipole_uniform_matsubara(input::BathFitInput)
    permutation = sortperm(input.frequencies)
    frequencies = input.frequencies[permutation]
    length(frequencies) >= 4 || throw(ArgumentError(
        "MiniPole real-pole fitting needs at least four Matsubara samples",
    ))
    all(value -> value >= 0, frequencies) || throw(ArgumentError(
        "MiniPole real-pole fitting requires a nonnegative Matsubara grid",
    ))
    steps = diff(frequencies)
    all(step -> step > 0, steps) ||
        throw(ArgumentError("MiniPole Matsubara frequencies must be distinct"))
    step = first(steps)
    all(candidate -> isapprox(candidate, step;
                              atol=32 * eps(Float64) * max(1.0, abs(step)),
                              rtol=32 * eps(Float64)), steps) ||
        throw(ArgumentError(
            "MiniPole real-pole fitting requires a uniformly spaced Matsubara grid",
        ))
    return permutation, frequencies
end

function _minipole_conformal_coordinates(frequencies::Vector{Float64},
                                          scale::Float64)
    z = ComplexF64.(im .* frequencies)
    coordinates = (z .- scale) ./ (z .+ scale)
    all(value -> isfinite(real(value)) && isfinite(imag(value)), coordinates) ||
        throw(ArgumentError("MiniPole conformal map produced a nonfinite value"))
    return coordinates
end

"""
Fit one common monic conformal denominator and entry-local numerators. The
`(1-w)` factor encodes the finite-bath high-frequency decay and leaves no
data-dependent diagonal or off-diagonal projection in this interpolation stage.
"""
function _minipole_shared_denominator(samples::Vector{Matrix{ComplexF64}},
                                      coordinates::Vector{ComplexF64},
                                      pole_count::Int)
    sample_count = length(samples)
    dimension = size(first(samples), 1)
    entries = dimension^2
    columns = pole_count * (entries + 1)
    sample_count * entries >= columns || throw(ArgumentError(
        "MiniPole has insufficient Matsubara samples for $pole_count shared-denominator poles",
    ))
    system = zeros(ComplexF64, sample_count * entries, columns)
    rhs = zeros(ComplexF64, sample_count * entries)
    for sample_index in 1:sample_count
        coordinate = coordinates[sample_index]
        powers = ComplexF64[coordinate^(power - 1) for power in 1:(pole_count + 1)]
        sample = samples[sample_index]
        for entry in 1:entries
            row = (sample_index - 1) * entries + entry
            value = sample[entry]
            @views system[row, 1:pole_count] .= value .* powers[1:pole_count]
            numerator_offset = pole_count + (entry - 1) * pole_count
            @views system[row, (numerator_offset + 1):(numerator_offset + pole_count)] .=
                -(1 - coordinate) .* powers[1:pole_count]
            rhs[row] = -value * powers[end]
        end
    end
    solution = system \ rhs
    return Vector{ComplexF64}(solution[1:pole_count]),
           norm(system * solution - rhs)
end

function _minipole_companion_roots(denominator::Vector{ComplexF64})
    count = length(denominator)
    companion = zeros(ComplexF64, count, count)
    for index in 1:(count - 1)
        companion[index + 1, index] = 1
    end
    companion[:, end] .= -denominator
    return ComplexF64.(eigen(companion).values)
end

function _minipole_mapped_real_energies(roots::Vector{ComplexF64},
                                         scale::Float64, tolerance::Float64)
    energies = Float64[]
    for root in roots
        abs(imag(root)) <= tolerance * max(1.0, abs(real(root))) ||
            throw(ArgumentError(
                "MiniPole real-pole route recovered a complex conformal node; use the typed BCF route for complex exponential data",
            ))
        denominator = 1 - root
        abs(denominator) > tolerance || throw(ArgumentError(
            "MiniPole conformal node maps to an infinite real-pole energy",
        ))
        energy = scale * (1 + root) / denominator
        abs(imag(energy)) <= tolerance * max(1.0, abs(real(energy))) ||
            throw(ArgumentError("MiniPole mapped energy is not real within tolerance"))
        finite_energy = Float64(real(energy))
        isfinite(finite_energy) ||
            throw(ArgumentError("MiniPole mapped energy is not finite"))
        push!(energies, finite_energy)
    end
    ordered = sort(energies)
    all(index -> abs(ordered[index + 1] - ordered[index]) >
                 tolerance * max(1.0, abs(ordered[index]), abs(ordered[index + 1])),
        1:(length(ordered) - 1)) || throw(ArgumentError(
        "MiniPole produced repeated real energies; reduce the requested rank",
    ))
    return energies
end

function _minipole_real_residues(samples::Vector{Matrix{ComplexF64}},
                                 frequencies::Vector{Float64},
                                 energies::Vector{Float64})
    dimension = size(first(samples), 1)
    system = ComplexF64[
        inv(im * frequency - energy)
        for frequency in frequencies, energy in energies
    ]
    residues = [zeros(ComplexF64, dimension, dimension) for _ in energies]
    for entry in 1:(dimension^2)
        values = ComplexF64[sample[entry] for sample in samples]
        weights = system \ values
        for pole_index in eachindex(energies)
            residues[pole_index][entry] = weights[pole_index]
        end
    end
    return residues
end

function _minipole_real_error(samples::Vector{Matrix{ComplexF64}},
                              frequencies::Vector{Float64},
                              energies::Vector{Float64},
                              residues::Vector{Matrix{ComplexF64}})
    isempty(samples) && return (; maximum=0.0, l2=0.0, relative_l2=0.0)
    maximum_error = 0.0
    squared_error = 0.0
    squared_target = 0.0
    for (frequency, sample) in zip(frequencies, samples)
        value = zeros(ComplexF64, size(sample)...)
        for (energy, residue) in zip(energies, residues)
            value .+= residue ./ (im * frequency - energy)
        end
        difference = norm(value - sample)
        maximum_error = max(maximum_error, difference)
        squared_error += difference^2
        squared_target += norm(sample)^2
    end
    l2 = sqrt(squared_error)
    return (; maximum=maximum_error, l2,
            relative_l2=squared_target == 0 ? l2 : l2 / sqrt(squared_target))
end

function _minipole_conformal_moment_weights(nodes::Vector{ComplexF64},
                                             energies::Vector{Float64},
                                             residues::Vector{Matrix{ComplexF64}},
                                             scale::Float64,
                                             tolerance::Float64)
    weights = Matrix{ComplexF64}[]
    for (node, energy, residue) in zip(nodes, energies, residues)
        denominator = scale + energy
        abs(denominator) > tolerance || throw(ArgumentError(
            "MiniPole conformal scale is singular for a recovered real energy",
        ))
        push!(weights, ((1 - node) / denominator) .* residue)
    end
    return weights
end

function _minipole_real_prune(energies::Vector{Float64},
                              residues::Vector{Matrix{ComplexF64}},
                              tolerance::Float64)
    norms = Float64[norm(residue) for residue in residues]
    scale = isempty(norms) ? 0.0 : maximum(norms)
    cutoff = scale == 0 ? 0.0 : max(tolerance * scale, 64 * eps(Float64) * scale)
    retained = Int[index for index in eachindex(energies) if norms[index] > cutoff]
    discarded = Int[index for index in eachindex(energies) if !(index in retained)]
    isempty(retained) && throw(ArgumentError(
        "MiniPole real-pole fit contains only negligible residues",
    ))
    return retained, discarded, norms, cutoff
end

function _minipole_zero_real_block_fit(samples::Vector{Matrix{ComplexF64}},
                                       frequencies::Vector{Float64},
                                       kernel::MiniPoleKernel, block::Symbol,
                                       training_count::Int)
    empty_energies = Float64[]
    empty_residues = Matrix{ComplexF64}[]
    training_error = (; maximum=0.0, l2=0.0, relative_l2=0.0)
    holdout_error = kernel.holdout_count == 0 ? training_error :
        _minipole_real_error(
            samples[(training_count + 1):end],
            frequencies[(training_count + 1):end], empty_energies, empty_residues,
        )
    iszero(holdout_error.maximum) || throw(ArgumentError(
        "MiniPole real-pole zero training sequence has nonzero held-out samples for block $block",
    ))
    attempt = (; scale=nothing, rank=0, status=:zero_sequence,
               training_error, holdout_error)
    return (; energies=empty_energies, residues=empty_residues,
            diagnostic=(; block, requested_poles=kernel.n_poles,
                        selected_poles=0, conformal_scale=nothing,
                        training_count, holdout_count=kernel.holdout_count,
                        attempts=NamedTuple[attempt], selected_attempt=attempt))
end

function _minipole_real_block_fit(samples::Vector{Matrix{ComplexF64}},
                                  frequencies::Vector{Float64},
                                  kernel::MiniPoleKernel, block::Symbol)
    training_count = length(frequencies) - kernel.holdout_count
    training_count >= 4 || throw(ArgumentError(
        "MiniPole holdout_count leaves fewer than four real-pole training samples",
    ))
    training_samples = samples[1:training_count]
    training_frequencies = frequencies[1:training_count]
    all(sample -> iszero(norm(sample)), training_samples) &&
        return _minipole_zero_real_block_fit(
            samples, frequencies, kernel, block, training_count,
        )
    dimension = size(first(samples), 1)
    entries = dimension^2
    maximum_by_equations = fld(training_count * entries, entries + 1)
    maximum_rank = min(kernel.n_poles, maximum_by_equations)
    maximum_rank >= 1 || throw(ArgumentError(
        "MiniPole has insufficient data for a shared real-pole denominator",
    ))
    domain_tolerance = sqrt(eps(Float64))
    attempts = NamedTuple[]

    for scale in _minipole_scale_candidates(training_frequencies, kernel)
        coordinates = _minipole_conformal_coordinates(training_frequencies, scale)
        for rank in maximum_rank:-1:1
            denominator, denominator_residual = _minipole_shared_denominator(
                training_samples, coordinates, rank,
            )
            roots = _minipole_companion_roots(denominator)
            try
                preliminary_energies = _minipole_mapped_real_energies(
                    roots, scale, domain_tolerance,
                )
                preliminary_residues = _minipole_real_residues(
                    training_samples, training_frequencies, preliminary_energies,
                )
                moment_weights = _minipole_conformal_moment_weights(
                    roots, preliminary_energies, preliminary_residues, scale,
                    domain_tolerance,
                )
                moment_count = max(4, 2 * length(roots) + 2)
                moments = _minipole_sequence_values(roots, moment_weights, moment_count)
                moment_kernel = MiniPoleKernel(
                    n_poles=rank, rank_tolerance=kernel.rank_tolerance,
                    conformal_scale=scale, holdout_count=0,
                )
                moment_fit = _minipole_exponential_fit(moments, moment_kernel)
                refined_energies = _minipole_mapped_real_energies(
                    moment_fit.nodes, scale, domain_tolerance,
                )
                order = sortperm(refined_energies)
                ordered_energies = refined_energies[order]
                residues = _minipole_real_residues(
                    training_samples, training_frequencies, ordered_energies,
                )
                retained, discarded, residue_norms, cutoff = _minipole_real_prune(
                    ordered_energies, residues, kernel.rank_tolerance,
                )
                pruned_energies = ordered_energies[retained]
                pruned_residues = _minipole_real_residues(
                    training_samples, training_frequencies, pruned_energies,
                )
                training_error = _minipole_real_error(
                    training_samples, training_frequencies, pruned_energies,
                    pruned_residues,
                )
                holdout_error = kernel.holdout_count == 0 ?
                    (; maximum=0.0, l2=0.0, relative_l2=0.0) :
                    _minipole_real_error(
                        samples[(training_count + 1):end],
                        frequencies[(training_count + 1):end], pruned_energies,
                        pruned_residues,
                    )
                kernel.fit_tolerance === nothing ||
                    training_error.relative_l2 <= kernel.fit_tolerance ||
                    throw(ArgumentError(
                        "MiniPole real-pole training error exceeds fit_tolerance",
                    ))
                accepted = (; scale, rank, status=:accepted, denominator,
                            denominator_residual, roots, preliminary_energies,
                            moment_engine=moment_fit.diagnostics,
                            energies_before_pruning=ordered_energies,
                            retained_indices=retained, discarded_indices=discarded,
                            residue_norms, pruning_threshold=cutoff,
                            energies=pruned_energies, residues=pruned_residues,
                            training_error, holdout_error)
                push!(attempts, accepted)
                return (; energies=pruned_energies, residues=pruned_residues,
                        diagnostic=(; block, requested_poles=kernel.n_poles,
                                    selected_poles=length(pruned_energies),
                                    conformal_scale=scale,
                                    training_count,
                                    holdout_count=kernel.holdout_count,
                                    attempts, selected_attempt=accepted))
            catch error
                error isa ArgumentError || rethrow()
                push!(attempts, (; scale, rank, status=:rejected,
                                  denominator, denominator_residual, roots,
                                  reason=sprint(showerror, error)))
            end
        end
    end
    throw(ArgumentError(
        "MiniPole real-pole route found no finite real Hamiltonian candidate for block $block; use fit_complex_bcf for BCF exponential data when appropriate",
    ))
end

"""
    real_pole_bath_fit(input, kernel::MiniPoleKernel, partition)

Fit a finite real-pole Hamiltonian expansion. The adapter uses the shared
matrix-ESPRIT engine only on a conformal moment sequence and retains full raw
matrix residues for the common `realize_bath` PSD/LDL-dagger gate. It rejects
complex or nonfinite energy candidates as incompatible with this *selected*
Hamiltonian route; `fit_complex_bcf` is the explicit complex BCF route. The
real-Hamiltonian adapter currently has the fermionic resolvent convention;
bosonic Matsubara inputs are explicitly directed to `PESKernel`.
"""
function real_pole_bath_fit(input::BathFitInput, kernel::MiniPoleKernel,
                            partition::Partition)
    started = time_ns()
    _validate_matsubara_fit(input, partition; kernel_name="MiniPoleKernel")
    input.statistics === :fermion || throw(ArgumentError(
        "MiniPoleKernel real-Hamiltonian output currently implements only the fermionic resolvent convention; use PESKernel for the bosonic convention",
    ))
    permutation, frequencies = _minipole_uniform_matsubara(input)
    poles = Float64[]
    residues = Matrix{ComplexF64}[]
    block_indices = Int[]
    fits = NamedTuple[]
    for (block_index_value, block) in enumerate(block_names(partition))
        fit = _minipole_real_block_fit(
            _fit_block_samples(input, block)[permutation], frequencies,
            kernel, block,
        )
        append!(poles, fit.energies)
        append!(residues, fit.residues)
        append!(block_indices, fill(block_index_value, length(fit.energies)))
        push!(fits, fit.diagnostic)
    end
    raw = BlockRealPoles(input.layout, partition, poles, residues, block_indices;
                         statistics=input.statistics)
    expansion = PoleExpansion(
        raw;
        kernel=:minipole,
        trace=(; plan=DiscretizationPlan(partition), fits,
               algorithm=:minipole_conformal_matrix_esprit,
               source_metadata=input.metadata),
    )
    return _with_fit_timing(expansion, started)
end
