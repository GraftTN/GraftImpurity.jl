function _esprit_tau_imaginary_time_grid(input::BathFitInput,
                                         partition::Partition)
    _validate_fit_input(input, partition)
    input.domain === :imaginary_time || throw(ArgumentError(
        "ESPRITTauKernel requires BathFitInput domain=:imaginary_time",
    ))
    input.statistics === :fermion || throw(ArgumentError(
        "ESPRITTauKernel implements only fermionic imaginary-time hybridization fitting",
    ))
    taus = input.frequencies
    length(taus) >= 4 || throw(ArgumentError(
        "ESPRITTauKernel needs at least four imaginary-time samples",
    ))
    beta = _bathfit_imaginary_time_beta(taus)
    steps = diff(taus)
    timestep = first(steps)
    spacing_tolerance = 64 * eps(Float64) * max(beta, timestep, 1.0)
    all(step -> isapprox(step, timestep; rtol=sqrt(eps(Float64)),
                         atol=spacing_tolerance), steps) || throw(ArgumentError(
        "ESPRITTauKernel tau grid must be uniformly spaced",
    ))
    return taus, beta, timestep
end

function _esprit_tau_nodes(samples::Vector{Matrix{ComplexF64}},
                           taus::Vector{Float64},
                           kernel::ESPRITTauKernel)
    sample_count = length(samples)
    lag = floor(Int, 0.4 * sample_count)
    row_count = sample_count - lag
    estimator = Graft.ESPRIT(
        rank=Graft.StrictRank(kernel.n_poles, Graft.NumericalRank()),
        reduction=kernel.reduction,
        hankel_rows=row_count,
    )
    estimate = Graft.estimate_nodes(
        estimator, Graft.UniformSequence(taus, samples),
    )
    outcome = estimate.outcome
    if outcome isa Graft.AbstractNodeFailure
        backend = estimate.backend
        common = estimate.common
        message = if backend isa Graft.ESPRITDiagnostics &&
                     kernel.n_poles >= backend.hankel_columns
            "ESPRITTauKernel requested n_poles=$(kernel.n_poles) must be smaller than the $(backend.hankel_columns) available Hankel singular directions"
        elseif outcome isa Graft.NodeEstimationFailure &&
               (outcome.reason === :rank_exceeds_evidence ||
                outcome.reason === :zero_evidence_rank)
            "ESPRITTauKernel requested n_poles=$(kernel.n_poles) exceeds block-Hankel numerical rank $(common.evidence_rank)"
        elseif outcome isa Graft.NodeEstimationFailure &&
               outcome.reason === :deficient_shift_solve
            "ESPRITTauKernel ESPRIT leading shift subspace is rank deficient"
        elseif outcome isa Graft.NodeEstimationFailure &&
               outcome.reason === :nonfinite_nodes
            "ESPRITTauKernel matrix ESPRIT produced a nonfinite shift node"
        elseif outcome isa Graft.NodeEstimationFailure
            "ESPRITTauKernel core ESPRIT node estimation failed: $(outcome.message)"
        else
            "ESPRITTauKernel node-evidence reduction erased the nonzero block signal"
        end
        throw(ArgumentError(message))
    elseif outcome isa Graft.ZeroSequence
        throw(ArgumentError(
            "ESPRITTauKernel zero sequence must use the zero-fit path",
        ))
    elseif !(outcome isa Graft.IdentifiedNodes)
        throw(ArgumentError(
            "ESPRITTauKernel core ESPRIT returned an unsupported node outcome",
        ))
    end

    backend = estimate.backend
    backend isa Graft.ESPRITDiagnostics || error(
        "Graft ESPRIT identified nodes without ESPRIT diagnostics",
    )
    singular_values = backend.singular_values
    leading_values = backend.leading_singular_values
    reduced_channels = estimate.common.reduction.reduced_channels
    evidence_threshold = max(
        backend.hankel_rows * reduced_channels, backend.hankel_columns,
    ) * eps(Float64) * first(singular_values)
    leading_threshold = max(
        backend.hankel_columns - 1, estimate.common.resolved_rank,
    ) * eps(Float64) * first(leading_values)
    hankel = (; lag=backend.hankel_columns - 1,
              row_count=backend.hankel_rows, singular_values,
              evidence_rank=estimate.common.evidence_rank,
              evidence_threshold,
              leading_singular_values=leading_values,
              leading_threshold)
    return outcome.nodes, hankel, estimate
end

function _esprit_tau_energies(nodes::Vector{ComplexF64}, timestep::Float64,
                              tolerance::Float64)
    energies = Float64[]
    for node in nodes
        iszero(node) && throw(ArgumentError(
            "ESPRITTauKernel ESPRIT produced a zero shift node with no finite pole energy",
        ))
        real(node) > 0 || throw(ArgumentError(
            "ESPRITTauKernel ESPRIT shift nodes must have positive real parts",
        ))
        node_imaginary_limit = tolerance * max(abs(real(node)), eps(Float64))
        abs(imag(node)) <= node_imaginary_limit || throw(ArgumentError(
            "ESPRITTauKernel ESPRIT shift node is not real within pole_tolerance",
        ))
        rate = log(node) / timestep
        isfinite(real(rate)) && isfinite(imag(rate)) || throw(ArgumentError(
            "ESPRITTauKernel ESPRIT produced a nonfinite pole energy",
        ))
        imaginary_limit = tolerance * max(abs(real(rate)), inv(timestep))
        abs(imag(rate)) <= imaginary_limit || throw(ArgumentError(
            "ESPRITTauKernel ESPRIT pole has an imaginary energy component larger than pole_tolerance",
        ))
        push!(energies, -real(rate))
    end
    order = sortperm(energies)
    return energies[order], nodes[order]
end

function _esprit_tau_fermi_kernel(tau::Float64, energy::Float64,
                                  beta::Float64)
    if energy >= 0
        return exp(-tau * energy) / (1 + exp(-beta * energy))
    end
    return exp((beta - tau) * energy) / (1 + exp(beta * energy))
end

function _esprit_tau_kernel_matrix(taus::Vector{Float64},
                                   energies::Vector{Float64}, beta::Float64)
    kernel = zeros(Float64, length(taus), length(energies))
    for pole_index in eachindex(energies), sample_index in eachindex(taus)
        kernel[sample_index, pole_index] = _esprit_tau_fermi_kernel(
            taus[sample_index], energies[pole_index], beta,
        )
    end
    all(isfinite, kernel) || throw(ArgumentError(
        "ESPRITTauKernel Fermi kernel contains nonfinite values",
    ))
    return kernel
end

function _esprit_tau_weight_fit(samples::Vector{Matrix{ComplexF64}},
                                kernel::Matrix{Float64})
    sample_count = length(samples)
    dimension = size(first(samples), 1)
    values = zeros(ComplexF64, sample_count, dimension^2)
    for sample_index in eachindex(samples)
        @views values[sample_index, :] .= vec(samples[sample_index])
    end

    column_scales = vec(maximum(abs, kernel; dims=1))
    all(scale -> isfinite(scale) && scale > 0, column_scales) ||
        throw(ArgumentError(
            "ESPRITTauKernel Fermi least-squares columns must have finite positive scales",
        ))
    equilibrated = kernel * Diagonal(inv.(column_scales))
    decomposition = svd(equilibrated; full=false)
    singular_values = Float64.(decomposition.S)
    threshold = max(size(equilibrated)...) * eps(Float64) * first(singular_values)
    numerical_rank = count(value -> value > threshold, singular_values)
    numerical_rank == size(kernel, 2) || throw(ArgumentError(
        "ESPRITTauKernel equilibrated Fermi least-squares matrix is rank deficient",
    ))
    balanced_weights = equilibrated \ values
    solved = Diagonal(inv.(column_scales)) * balanced_weights
    weights = Matrix{ComplexF64}[
        reshape(copy(@view solved[pole_index, :]), dimension, dimension)
        for pole_index in axes(solved, 1)
    ]
    condition = first(singular_values) / last(singular_values)
    return weights, (; column_scales, singular_values, numerical_rank,
                     threshold, condition)
end

function _esprit_tau_reconstruct(kernel::Matrix{Float64},
                                 weights::Vector{Matrix{ComplexF64}},
                                 sign::Float64)
    dimension = isempty(weights) ? 0 : size(first(weights), 1)
    reconstruction = Matrix{ComplexF64}[]
    for sample_index in axes(kernel, 1)
        value = zeros(ComplexF64, dimension, dimension)
        for pole_index in eachindex(weights)
            value .+= sign * kernel[sample_index, pole_index] .* weights[pole_index]
        end
        push!(reconstruction, value)
    end
    return reconstruction
end

function _esprit_tau_fit_error(samples::Vector{Matrix{ComplexF64}},
                               reconstruction::Vector{Matrix{ComplexF64}})
    length(samples) == length(reconstruction) || throw(DimensionMismatch(
        "ESPRITTauKernel error calculation needs matching sample counts",
    ))
    maximum_error = 0.0
    squared_error = 0.0
    squared_target = 0.0
    for (sample, value) in zip(samples, reconstruction)
        difference = value - sample
        maximum_error = max(
            maximum_error, maximum(abs, difference; init=0.0),
        )
        squared_error += sum(abs2, difference)
        squared_target += sum(abs2, sample)
    end
    l2 = sqrt(squared_error)
    relative_l2 = iszero(squared_target) ? (iszero(l2) ? 0.0 : Inf) :
                  l2 / sqrt(squared_target)
    return (; maximum=maximum_error, l2, relative_l2)
end

function _esprit_tau_positive_part(weights::Vector{Matrix{ComplexF64}},
                                   energies::Vector{Float64},
                                   tolerance::Float64)
    residues = Matrix{ComplexF64}[]
    diagnostics = NamedTuple[]
    for (pole_index, (weight, energy)) in enumerate(zip(weights, energies))
        hermitian_weight = -(weight + adjoint(weight)) / 2
        decomposition = eigen(Hermitian(hermitian_weight))
        eigenvalues = Float64.(real.(decomposition.values))
        scale = maximum(abs, eigenvalues; init=0.0)
        threshold = tolerance * scale
        projected_eigenvalues = Float64[
            value > threshold ? value : 0.0 for value in eigenvalues
        ]
        projected = decomposition.vectors * Diagonal(projected_eigenvalues) *
                    adjoint(decomposition.vectors)
        residue = Matrix{ComplexF64}(Hermitian(projected))
        push!(residues, residue)
        push!(diagnostics,
              (; pole_index, energy, eigenvalues,
               projected_eigenvalues, scale, threshold,
               retained_rank=count(>(0), projected_eigenvalues),
               antihermitian_norm=norm(weight - adjoint(weight)) / 2,
               negative_eigenvalue_weight=sum(value -> max(-value, 0.0),
                                              eigenvalues),
               subthreshold_positive_weight=sum(
                   value -> 0 < value <= threshold ? value : 0.0,
                   eigenvalues,
               ),
               correction_norm=norm(residue - hermitian_weight)))
    end
    return residues, diagnostics
end

function _esprit_tau_zero_fit(block::Symbol, requested_poles::Int)
    empty_error = (; maximum=0.0, l2=0.0, relative_l2=0.0)
    return (; block, status=:zero_sequence, requested_poles,
            selected_poles=0, energies=Float64[], nodes=ComplexF64[],
            singular_values=Float64[], raw_weights=Matrix{ComplexF64}[],
            projection_diagnostics=NamedTuple[], raw_error=empty_error,
            physical_error=empty_error,
            hankel=(; lag=0, row_count=0, singular_values=Float64[],
                     evidence_rank=0, evidence_threshold=0.0,
                     leading_singular_values=Float64[],
                     leading_threshold=0.0),
            least_squares=(; column_scales=Float64[],
                            singular_values=Float64[], numerical_rank=0,
                            threshold=0.0, condition=1.0))
end

function _esprit_tau_block_fit(samples::Vector{Matrix{ComplexF64}},
                               taus::Vector{Float64}, beta::Float64,
                               timestep::Float64,
                               kernel::ESPRITTauKernel, block::Symbol)
    if all(sample -> iszero(norm(sample)), samples)
        return (; energies=Float64[], residues=Matrix{ComplexF64}[],
                diagnostic=_esprit_tau_zero_fit(block, kernel.n_poles))
    end

    nodes, hankel, node_estimate = _esprit_tau_nodes(
        samples, taus, kernel,
    )
    energies, ordered_nodes = _esprit_tau_energies(
        nodes, timestep, kernel.pole_tolerance,
    )
    fermi_kernel = _esprit_tau_kernel_matrix(taus, energies, beta)
    raw_weights, least_squares = _esprit_tau_weight_fit(samples, fermi_kernel)
    raw_error = _esprit_tau_fit_error(
        samples, _esprit_tau_reconstruct(fermi_kernel, raw_weights, 1.0),
    )
    residues, projection_diagnostics = _esprit_tau_positive_part(
        raw_weights, energies, kernel.projection_tolerance,
    )
    physical_error = _esprit_tau_fit_error(
        samples, _esprit_tau_reconstruct(fermi_kernel, residues, -1.0),
    )
    kernel.fit_tolerance === nothing ||
        physical_error.relative_l2 <= kernel.fit_tolerance || throw(ArgumentError(
            "ESPRITTauKernel physical projected fit for block $block exceeds fit_tolerance",
        ))
    diagnostic = (; block, status=:accepted,
                  requested_poles=kernel.n_poles,
                  selected_poles=length(energies), energies,
                  nodes=ordered_nodes,
                  singular_values=hankel.singular_values,
                  raw_weights, projection_diagnostics,
                  raw_error, physical_error, hankel, least_squares,
                  node_estimate)
    return (; energies, residues, diagnostic)
end

"""
    real_pole_bath_fit(input, kernel::ESPRITTauKernel, partition)

Fit each named fermionic imaginary-time hybridization block independently with
the NumericsSOE-style matrix ESPRIT construction. Fixed real energies
are refit with the finite-temperature Fermi kernel, then every raw weight is
explicitly mapped to the PSD positive part of `-Hermitian(W)`. The result is a
normal Hamiltonian `PoleExpansion`; it has no residual-SOE semantics.
"""
function real_pole_bath_fit(input::BathFitInput,
                            kernel::ESPRITTauKernel,
                            partition::Partition)
    started = time_ns()
    taus, beta, timestep = _esprit_tau_imaginary_time_grid(input, partition)
    poles = Float64[]
    residues = Matrix{ComplexF64}[]
    block_indices = Int[]
    fits = NamedTuple[]
    for (block_index_value, block) in enumerate(block_names(partition))
        fit = _esprit_tau_block_fit(
            _fit_block_samples(input, block), taus, beta, timestep, kernel, block,
        )
        append!(poles, fit.energies)
        append!(residues, fit.residues)
        append!(block_indices,
                fill(block_index_value, length(fit.energies)))
        push!(fits, fit.diagnostic)
    end
    raw = BlockRealPoles(input.layout, partition, poles, residues, block_indices;
                         statistics=input.statistics)
    expansion = PoleExpansion(
        raw;
        kernel=:esprit_tau,
        trace=(; plan=DiscretizationPlan(partition),
               method=:imaginary_time_esprit, fits,
               source_metadata=input.metadata),
    )
    return _with_fit_timing(expansion, started)
end
