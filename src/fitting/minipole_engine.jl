"""
Internal clean-room matrix-ESPRIT engine shared by MiniPole output routes.

The engine accepts a uniformly sampled scalar or matrix exponential sequence,
stacks flattened matrix samples into a block-Hankel matrix, recovers shared
shift nodes through an SVD shift-invariance solve, and obtains full matrix
weights by least squares. It deliberately has no bath realization semantics.
"""

function _minipole_hankel(samples::Vector{Matrix{ComplexF64}})
    sample_count = length(samples)
    sample_count >= 4 || throw(ArgumentError(
        "MiniPole matrix ESPRIT needs at least four uniformly sampled values",
    ))
    dimension = size(first(samples), 1)
    all(sample -> size(sample) == (dimension, dimension), samples) ||
        throw(DimensionMismatch("MiniPole matrix samples need a common square dimension"))

    lag = clamp(fld(2 * (sample_count - 1), 5), 1, fld(sample_count - 1, 2))
    row_count = sample_count - lag
    component_count = dimension^2
    hankel = zeros(ComplexF64, component_count * row_count, lag + 1)
    for row in 1:row_count
        offset = (row - 1) * component_count
        for column in 1:(lag + 1)
            @views hankel[(offset + 1):(offset + component_count), column] .=
                vec(samples[row + column - 1])
        end
    end
    return hankel, lag, row_count
end

function _minipole_effective_rank(singular_values::Vector{Float64},
                                  tolerance::Float64)
    isempty(singular_values) && return 0
    scale = first(singular_values)
    iszero(scale) && return 0
    return count(value -> value > tolerance * scale, singular_values)
end

function _minipole_shift_nodes(u, component_count::Int, rank::Int)
    rank > 0 || throw(ArgumentError("MiniPole matrix ESPRIT rank must be positive"))
    left_subspace = Matrix{ComplexF64}(u[:, 1:rank])
    size(left_subspace, 1) > component_count || throw(ArgumentError(
        "MiniPole matrix ESPRIT needs at least two block-Hankel row groups",
    ))
    leading = @view left_subspace[1:(end - component_count), :]
    trailing = @view left_subspace[(component_count + 1):end, :]
    shift = leading \ trailing
    nodes = ComplexF64.(eigen(shift).values)
    all(node -> isfinite(real(node)) && isfinite(imag(node)), nodes) ||
        throw(ArgumentError("MiniPole matrix ESPRIT produced a nonfinite shift node"))
    return nodes
end

function _minipole_vandermonde(nodes::Vector{ComplexF64}, sample_count::Int)
    vandermonde = zeros(ComplexF64, sample_count, length(nodes))
    for sample_index in 1:sample_count, node_index in eachindex(nodes)
        vandermonde[sample_index, node_index] = nodes[node_index]^(sample_index - 1)
    end
    return vandermonde
end

function _minipole_weight_fit(samples::Vector{Matrix{ComplexF64}},
                              nodes::Vector{ComplexF64})
    dimension = size(first(samples), 1)
    sample_count = length(samples)
    vandermonde = _minipole_vandermonde(nodes, sample_count)
    values = zeros(ComplexF64, sample_count, dimension^2)
    for sample_index in 1:sample_count
        @views values[sample_index, :] .= vec(samples[sample_index])
    end
    solved = vandermonde \ values
    weights = Matrix{ComplexF64}[
        reshape(copy(@view solved[node_index, :]), dimension, dimension)
        for node_index in eachindex(nodes)
    ]
    singular_values = Float64.(svd(vandermonde; full=false).S)
    threshold = isempty(singular_values) ? 0.0 :
        max(64 * eps(Float64) * first(singular_values), eps(Float64))
    rank = count(value -> value > threshold, singular_values)
    condition = isempty(singular_values) || last(singular_values) <= threshold ?
        Inf : first(singular_values) / last(singular_values)
    return weights, (; rank, condition, singular_values)
end

function _minipole_sequence_values(nodes::Vector{ComplexF64},
                                   weights::Vector{Matrix{ComplexF64}},
                                   sample_count::Int)
    dimension = isempty(weights) ? 0 : size(first(weights), 1)
    values = Matrix{ComplexF64}[]
    for sample_index in 1:sample_count
        value = zeros(ComplexF64, dimension, dimension)
        for (node, weight) in zip(nodes, weights)
            value .+= node^(sample_index - 1) .* weight
        end
        push!(values, value)
    end
    return values
end

function _minipole_sequence_error(samples::Vector{Matrix{ComplexF64}},
                                  nodes::Vector{ComplexF64},
                                  weights::Vector{Matrix{ComplexF64}};
                                  start_index::Int=1)
    start_index >= 1 ||
        throw(ArgumentError("MiniPole sequence start index must be positive"))
    isempty(samples) && return (; maximum=0.0, l2=0.0, relative_l2=0.0)
    maximum_error = 0.0
    squared_error = 0.0
    squared_target = 0.0
    for (offset, sample) in enumerate(samples)
        value = zeros(ComplexF64, size(sample)...)
        exponent = start_index + offset - 2
        for (node, weight) in zip(nodes, weights)
            value .+= node^exponent .* weight
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

function _minipole_prune_modes(nodes::Vector{ComplexF64},
                               weights::Vector{Matrix{ComplexF64}},
                               tolerance::Float64)
    norms = Float64[norm(weight) for weight in weights]
    scale = isempty(norms) ? 0.0 : maximum(norms)
    cutoff = scale == 0 ? 0.0 : max(tolerance * scale, 64 * eps(Float64) * scale)
    retained = Int[index for index in eachindex(nodes) if norms[index] > cutoff]
    discarded = Int[index for index in eachindex(nodes) if !(index in retained)]
    return retained, discarded, norms, cutoff
end

function _minipole_reorder(nodes::Vector{ComplexF64},
                            weights::Vector{Matrix{ComplexF64}},
                            order::AbstractVector{<:Integer})
    return nodes[order], weights[order]
end

"""
    _minipole_exponential_fit(samples, kernel)

Fit a uniform exponential sequence. This private engine is deliberately route
neutral: callers decide whether its nodes represent BCF decay factors or a
conformal-moment sequence. The requested rank is capped by the block-Hankel
evidence and the final least-squares pass prunes negligible modes without
restarting the rank-estimation stage.
"""
function _minipole_exponential_fit(samples::Vector{Matrix{ComplexF64}},
                                   kernel::MiniPoleKernel)
    sample_count = length(samples)
    training_count = sample_count - kernel.holdout_count
    training_count >= 4 || throw(ArgumentError(
        "MiniPole holdout_count leaves fewer than four training samples",
    ))
    training_samples = samples[1:training_count]
    if all(sample -> iszero(norm(sample)), training_samples)
        empty_nodes = ComplexF64[]
        empty_weights = Matrix{ComplexF64}[]
        training_error = (; maximum=0.0, l2=0.0, relative_l2=0.0)
        holdout_error = kernel.holdout_count == 0 ? training_error :
            _minipole_sequence_error(
                samples[(training_count + 1):end], empty_nodes, empty_weights;
                start_index=training_count + 1,
            )
        iszero(holdout_error.maximum) || throw(ArgumentError(
            "MiniPole zero training sequence has nonzero held-out samples and cannot validate an empty fit",
        ))
        attempt = (; rank=0, status=:zero_sequence,
                   nodes_before_pruning=empty_nodes, retained_indices=Int[],
                   discarded_indices=Int[], weight_norms=Float64[],
                   pruning_threshold=0.0, nodes=empty_nodes,
                   weights=empty_weights,
                   least_squares=(; rank=0, condition=1.0,
                                  singular_values=Float64[]),
                   training_error, holdout_error, control_limit=0.0)
        return (; nodes=ComplexF64[], weights=Matrix{ComplexF64}[],
                diagnostics=(; sample_count, training_count,
                             holdout_count=kernel.holdout_count, hankel_lag=0,
                             hankel_singular_values=Float64[],
                             effective_rank=0, requested_rank=kernel.n_poles,
                             attempts=NamedTuple[attempt], selected_attempt=attempt))
    end
    hankel, lag, row_count = _minipole_hankel(training_samples)
    decomposition = svd(hankel; full=false)
    singular_values = Float64.(decomposition.S)
    effective_rank = _minipole_effective_rank(
        singular_values, kernel.rank_tolerance,
    )
    component_count = size(first(training_samples), 1)^2
    shift_rank_limit = min(lag + 1, component_count * (row_count - 1))
    initial_rank = min(kernel.n_poles, effective_rank, shift_rank_limit)
    initial_rank >= 1 || throw(ArgumentError(
        "MiniPole matrix-Hankel evidence has numerical rank zero",
    ))

    attempts = NamedTuple[]
    best = nothing
    for rank in initial_rank:-1:1
        nodes = _minipole_shift_nodes(
            decomposition.U, component_count, rank,
        )
        weights, least_squares = _minipole_weight_fit(training_samples, nodes)
        retained, discarded, weight_norms, cutoff = _minipole_prune_modes(
            nodes, weights, kernel.rank_tolerance,
        )
        isempty(retained) && continue
        pruned_nodes, _ = _minipole_reorder(nodes, weights, retained)
        pruned_weights, least_squares = _minipole_weight_fit(
            training_samples, pruned_nodes,
        )
        training_error = _minipole_sequence_error(
            training_samples, pruned_nodes, pruned_weights,
        )
        holdout_error = kernel.holdout_count == 0 ?
            (; maximum=0.0, l2=0.0, relative_l2=0.0) :
            _minipole_sequence_error(
                samples[(training_count + 1):end], pruned_nodes, pruned_weights;
                start_index=training_count + 1,
            )
        next_singular_value = rank < length(singular_values) ?
            singular_values[rank + 1] : 0.0
        control_limit = max(10 * next_singular_value,
                            64 * eps(Float64) * first(singular_values))
        controlled = training_error.maximum <= control_limit
        attempt = (; rank, status=controlled ? :controlled : :uncontrolled,
                   nodes_before_pruning=nodes,
                   retained_indices=retained, discarded_indices=discarded,
                   weight_norms, pruning_threshold=cutoff,
                   nodes=pruned_nodes, weights=pruned_weights,
                   least_squares, training_error, holdout_error,
                   control_limit)
        push!(attempts, attempt)
        if best === nothing ||
           attempt.training_error.relative_l2 < best.training_error.relative_l2
            best = attempt
        end
        controlled && break
    end
    best === nothing && throw(ArgumentError(
        "MiniPole matrix ESPRIT produced no non-negligible exponential mode",
    ))
    kernel.fit_tolerance === nothing ||
        best.training_error.relative_l2 <= kernel.fit_tolerance ||
        throw(ArgumentError(
            "MiniPole relative training error exceeds the requested fit_tolerance",
        ))
    return (; nodes=best.nodes, weights=best.weights,
            diagnostics=(; sample_count, training_count,
                         holdout_count=kernel.holdout_count, hankel_lag=lag,
                         hankel_singular_values=singular_values,
                         effective_rank, requested_rank=kernel.n_poles,
                         attempts, selected_attempt=best))
end
