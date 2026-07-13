"""
Convert a shared matrix-ESPRIT shift node to the BCF convention
`exp(-z*t)`. The principal logarithm fixes the first implementation's branch;
the Nyquist alias boundary is retained in diagnostics rather than hidden.
"""
function _minipole_bcf_exponents(nodes::Vector{ComplexF64}, timestep::Float64,
                                 damping_tolerance::Float64)
    exponents = ComplexF64[]
    for node in nodes
        !iszero(node) || throw(ArgumentError(
            "MiniPole BCF shift node is numerically zero and has no finite exponent",
        ))
        exponent = -log(node) / timestep
        isfinite(real(exponent)) && isfinite(imag(exponent)) ||
            throw(ArgumentError("MiniPole BCF exponent is nonfinite"))
        real(exponent) >= -damping_tolerance || throw(ArgumentError(
            "MiniPole BCF fit contains an unstable negative-damping exponent",
        ))
        push!(exponents, ComplexF64(exponent))
    end
    return exponents
end

function _minipole_bcf_order(exponents::Vector{ComplexF64})
    return sortperm(exponents; by=value -> (real(value), imag(value)))
end

"""
    fit_complex_bcf(input, kernel::MiniPoleKernel, partition) -> ComplexPoles

Fit each named BCF block with the shared clean-room matrix-ESPRIT engine. The
returned value preserves complex exponents and full matrix weights under
`C(t)=sum(W_k * exp(-z_k*t))`; it is intentionally not passed through the
real-pole Hamiltonian realization gate.
"""
function fit_complex_bcf(input::BCFFitInput, kernel::MiniPoleKernel,
                         partition::Partition)
    _validate_bcf_input(input, partition)
    timestep = _bcf_timestep(input)
    damping_tolerance = sqrt(eps(Float64)) / timestep
    alias_limit = pi / timestep
    alias_tolerance = sqrt(eps(Float64)) * max(inv(timestep), alias_limit)
    alias_warning_fraction = 0.98
    exponents = ComplexF64[]
    weights = Matrix{ComplexF64}[]
    block_indices = Int[]
    fits = NamedTuple[]
    for (block_index_value, block) in enumerate(block_names(partition))
        fit = _minipole_exponential_fit(
            _bcf_block_samples(input, block), kernel,
        )
        isempty(fit.nodes) && begin
            push!(fits, (; block, timestep, alias_limit=pi / timestep,
                          alias_warning=false, exponents=ComplexF64[],
                          engine=fit.diagnostics))
            continue
        end
        block_exponents = _minipole_bcf_exponents(
            fit.nodes, timestep, damping_tolerance,
        )
        order = _minipole_bcf_order(block_exponents)
        ordered_exponents, ordered_weights = _minipole_reorder(
            block_exponents, fit.weights, order,
        )
        append!(exponents, ordered_exponents)
        append!(weights, ordered_weights)
        append!(block_indices, fill(block_index_value, length(ordered_exponents)))
        push!(fits, (; block, timestep, alias_limit,
                      damping_tolerance, alias_tolerance, alias_warning_fraction,
                      alias_warning=any(abs(imag(exponent)) >=
                                        alias_warning_fraction * alias_limit - alias_tolerance
                                        for exponent in ordered_exponents),
                      exponents=ordered_exponents,
                      engine=fit.diagnostics))
    end
    return ComplexPoles(
        input.layout, partition, exponents, weights, block_indices;
        channel=input.channel,
        diagnostics=(; kernel=:minipole, algorithm=:matrix_esprit,
                     source_metadata=input.metadata, fits),
        stability_tolerance=damping_tolerance,
    )
end
