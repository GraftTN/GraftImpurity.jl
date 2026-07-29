function _validate_minipole_bcf_exponents(
    exponents::Vector{ComplexF64},
    damping_tolerance::Float64,
)
    for exponent in exponents
        real(exponent) >= -damping_tolerance || throw(ArgumentError(
            "MiniPole BCF fit contains an unstable negative-damping exponent",
        ))
    end
    return exponents
end

function _minipole_bcf_order(exponents::Vector{ComplexF64})
    return sortperm(exponents; by=value -> (real(value), imag(value)))
end

"""
    fit_complex_bcf(input, kernel::MiniPoleKernel, partition) -> ComplexPoles

Fit each named BCF block with Graft's ESPRIT and descending-rank search. The
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
        fit = _fit_minipole_exponential_sequence(
            input.times, _bcf_block_samples(input, block), kernel,
        )
        if fit.outcome isa ZeroFit
            push!(fits, (; block, timestep, alias_limit,
                          damping_tolerance, alias_tolerance,
                          alias_warning_fraction, alias_warning=false,
                          exponents=ComplexF64[], engine=fit))
            continue
        end
        fit.outcome isa IdentifiedFit ||
            error("successful MiniPole core fit has an unknown outcome")
        value = fit.outcome.value
        block_exponents = _validate_minipole_bcf_exponents(
            ComplexF64.(value.poles), damping_tolerance,
        )
        order = _minipole_bcf_order(block_exponents)
        ordered_exponents = block_exponents[order]
        ordered_weights = value.weights[order]
        append!(exponents, ordered_exponents)
        append!(weights, ordered_weights)
        append!(block_indices, fill(block_index_value, length(ordered_exponents)))
        push!(fits, (; block, timestep, alias_limit,
                      damping_tolerance, alias_tolerance, alias_warning_fraction,
                      alias_warning=any(abs(imag(exponent)) >=
                                        alias_warning_fraction * alias_limit - alias_tolerance
                                        for exponent in ordered_exponents),
                      exponents=ordered_exponents,
                      engine=fit))
    end
    return ComplexPoles(
        input.layout, partition, exponents, weights, block_indices;
        channel=input.channel,
        diagnostics=(; kernel=:minipole, algorithm=:matrix_esprit,
                     fit_tolerance=kernel.fit_tolerance,
                     source_metadata=input.metadata, fits),
        stability_tolerance=damping_tolerance,
    )
end
