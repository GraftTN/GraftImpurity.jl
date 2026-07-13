function _validate_matsubara_fit(input::BathFitInput, partition::Partition)
    _validate_fit_input(input, partition)
    input.domain === :matsubara ||
        throw(ArgumentError("PESKernel requires BathFitInput domain=:matsubara"))
    return nothing
end

function _pes_values(samples::Vector{Matrix{ComplexF64}})
    size(first(samples)) == (1, 1) &&
        return ComplexF64[sample[1, 1] for sample in samples]
    return samples
end

"""
    real_pole_bath_fit(input, kernel::PESKernel, partition)

Run the existing independent PES/AAA per-block Matsubara algorithm and adapt
its raw pole matrices into the shared `PoleExpansion`/realization boundary.
PES keeps its own candidate-pole/residue optimization; canonical ownership and
ordered factorization remain exclusively the responsibility of `realize_bath`.
"""
function real_pole_bath_fit(input::BathFitInput, kernel::PESKernel,
                            partition::Partition)
    _validate_matsubara_fit(input, partition)
    frequencies = im .* input.frequencies
    poles = Float64[]
    residues = Matrix{ComplexF64}[]
    block_indices = Int[]
    fits = NamedTuple[]
    for (block_index_value, block) in enumerate(block_names(partition))
        fit = pes_fit(
            _pes_values(_fit_block_samples(input, block)), frequencies;
            tolerance=kernel.tolerance, n_poles=kernel.n_poles,
            statistics=input.statistics, solver=kernel.solver,
            maxiter=kernel.maxiter, min_support=kernel.min_support,
            max_support=kernel.max_support, aaa_tolerance=kernel.aaa_tolerance,
            residue_tolerance=kernel.residue_tolerance,
            conic_diagnostic=kernel.conic_diagnostic,
        )
        append!(poles, fit.poles)
        append!(residues, fit.weights)
        append!(block_indices, fill(block_index_value, length(fit)))
        push!(fits, (; block, diagnostics=fit.diagnostics,
                      residue_constraint=fit.residue_constraint,
                      pole_count=length(fit)))
    end
    raw = BlockRealPoles(input.layout, partition, poles, residues, block_indices;
                         statistics=input.statistics)
    return PoleExpansion(
        raw;
        kernel=:pes,
        trace=(; plan=DiscretizationPlan(partition), fits,
               algorithm=:pes_aaa, solver=kernel.solver,
               conic_diagnostic=kernel.conic_diagnostic,
               source_metadata=input.metadata),
    )
end
