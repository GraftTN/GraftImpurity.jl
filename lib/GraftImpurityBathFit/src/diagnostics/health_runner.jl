function _bathfit_expansion_block_value(expansion::PoleExpansion,
                                        block_index_value::Int,
                                        z::ComplexF64)
    block = block_names(expansion.poles.partition)[block_index_value]
    dimension = length(block_flavors(expansion.poles.partition, block))
    value = zeros(ComplexF64, dimension, dimension)
    for pole_index in eachindex(expansion.poles.poles)
        expansion.poles.block_indices[pole_index] == block_index_value ||
            continue
        residue = expansion.poles.residues[pole_index]
        matrix = residue isa Number ?
                 reshape(ComplexF64[residue], 1, 1) :
                 residue
        value .+= _bathfit_resolvent(
            z, expansion.poles.poles[pole_index],
            expansion.poles.statistics,
        ) .* matrix
    end
    return value
end

function _bathfit_expansion_imtime_value(expansion::PoleExpansion,
                                         block_index_value::Int,
                                         tau::Float64,
                                         beta::Float64)
    expansion.poles.statistics === :fermion || throw(ArgumentError(
        "imaginary-time bath diagnostics currently support only fermions",
    ))
    block = block_names(expansion.poles.partition)[block_index_value]
    dimension = length(block_flavors(expansion.poles.partition, block))
    value = zeros(ComplexF64, dimension, dimension)
    for pole_index in eachindex(expansion.poles.poles)
        expansion.poles.block_indices[pole_index] == block_index_value ||
            continue
        residue = expansion.poles.residues[pole_index]
        matrix = residue isa Number ?
                 reshape(ComplexF64[residue], 1, 1) :
                 residue
        factor = _bathfit_imaginary_time_factor(
            tau, beta, expansion.poles.poles[pole_index],
        )
        value .-= factor .* matrix
    end
    return value
end

function _bathfit_expansion_prediction(expansion::PoleExpansion,
                                       input::BathFitInput)
    expansion.poles.layout == input.layout || throw(ArgumentError(
        "diagnostic prediction layout does not match the pole expansion",
    ))
    expansion.poles.statistics === input.statistics || throw(ArgumentError(
        "diagnostic prediction statistics do not match the pole expansion",
    ))
    _validate_fit_input(input, expansion.poles.partition)
    component = _bathfit_component(input)
    broadening = if input.domain === :real_axis
        candidate = _bathfit_trace_broadening(expansion)
        _reconstruction_broadening(input, candidate)
    else
        nothing
    end
    beta = input.domain === :imaginary_time ?
           _bathfit_imaginary_time_beta(input.frequencies) : nothing
    names = block_names(expansion.poles.partition)
    samples = Tuple(begin
        block_index_value = block_index(expansion.poles.partition, block)
        predictions = Matrix{ComplexF64}[]
        for frequency in input.frequencies
            if input.domain === :imaginary_time
                prediction = _bathfit_expansion_imtime_value(
                    expansion, block_index_value, frequency,
                    something(beta),
                )
            else
                z = component === :matsubara ?
                    im * frequency : frequency + im * something(broadening)
                retarded = _bathfit_expansion_block_value(
                    expansion, block_index_value, ComplexF64(z),
                )
                prediction = component === :spectral ?
                             (adjoint(retarded) - retarded) / (2pi * im) :
                             retarded
            end
            push!(predictions, prediction)
        end
        predictions
    end for block in names)
    blocks = NamedTuple{names}(samples)
    return BathFitInput(
        input.layout, input.domain, input.statistics, copy(input.frequencies),
        blocks, input.target_labels, input.metadata,
        _reconstructed_template(input, blocks), Val(:validated),
    )
end

function _bathfit_returned_poles(expansion::PoleExpansion)
    return length(expansion.poles)
end

function _bathfit_realized_modes(result)
    return result isa DiscretizationResult ? length(result.bath) : 0
end

function _bathfit_diagnostic_orders(orders)
    requested = Int[order for order in orders]
    isempty(requested) &&
        throw(ArgumentError("bath-fit diagnostics need at least one order"))
    all(>(0), requested) ||
        throw(ArgumentError("bath-fit diagnostic orders must be positive"))
    allunique(requested) ||
        throw(ArgumentError("bath-fit diagnostic orders must be unique"))
    return requested
end

function _bathfit_failure_candidate(order::Int, replica::Int, start::Int,
                                    error)
    return BathFitHealthCandidate(
        order, replica, start, :failed, nothing, 0, 0, false,
        nothing, nothing, nothing, sprint(showerror, error),
    )
end

function _bathfit_trace_nonconverged(value, depth::Int=0,
                                     inside_diagnostics::Bool=false)
    depth >= 12 && return false
    if value isa NamedTuple
        hasproperty(value, :status) &&
            getproperty(value, :status) === :nonconverged && return true
        inside_diagnostics && hasproperty(value, :converged) &&
            getproperty(value, :converged) === false && return true
        for name in keys(value)
            child_is_diagnostics = inside_diagnostics || name === :diagnostics
            _bathfit_trace_nonconverged(
                getproperty(value, name), depth + 1, child_is_diagnostics,
            ) && return true
        end
    elseif value isa AbstractVector
        for child in value
            _bathfit_trace_nonconverged(
                child, depth + 1, inside_diagnostics,
            ) && return true
        end
    end
    return false
end

function _bathfit_run_candidate(
    input::BathFitInput, replica_input::BathFitInput,
    validation_input::BathFitInput, partition::Partition,
    order::Int, replica::Int, start::Int, kernel_factory,
)
    try
        kernel = kernel_factory(order, replica, start)
        kernel isa AbstractRealPoleBathFitKernel || throw(ArgumentError(
            "kernel_factory must return an AbstractRealPoleBathFitKernel",
        ))
        expansion = real_pole_bath_fit(replica_input, kernel, partition)
        result = realize_bath(replica_input, expansion, partition)
        training_prediction = _bathfit_expansion_prediction(
            expansion, replica_input,
        )
        validation_prediction = _bathfit_expansion_prediction(
            expansion, validation_input,
        )
        nonconverged = _bathfit_trace_nonconverged(expansion.trace)
        status = nonconverged ? :nonconverged : :success
        message = nonconverged ? "fit trace reports nonconvergence" : ""
        candidate = BathFitHealthCandidate(
            order, replica, start, status, expansion,
            _bathfit_returned_poles(expansion),
            _bathfit_realized_modes(result),
            result isa DiscretizationResult,
            replica_input, training_prediction, validation_prediction, message,
        )
        return candidate
    catch error
        error isa InterruptException && rethrow()
        return _bathfit_failure_candidate(order, replica, start, error)
    end
end

"""
    run_bathfit_diagnostics(input, partition, orders, kernel_factory;
                            validation_input, perturbation=nothing,
                            config=BathFitDiagnosticConfig())

Explicitly run the perturb-and-refit bath-health workflow. This function is
never called by ordinary fitting or realization.

`kernel_factory(order, replica_id, start_id)` supplies one independent kernel
configuration for each attempt. A generated or empirical replica is reused
across all requested orders, while `validation_input` remains fixed. Individual
factory, fit, rank, realization, and prediction failures become structured
failed candidates and do not stop other attempts.
"""
function run_bathfit_diagnostics(
    input::BathFitInput, partition::Partition, orders,
    kernel_factory::F;
    validation_input::BathFitInput,
    perturbation::Union{Nothing,AbstractBathFitPerturbation}=nothing,
    config::BathFitDiagnosticConfig=BathFitDiagnosticConfig(),
) where {F}
    requested = _bathfit_diagnostic_orders(orders)
    _validate_fit_input(input, partition)
    _validate_fit_input(validation_input, partition)
    validation_input.layout == input.layout || throw(ArgumentError(
        "bath-fit diagnostic validation input must use the training FlavorLayout",
    ))
    validation_input.statistics === input.statistics || throw(ArgumentError(
        "bath-fit diagnostic validation input must use the training statistics",
    ))
    replicas, covariance = _bathfit_diagnostic_replicas(
        input, partition, perturbation, config,
    )
    candidates = BathFitHealthCandidate[]
    for (replica_id, replica_input) in enumerate(replicas)
        for order in requested
            append!(candidates, [
                _bathfit_run_candidate(
                    input, replica_input, validation_input, partition,
                    order, replica_id, start_id, kernel_factory,
                )
                for start_id in 1:config.starts
            ])
        end
    end
    return analyze_bathfit(
        candidates, input, validation_input; covariance, config,
    )
end
