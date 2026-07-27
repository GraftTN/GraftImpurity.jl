"""
Internal deterministic generator used by opt-in bath-fit diagnostics.

The implementation is deliberately local rather than depending on `Random`:
diagnostics are package functionality, while `Random` is only a test
dependency. The stream is SplitMix64 and is therefore independent of Julia's
default RNG implementation.
"""
mutable struct _BathFitDiagnosticRNG
    state::UInt64
end

_BathFitDiagnosticRNG(seed::Unsigned) =
    _BathFitDiagnosticRNG(UInt64(seed))

_BathFitDiagnosticRNG(seed::Signed) =
    _BathFitDiagnosticRNG(reinterpret(UInt64, Int64(seed)))

function _bathfit_next_uint64!(rng::_BathFitDiagnosticRNG)
    rng.state += 0x9e3779b97f4a7c15
    value = rng.state
    value = (value ⊻ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ⊻ (value >> 27)) * 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

function _bathfit_uniform!(rng::_BathFitDiagnosticRNG)
    bits = _bathfit_next_uint64!(rng) >> 11
    return Float64(bits) * 0x1.0p-53
end

function _bathfit_standard_normals!(output::Vector{Float64},
                                    rng::_BathFitDiagnosticRNG)
    index = firstindex(output)
    while index <= lastindex(output)
        first_uniform = _bathfit_uniform!(rng)
        second_uniform = _bathfit_uniform!(rng)
        radius = sqrt(-2 * log(max(first_uniform, floatmin(Float64))))
        angle = 2pi * second_uniform
        output[index] = radius * cos(angle)
        if index < lastindex(output)
            output[index + 1] = radius * sin(angle)
        end
        index += 2
    end
    return output
end

"""
Flatten every complex entry of every named block and sample in the declared
input order. Real parts precede imaginary parts, so a covariance matrix covers
all diagonal and off-diagonal matrix entries without scalarization.
"""
function _bathfit_complex_sample_vector(input::BathFitInput)
    sample_count = sum(length(sample) for samples in values(input.blocks)
                       for sample in samples)
    flattened = Vector{ComplexF64}(undef, sample_count)
    offset = 0
    for samples in values(input.blocks), sample in samples
        entries = vec(sample)
        copyto!(flattened, offset + 1, entries, firstindex(entries),
                length(entries))
        offset += length(entries)
    end
    return flattened
end

function _bathfit_realimag_sample_vector(input::BathFitInput)
    values = _bathfit_complex_sample_vector(input)
    return vcat(real.(values), imag.(values))
end

function _bathfit_input_with_realimag(input::BathFitInput,
                                     data::AbstractVector{<:Real})
    complex_count = sum(length(sample) for samples in Base.values(input.blocks)
                        for sample in samples)
    length(data) == 2complex_count || throw(DimensionMismatch(
        "real/imag bath-fit data vector has the wrong length",
    ))
    complex_values = ComplexF64[
        complex(data[index], data[complex_count + index])
        for index in 1:complex_count
    ]
    offset = 0
    names = Tuple(keys(input.blocks))
    samples = Tuple(begin
        original_samples = getproperty(input.blocks, name)
        reconstructed = Matrix{ComplexF64}[]
        for original in original_samples
            count = length(original)
            matrix = reshape(
                copy(@view complex_values[(offset + 1):(offset + count)]),
                size(original),
            )
            offset += count
            push!(reconstructed, matrix)
        end
        reconstructed
    end for name in names)
    blocks = NamedTuple{names}(samples)
    return BathFitInput(
        input.layout, input.domain, input.statistics, copy(input.frequencies),
        blocks, input.target_labels, input.metadata, nothing,
        Val(:validated),
    )
end

function _bathfit_covariance_factor(covariance::AbstractMatrix{<:Real},
                                    dimension::Int)
    size(covariance) == (dimension, dimension) || throw(DimensionMismatch(
        "bath-fit perturbation covariance must match the full real/imag data vector",
    ))
    matrix = Matrix{Float64}(covariance)
    all(isfinite, matrix) ||
        throw(ArgumentError("bath-fit perturbation covariance must be finite"))
    symmetry_tolerance = sqrt(eps(Float64)) * max(opnorm(matrix), 1.0)
    norm(matrix - transpose(matrix)) <= symmetry_tolerance ||
        throw(ArgumentError(
            "bath-fit perturbation covariance must be real symmetric",
        ))
    decomposition = eigen(Hermitian((matrix + transpose(matrix)) / 2))
    minimum_value = minimum(decomposition.values; init=0.0)
    minimum_value >= -symmetry_tolerance || throw(ArgumentError(
        "bath-fit perturbation covariance must be positive semidefinite",
    ))
    scales = sqrt.(max.(decomposition.values, 0.0))
    return decomposition.vectors * Diagonal(scales)
end

function _bathfit_covariance_replica(input::BathFitInput,
                                     factor::AbstractMatrix{<:Real},
                                     rng::_BathFitDiagnosticRNG)
    center = _bathfit_realimag_sample_vector(input)
    size(factor) == (length(center), length(center)) ||
        throw(DimensionMismatch(
            "bath-fit covariance factor does not match the input data",
        ))
    standard = Vector{Float64}(undef, length(center))
    _bathfit_standard_normals!(standard, rng)
    return _bathfit_input_with_realimag(input, center + factor * standard)
end

function _bathfit_validate_replica(input::BathFitInput,
                                   replica::BathFitInput,
                                   partition::Partition)
    _validate_fit_input(replica, partition)
    replica.layout == input.layout || throw(ArgumentError(
        "an empirical bath-fit replica must use the source FlavorLayout",
    ))
    replica.domain === input.domain || throw(ArgumentError(
        "an empirical bath-fit replica must use the source domain",
    ))
    replica.statistics === input.statistics || throw(ArgumentError(
        "an empirical bath-fit replica must use the source statistics",
    ))
    replica.frequencies == input.frequencies || throw(ArgumentError(
        "an empirical bath-fit replica must use the source sampling grid",
    ))
    Tuple(keys(replica.blocks)) == Tuple(keys(input.blocks)) ||
        throw(ArgumentError(
            "an empirical bath-fit replica must preserve named block order",
        ))
    return replica
end

function _bathfit_diagnostic_replicas(
    input::BathFitInput, partition::Partition,
    ::Nothing, ::BathFitDiagnosticConfig,
)
    _validate_fit_input(input, partition)
    return BathFitInput[input], nothing
end

function _bathfit_diagnostic_replicas(
    input::BathFitInput, partition::Partition,
    perturbation::CovariancePerturbation,
    config::BathFitDiagnosticConfig,
)
    _validate_fit_input(input, partition)
    covariance = perturbation.covariance
    dimension = length(_bathfit_realimag_sample_vector(input))
    factor = perturbation.scale .* _bathfit_covariance_factor(
        covariance, dimension,
    )
    rng = _BathFitDiagnosticRNG(config.seed)
    replicas = BathFitInput[
        _bathfit_covariance_replica(input, factor, rng)
        for _ in 1:config.n_replicas
    ]
    calibrated_covariance = perturbation.scale^2 .* covariance
    return replicas, calibrated_covariance
end

function _bathfit_diagnostic_replicas(
    input::BathFitInput, partition::Partition,
    perturbation::EmpiricalReplicaPerturbation,
    ::BathFitDiagnosticConfig,
)
    _validate_fit_input(input, partition)
    replicas = BathFitInput[]
    for replica in perturbation.replicas
        validated = _bathfit_validate_replica(input, replica, partition)
        push!(replicas, _bathfit_input_with_realimag(
            input, _bathfit_realimag_sample_vector(validated),
        ))
    end
    if length(replicas) == 1
        return replicas, nothing
    end
    embedded = [_bathfit_realimag_sample_vector(replica) for replica in replicas]
    dimension = length(first(embedded))
    center = zeros(Float64, dimension)
    for sample in embedded
        center .+= sample
    end
    center ./= length(embedded)
    covariance = zeros(Float64, dimension, dimension)
    for sample in embedded
        difference = sample - center
        covariance .+= difference * transpose(difference)
    end
    covariance ./= length(embedded) - 1
    shrinkage = clamp(
        dimension / (dimension + length(embedded) - 1), 0.05, 1.0,
    )
    diagonal_target = Matrix(Diagonal(diag(covariance)))
    regularized = (1 - shrinkage) .* covariance .+
                  shrinkage .* diagonal_target
    return replicas, (training=regularized, validation=regularized,
                      shrinkage)
end
