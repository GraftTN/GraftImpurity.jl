function _bathhealth_band_edges(input::BathFitInput)
    frequencies = sort!(unique(abs(frequency) for frequency in input.frequencies
                               if !(input.statistics === :boson &&
                                    iszero(frequency))))
    isempty(frequencies) && return (0.0, 0.0)
    first_cut = frequencies[clamp(ceil(Int, length(frequencies) / 3),
                                  1, length(frequencies))]
    second_cut = frequencies[clamp(ceil(Int, 2length(frequencies) / 3),
                                   1, length(frequencies))]
    return (first_cut, second_cut)
end

function _bathhealth_coordinate_frequencies(input::BathFitInput)
    labels = Float64[]
    for samples in values(input.blocks)
        for (frequency, sample) in zip(input.frequencies, samples)
            input.statistics === :boson && iszero(frequency) && continue
            append!(labels, fill(abs(frequency), length(sample)))
        end
    end
    return vcat(labels, labels)
end

function _bathhealth_profile_covariance(covariance, role::Symbol,
                                        input::BathFitInput)
    full_dimension = 2length(_bathhealth_complex_vector(input))
    matrix, mode = _bathhealth_covariance(
        covariance, role, full_dimension,
    )
    matrix === nothing && return nothing, mode
    complex_mask = _bathhealth_dynamic_complex_mask(input)
    retained = findall(vcat(complex_mask, complex_mask))
    return matrix[retained, retained], mode
end

function _bathhealth_band_q(residual, target, covariance, indices)
    isempty(indices) && return missing
    if covariance === nothing
        denominator = norm(target[indices])
        return iszero(denominator) ?
               (iszero(norm(residual[indices])) ? 0.0 : Inf) :
               norm(residual[indices]) / denominator
    end
    metric = _bathhealth_metric(
        covariance[indices, indices], :training, length(indices),
    )
    return _bathhealth_score(residual[indices], target[indices], metric)
end

function _bathhealth_diagonal_profiles(target::BathFitInput,
                                       prediction::BathFitInput,
                                       covariance)
    records = NamedTuple[]
    complex_offset = 0
    complex_total = length(_bathhealth_complex_vector(target))
    dynamic_complex = _bathhealth_dynamic_complex_mask(target)
    dynamic_full = vcat(dynamic_complex, dynamic_complex)
    retained_full = findall(dynamic_full)
    dynamic_position = Dict(
        full_index => position
        for (position, full_index) in enumerate(retained_full)
    )
    masses = Float64[]
    channel_data = NamedTuple[]
    for block in keys(target.blocks)
        expected_samples = getproperty(target.blocks, block)
        actual_samples = getproperty(prediction.blocks, block)
        dimension = size(first(expected_samples), 1)
        for channel in 1:dimension
            expected = ComplexF64[]
            actual = ComplexF64[]
            complex_indices = Int[]
            sample_offset = complex_offset
            for (frequency, expected_sample, actual_sample) in zip(
                target.frequencies, expected_samples, actual_samples,
            )
                if !(target.statistics === :boson && iszero(frequency))
                    push!(expected, expected_sample[channel, channel])
                    push!(actual, actual_sample[channel, channel])
                    push!(complex_indices,
                          sample_offset + channel + (channel - 1) * dimension)
                end
                sample_offset += length(expected_sample)
            end
            mass = sum(abs2, expected)
            push!(masses, mass)
            push!(channel_data, (; block, channel, expected, actual,
                                  complex_indices))
        end
        complex_offset += sum(length, expected_samples)
    end
    total_mass = sum(masses)
    threshold = sqrt(eps(Float64)) * total_mass
    has_target_mass = !iszero(total_mass)
    for (data, mass) in zip(channel_data, masses)
        residual = data.expected - data.actual
        embedded_residual = vcat(real.(residual), imag.(residual))
        embedded_target = vcat(real.(data.expected), imag.(data.expected))
        full_indices = vcat(data.complex_indices,
                            complex_total .+ data.complex_indices)
        indices = Int[dynamic_position[index] for index in full_indices]
        q = if covariance === nothing
            denominator = norm(embedded_target)
            iszero(denominator) ?
                (iszero(norm(embedded_residual)) ? 0.0 : Inf) :
                norm(embedded_residual) / denominator
        else
            variances = max.(diag(covariance)[indices], eps(Float64))
            norm(embedded_residual ./ sqrt.(variances)) /
                sqrt(length(indices))
        end
        push!(records, (;
            block=data.block, channel=data.channel, q,
            target_mass=mass,
            active=has_target_mass && mass >= threshold,
        ))
    end
    active = filter(record -> record.active, records)
    active_mass = sum(record.target_mass for record in active; init=0.0)
    aggregate = iszero(active_mass) ? missing :
        sum(record.target_mass * record.q for record in active) / active_mass
    return Tuple(records), aggregate
end

function _bathhealth_residual_profile(
    target::BathFitInput, prediction::BathFitInput, covariance, role::Symbol;
    band_edges::Tuple{Float64,Float64}=_bathhealth_band_edges(target),
)
    residual = _bathhealth_residual_vector(target, prediction)
    embedded_target = _bathhealth_real_vector(target)
    matrix, mode = _bathhealth_profile_covariance(
        covariance, role, target,
    )
    if matrix === nothing
        scale = max(norm(embedded_target) / sqrt(max(length(embedded_target), 1)),
                    eps(Float64))
        standardized = abs.(residual) ./ scale
    else
        standardized = abs.(residual) ./
                       sqrt.(max.(diag(matrix), eps(Float64)))
    end
    labels = _bathhealth_coordinate_frequencies(target)
    frequency_domain = target.domain in (:matsubara, :real_axis)
    low = frequency_domain ? findall(<=(band_edges[1]), labels) : Int[]
    mid = frequency_domain ?
          findall(value -> band_edges[1] < value <= band_edges[2], labels) :
          Int[]
    high = frequency_domain ? findall(>(band_edges[2]), labels) : Int[]
    channels, aggregate = _bathhealth_diagonal_profiles(
        target, prediction, matrix,
    )
    return (;
        calibrated=matrix !== nothing,
        covariance_mode=mode,
        max_standardized=isempty(standardized) ? 0.0 : maximum(standardized),
        bands=(;
            low=frequency_domain ?
                _bathhealth_band_q(residual, embedded_target, matrix, low) :
                missing,
            mid=frequency_domain ?
                _bathhealth_band_q(residual, embedded_target, matrix, mid) :
                missing,
            high=frequency_domain ?
                 _bathhealth_band_q(residual, embedded_target, matrix, high) :
                 missing,
        ),
        band_edges=frequency_domain ? band_edges : missing,
        diagonal_channels=channels,
        diagonal_aggregate=aggregate,
    )
end
