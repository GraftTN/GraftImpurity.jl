function _bathhealth_residue_matrix(residue)
    return residue isa Number ? reshape(ComplexF64[residue], 1, 1) :
           Matrix{ComplexF64}(residue)
end

function _bathhealth_state(residue)
    matrix = _bathhealth_residue_matrix(residue)
    tolerance = sqrt(eps(Float64)) * max(opnorm(matrix), 1.0)
    norm(matrix - adjoint(matrix)) <= tolerance || return nothing
    hermitian = Matrix{ComplexF64}(Hermitian((matrix + adjoint(matrix)) / 2))
    decomposition = eigen(Hermitian(hermitian))
    minimum(real.(decomposition.values); init=0.0) >= -tolerance ||
        return nothing
    positive = decomposition.vectors *
               Diagonal(max.(real.(decomposition.values), 0.0)) *
               adjoint(decomposition.vectors)
    weight = real(tr(positive))
    isfinite(weight) && weight >= 0 || return nothing
    return weight, positive
end

function _bathhealth_measures(expansion::PoleExpansion)
    names = Tuple(block_names(expansion.poles.partition))
    measures = Tuple(begin
        block_value = block_index(expansion.poles.partition, block)
        energies = Float64[]
        weights = Float64[]
        states = Matrix{ComplexF64}[]
        valid = true
        for pole_index in eachindex(expansion.poles.poles)
            expansion.poles.block_indices[pole_index] == block_value || continue
            resolved = _bathhealth_state(expansion.poles.residues[pole_index])
            if resolved === nothing
                valid = false
                break
            end
            weight, matrix = resolved
            iszero(weight) && continue
            push!(energies, expansion.poles.poles[pole_index])
            push!(weights, weight)
            push!(states, matrix ./ weight)
        end
        mass = sum(weights)
        if !valid || !isfinite(mass) || mass <= 0
            nothing
        else
            ordering = sortperm(energies)
            _BathHealthMeasure(
                energies[ordering], weights[ordering], states[ordering], mass,
            )
        end
    end for block in names)
    return NamedTuple{names}(measures)
end

function _bathhealth_matrix_sqrt(matrix::Matrix{ComplexF64})
    decomposition = eigen(Hermitian((matrix + adjoint(matrix)) / 2))
    values = max.(real.(decomposition.values), 0.0)
    return decomposition.vectors * Diagonal(sqrt.(values)) *
           adjoint(decomposition.vectors)
end

function _bathhealth_bures(left::Matrix{ComplexF64},
                           right::Matrix{ComplexF64})
    size(left) == size(right) || return Inf
    root = _bathhealth_matrix_sqrt(left)
    fidelity = real(tr(_bathhealth_matrix_sqrt(
        root * right * root,
    )))
    return sqrt(max(0.0, 2 - 2clamp(fidelity, 0.0, 1.0)))
end

function _bathhealth_kernel_value(input::BathFitInput, frequency::Float64,
                                  energy::Float64)
    if input.domain === :imaginary_time
        input.statistics === :fermion || return 0.0 + 0im
        beta = _bathfit_imaginary_time_beta(input.frequencies)
        return -_bathfit_imaginary_time_factor(frequency, beta, energy) + 0im
    end
    if input.domain === :matsubara
        return _bathfit_resolvent(im * frequency, energy, input.statistics)
    end
    spacing = length(input.frequencies) <= 1 ? 1.0 :
              minimum(abs, diff(sort(input.frequencies)))
    eta = hasproperty(input.metadata, :broadening) ?
          Float64(getproperty(input.metadata, :broadening)) : spacing / 2
    retarded = _bathfit_resolvent(
        frequency + im * eta, energy, input.statistics,
    )
    return _bathfit_component(input) === :spectral ?
           -imag(retarded) / pi + 0im : retarded
end

function _bathhealth_fisher_resolution(input::BathFitInput, block::Symbol,
                                       energy::Float64,
                                       state::Matrix{ComplexF64},
                                       metric::_BathHealthMetric)
    step = cbrt(eps(Float64)) * max(abs(energy), 1.0)
    names = Tuple(keys(input.blocks))
    derivative = ComplexF64[]
    for name in names
        samples = getproperty(input.blocks, name)
        for (frequency, sample) in zip(input.frequencies, samples)
            if name === block
                upper = _bathhealth_kernel_value(input, frequency, energy + step)
                lower = _bathhealth_kernel_value(input, frequency, energy - step)
                append!(derivative, vec(((upper - lower) / (2step)) .* state))
            else
                append!(derivative, zeros(ComplexF64, length(sample)))
            end
        end
    end
    embedded = vcat(real.(derivative), imag.(derivative))
    complex_mask = _bathhealth_dynamic_complex_mask(input)
    embedded = embedded[vcat(complex_mask, complex_mask)]
    information = if metric.whitening === nothing
        sum(abs2, embedded)
    else
        sum(abs2, metric.whitening * embedded)
    end
    return inv(sqrt(max(information, eps(Float64))))
end

function _bathhealth_transport(left::_BathHealthMeasure,
                               right::_BathHealthMeasure,
                               validation::BathFitInput, block::Symbol,
                               metric::_BathHealthMetric)
    left_weights = left.weights ./ left.mass
    right_weights = right.weights ./ right.mass
    scalar = all(state -> size(state) == (1, 1),
                 vcat(left.states, right.states))
    left_remaining = copy(left_weights)
    right_remaining = copy(right_weights)
    distance = 0.0
    tolerance = 64eps(Float64)
    while true
        left_indices = findall(>(tolerance), left_remaining)
        right_indices = findall(>(tolerance), right_remaining)
        (isempty(left_indices) || isempty(right_indices)) && break
        best = nothing
        best_cost = Inf
        if scalar
            left_index = first(left_indices)
            right_index = first(right_indices)
            pairs = ((left_index, right_index),)
        else
            pairs = ((left_index, right_index)
                     for left_index in left_indices
                     for right_index in right_indices)
        end
        for (left_index, right_index) in pairs
            left_energy = left.energies[left_index]
            right_energy = right.energies[right_index]
            midpoint = (left.energies[left_index] +
                        right.energies[right_index]) / 2
            average_state = (left.states[left_index] +
                             right.states[right_index]) / 2
            left_resolution = _bathhealth_fisher_resolution(
                validation, block, left_energy,
                left.states[left_index], metric,
            )
            midpoint_resolution = _bathhealth_fisher_resolution(
                validation, block, midpoint, average_state, metric,
            )
            right_resolution = _bathhealth_fisher_resolution(
                validation, block, right_energy,
                right.states[right_index], metric,
            )
            inverse_resolution = (
                inv(max(left_resolution, eps(Float64))) +
                4inv(max(midpoint_resolution, eps(Float64))) +
                inv(max(right_resolution, eps(Float64)))
            ) / 6
            energy_cost = abs(left_energy - right_energy) * inverse_resolution
            cost = energy_cost
            if !scalar
                cost += _bathhealth_bures(
                    left.states[left_index], right.states[right_index],
                )
            end
            if cost < best_cost
                best_cost = cost
                best = (left_index, right_index)
            end
        end
        best === nothing && break
        left_index, right_index = best
        transported = min(left_remaining[left_index],
                          right_remaining[right_index])
        distance += transported * best_cost
        left_remaining[left_index] -= transported
        right_remaining[right_index] -= transported
    end
    return distance
end

function _bathhealth_measure_distance(left::PoleExpansion,
                                      right::PoleExpansion,
                                      validation::BathFitInput,
                                      metric::_BathHealthMetric)
    left_measures = _bathhealth_measures(left)
    right_measures = _bathhealth_measures(right)
    Tuple(keys(left_measures)) == Tuple(keys(right_measures)) ||
        return nothing, nothing
    mass_values = Float64[]
    shape_values = Float64[]
    shape_weights = Float64[]
    for block in keys(left_measures)
        left_measure = getproperty(left_measures, block)
        right_measure = getproperty(right_measures, block)
        (left_measure === nothing || right_measure === nothing) &&
            return nothing, nothing
        scale = max(left_measure.mass, right_measure.mass)
        push!(mass_values, abs(left_measure.mass - right_measure.mass) / scale)
        push!(shape_values, _bathhealth_transport(
            left_measure, right_measure, validation, block, metric,
        ))
        push!(shape_weights, scale)
    end
    mass = maximum(mass_values; init=0.0)
    shape = sum(shape_values .* shape_weights) / sum(shape_weights)
    return mass, shape
end
