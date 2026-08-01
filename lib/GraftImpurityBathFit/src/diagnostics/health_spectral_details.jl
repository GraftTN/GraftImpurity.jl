function _bathhealth_energy_window(energy_window)
    energy_window === nothing && return nothing
    energy_window isa Union{Tuple,AbstractVector} &&
        length(energy_window) == 2 || throw(ArgumentError(
        "bath-fit spectral diagnostics need two energy-window endpoints",
    ))
    window = (Float64(first(energy_window)), Float64(last(energy_window)))
    all(isfinite, window) && window[1] < window[2] || throw(ArgumentError(
        "bath-fit spectral diagnostics need a finite increasing energy window",
    ))
    return window
end

function _bathhealth_residue_rank(matrix::Matrix{ComplexF64})
    singular_values = Float64.(LinearAlgebra.svdvals(matrix))
    isempty(singular_values) && return 0, 0.0
    threshold = max(size(matrix)...) * eps(Float64) *
                maximum(singular_values; init=0.0)
    numerical = count(>(threshold), singular_values)
    mass = sum(singular_values)
    iszero(mass) && return numerical, 0.0
    probabilities = singular_values ./ mass
    entropy = -sum(value -> iszero(value) ? 0.0 : value * log(value),
                   probabilities)
    return numerical, exp(entropy)
end

function _bathhealth_residue_admissible(matrix::Matrix{ComplexF64})
    tolerance = sqrt(eps(Float64)) * max(opnorm(matrix), 1.0)
    norm(matrix - adjoint(matrix)) <= tolerance || return false
    hermitian = Hermitian((matrix + adjoint(matrix)) / 2)
    return minimum(real.(eigvals(hermitian)); init=0.0) >= -tolerance
end

function _bathhealth_total_residue_matrices(expansion::PoleExpansion)
    names = Tuple(block_names(expansion.poles.partition))
    totals = Tuple(begin
        dimension = length(block_flavors(expansion.poles.partition, block))
        total = zeros(ComplexF64, dimension, dimension)
        block_value = block_index(expansion.poles.partition, block)
        for index in eachindex(expansion.poles.poles)
            expansion.poles.block_indices[index] == block_value || continue
            total .+= _bathhealth_residue_matrix(
                expansion.poles.residues[index],
            )
        end
        total
    end for block in names)
    return NamedTuple{names}(totals)
end

function _bathhealth_minimum_pole_spacing(expansion::PoleExpansion)
    minimum_spacing = Inf
    for block_value in eachindex(block_names(expansion.poles.partition))
        indices = findall(==(block_value), expansion.poles.block_indices)
        length(indices) <= 1 && continue
        energies = sort(expansion.poles.poles[indices])
        minimum_spacing = min(minimum_spacing, minimum(diff(energies)))
    end
    return isfinite(minimum_spacing) ? minimum_spacing : nothing
end

function _bathhealth_outside_mass(expansion::PoleExpansion,
                                  window::Union{Nothing,Tuple{Float64,Float64}},
                                  total::Float64)
    window === nothing &&
        return (; below=missing, above=missing, total=missing)
    below = 0.0
    above = 0.0
    for (energy, residue) in zip(
        expansion.poles.poles, expansion.poles.residues,
    )
        mass = real(tr(_bathhealth_residue_matrix(residue)))
        energy < window[1] && (below += mass)
        energy > window[2] && (above += mass)
    end
    if iszero(total)
        return (; below=missing, above=missing, total=missing)
    end
    return (; below=below / total, above=above / total,
            total=(below + above) / total)
end

function _bathhealth_hermitian_basis(dimension::Int)
    basis = Matrix{ComplexF64}[]
    for row in 1:dimension
        diagonal = zeros(ComplexF64, dimension, dimension)
        diagonal[row, row] = 1
        push!(basis, diagonal)
    end
    scale = inv(sqrt(2.0))
    for column in 2:dimension, row in 1:(column - 1)
        symmetric = zeros(ComplexF64, dimension, dimension)
        symmetric[row, column] = scale
        symmetric[column, row] = scale
        push!(basis, symmetric)
        antisymmetric = zeros(ComplexF64, dimension, dimension)
        antisymmetric[row, column] = im * scale
        antisymmetric[column, row] = -im * scale
        push!(basis, antisymmetric)
    end
    return basis
end

function _bathhealth_design_column(validation::BathFitInput, block::Symbol,
                                   energy::Float64,
                                   residue::Matrix{ComplexF64};
                                   differentiate_energy::Bool=false)
    values = ComplexF64[]
    step = cbrt(eps(Float64)) * max(abs(energy), 1.0)
    for name in keys(validation.blocks)
        samples = getproperty(validation.blocks, name)
        for (frequency, sample) in zip(validation.frequencies, samples)
            if name === block
                kernel = if differentiate_energy
                    upper = _bathhealth_kernel_value(
                        validation, frequency, energy + step,
                    )
                    lower = _bathhealth_kernel_value(
                        validation, frequency, energy - step,
                    )
                    (upper - lower) / (2step)
                else
                    _bathhealth_kernel_value(validation, frequency, energy)
                end
                append!(values, vec(kernel .* residue))
            else
                append!(values, zeros(ComplexF64, length(sample)))
            end
        end
    end
    mask = _bathhealth_dynamic_complex_mask(validation)
    return vcat(real.(values[mask]), imag.(values[mask]))
end

function _bathhealth_design_capacity(expansion::PoleExpansion,
                                     validation::BathFitInput,
                                     metric::_BathHealthMetric)
    metric.whitening === nothing && return missing, missing
    columns = Vector{Float64}[]
    names = block_names(expansion.poles.partition)
    for index in eachindex(expansion.poles.poles)
        block = names[expansion.poles.block_indices[index]]
        dimension = length(block_flavors(expansion.poles.partition, block))
        residue = _bathhealth_residue_matrix(
            expansion.poles.residues[index],
        )
        energy_column = _bathhealth_design_column(
            validation, block, expansion.poles.poles[index], residue;
            differentiate_energy=true,
        )
        push!(columns, metric.whitening * energy_column)
        for basis in _bathhealth_hermitian_basis(dimension)
            column = _bathhealth_design_column(
                validation, block, expansion.poles.poles[index], basis,
            )
            push!(columns, metric.whitening * column)
        end
    end
    isempty(columns) && return 0, 0.0
    design = hcat(columns...)
    singular_values = Float64.(LinearAlgebra.svdvals(design))
    threshold = max(size(design)...) * eps(Float64) *
                maximum(singular_values; init=0.0)
    design_rank = count(>(threshold), singular_values)
    kernel_noise_rank = design_rank
    capacity_ratio = size(design, 2) / max(design_rank, 1)
    return kernel_noise_rank, capacity_ratio
end

function _bathhealth_spectral_details(
    expansion::PoleExpansion, energy_window, validation::BathFitInput,
    metric::_BathHealthMetric,
)
    window = _bathhealth_energy_window(energy_window)
    expansion.poles.layout == validation.layout &&
        expansion.poles.statistics === validation.statistics ||
        throw(ArgumentError(
            "bath-fit spectral diagnostics require matching layout and statistics",
        ))
    _validate_fit_input(validation, expansion.poles.partition)
    matrices = Matrix{ComplexF64}[
        _bathhealth_residue_matrix(residue)
        for residue in expansion.poles.residues
    ]
    trace_masses = Float64[real(tr(matrix)) for matrix in matrices]
    total_trace_mass = sum(trace_masses)
    ranks = [_bathhealth_residue_rank(matrix) for matrix in matrices]
    admissible = all(_bathhealth_residue_admissible, matrices)
    weak_threshold = sqrt(eps(Float64)) * total_trace_mass
    weak_mass = sum(
        filter(mass -> mass < weak_threshold, trace_masses);
        init=0.0,
    )
    weak_fraction = iszero(total_trace_mass) ? missing :
                    weak_mass / total_trace_mass
    fisher_resolution = if metric.whitening === nothing
        missing
    else
        names = block_names(expansion.poles.partition)
        Float64[
            _bathhealth_fisher_resolution(
                validation, names[expansion.poles.block_indices[index]],
                expansion.poles.poles[index], matrices[index], metric,
            )
            for index in eachindex(matrices)
        ]
    end
    kernel_noise_rank, capacity_ratio = _bathhealth_design_capacity(
        expansion, validation, metric,
    )
    return (;
        total_trace_mass,
        total_residue_matrices=_bathhealth_total_residue_matrices(expansion),
        admissible,
        residue_ranks=Int[first(rank) for rank in ranks],
        effective_ranks=Float64[last(rank) for rank in ranks],
        weak_residue_fraction=weak_fraction,
        minimum_pole_spacing=_bathhealth_minimum_pole_spacing(expansion),
        outside_mass=_bathhealth_outside_mass(
            expansion, window, total_trace_mass,
        ),
        fisher_resolution,
        kernel_noise_rank,
        capacity_ratio,
    )
end

function _bathhealth_boundary_flux(reference::PoleExpansion,
                                   current::PoleExpansion, energy_window)
    window = _bathhealth_energy_window(energy_window)
    window === nothing && return (;
        below=missing, above=missing, total=missing,
        fractions=(; below=missing, above=missing, total=missing),
    )
    reference.poles.layout == current.poles.layout &&
        reference.poles.partition == current.poles.partition &&
        reference.poles.statistics === current.poles.statistics ||
        throw(ArgumentError(
            "bath-fit boundary flux requires matching layout and statistics",
        ))
    function outside(expansion)
        below = 0.0
        above = 0.0
        for (energy, residue) in zip(
            expansion.poles.poles, expansion.poles.residues,
        )
            mass = real(tr(_bathhealth_residue_matrix(residue)))
            energy < window[1] && (below += mass)
            energy > window[2] && (above += mass)
        end
        return below, above
    end
    reference_below, reference_above = outside(reference)
    current_below, current_above = outside(current)
    below = current_below - reference_below
    above = current_above - reference_above
    scale = sum(
        Float64[
            real(tr(_bathhealth_residue_matrix(residue)))
            for residue in reference.poles.residues
        ];
        init=0.0,
    )
    denominator = abs(scale)
    fractions = iszero(denominator) ?
        (; below=missing, above=missing, total=missing) :
        (; below=below / denominator, above=above / denominator,
         total=(abs(below) + abs(above)) / denominator)
    return (; below, above, total=abs(below) + abs(above), fractions)
end
