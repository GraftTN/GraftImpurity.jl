function _fit_input_component(input::BathFitInput)
    hasproperty(input.metadata, :component) || return :spectral
    return getproperty(input.metadata, :component)
end

function _validate_real_axis_fit(input::BathFitInput,
                                 plan::DiscretizationPlan,
                                 partition::Partition)
    _validate_fit_input(input, partition)
    input.domain === :real_axis ||
        throw(ArgumentError("this kernel requires BathFitInput domain=:real_axis"))
    Tuple(keys(plan.blocks)) == block_names(partition) ||
        throw(ArgumentError("DiscretizationPlan block names must match Partition"))
    return nothing
end

function _sorted_real_axis_samples(input::BathFitInput, block::Symbol)
    permutation = sortperm(input.frequencies)
    frequencies = input.frequencies[permutation]
    samples = _fit_block_samples(input, block)[permutation]
    return frequencies, samples
end

function _sample_at(frequencies::Vector{Float64}, samples::Vector{Matrix{ComplexF64}},
                    point::Float64)
    frequencies[1] <= point <= frequencies[end] ||
        throw(ArgumentError(
            "discretization interval lies outside the supplied real-frequency mesh",
        ))
    right = searchsortedfirst(frequencies, point)
    if right == 1 || frequencies[right] == point
        return copy(samples[right])
    elseif right > length(frequencies)
        return copy(samples[end])
    end
    left = right - 1
    fraction = (point - frequencies[left]) /
               (frequencies[right] - frequencies[left])
    return (1 - fraction) .* samples[left] .+ fraction .* samples[right]
end

function _integrate_linear_matrix(frequencies::Vector{Float64},
                                  samples::Vector{Matrix{ComplexF64}},
                                  lower::Float64, upper::Float64)
    lower < upper || throw(ArgumentError("integration bounds must be ordered"))
    interior = frequencies[(lower .< frequencies) .& (frequencies .< upper)]
    nodes = vcat(lower, interior, upper)
    values = [_sample_at(frequencies, samples, point) for point in nodes]
    integral = zeros(ComplexF64, size(first(samples))...)
    for index in 1:(length(nodes) - 1)
        integral .+= (nodes[index + 1] - nodes[index]) / 2 .*
                     (values[index] .+ values[index + 1])
    end
    return integral
end

function _forced_pole_slots(poles::Vector{Float64}, forced::Tuple{Vararg{Float64}})
    count = length(forced)
    count == 0 && return Int[]
    count <= length(poles) ||
        throw(ArgumentError("forced poles exceed the available interval bins"))
    cost = fill(Inf, count + 1, length(poles) + 1)
    choose = falses(count, length(poles))
    cost[1, :] .= 0.0
    for forced_index in 1:count, pole_index in eachindex(poles)
        skipped = cost[forced_index + 1, pole_index]
        matched = cost[forced_index, pole_index] +
                  abs(forced[forced_index] - poles[pole_index])
        if matched < skipped
            cost[forced_index + 1, pole_index + 1] = matched
            choose[forced_index, pole_index] = true
        else
            cost[forced_index + 1, pole_index + 1] = skipped
        end
    end
    slots = zeros(Int, count)
    forced_index = count
    pole_index = length(poles)
    while forced_index > 0
        choose[forced_index, pole_index] || (pole_index -= 1; continue)
        slots[forced_index] = pole_index
        forced_index -= 1
        pole_index -= 1
    end
    return slots
end

function _interval_bin_grid(interval::SpectralInterval)
    edges = collect(range(interval.lower, interval.upper;
                          length=interval.modes + 1))
    poles = Float64[(edges[index] + edges[index + 1]) / 2
                    for index in 1:interval.modes]
    for (slot, forced) in zip(_forced_pole_slots(poles, interval.forced_poles),
                              interval.forced_poles)
        poles[slot] = forced
    end
    return edges, poles
end

function _quadrature_components(input::BathFitInput,
                                plan::DiscretizationPlan,
                                partition::Partition)
    poles = Float64[]
    residues = Matrix{ComplexF64}[]
    block_indices = Int[]
    bins = NamedTuple[]
    for (block_index_value, block) in enumerate(block_names(partition))
        block_plan = plan_block(plan, block)
        isempty(block_plan.intervals) &&
            throw(ArgumentError("QuadratureKernel needs support intervals for block $block"))
        dimension = length(block_flavors(partition, block))
        dimension == 1 || plan.shared_grid ||
            throw(ArgumentError(
                "matrix block $block requires DiscretizationPlan(shared_grid=true)",
            ))
        frequencies, samples = _sorted_real_axis_samples(input, block)
        for (interval_index, interval) in enumerate(block_plan.intervals)
            edges, interval_poles = _interval_bin_grid(interval)
            for bin_index in eachindex(interval_poles)
                residue = _integrate_linear_matrix(
                    frequencies, samples, edges[bin_index], edges[bin_index + 1],
                )
                push!(poles, interval_poles[bin_index])
                push!(residues, residue)
                push!(block_indices, block_index_value)
                push!(bins, (; block, interval_index, bin_index,
                              lower=edges[bin_index], upper=edges[bin_index + 1],
                              pole=interval_poles[bin_index]))
            end
        end
    end
    return (; poles, residues, block_indices, bins)
end

function _quadrature_expansion(input::BathFitInput, plan::DiscretizationPlan,
                               partition::Partition, kernel::Symbol,
                               trace::NamedTuple)
    components = _quadrature_components(input, plan, partition)
    raw = BlockRealPoles(input.layout, partition, components.poles,
                         components.residues, components.block_indices;
                         statistics=input.statistics)
    return PoleExpansion(raw; kernel,
                         trace=(; plan, bins=components.bins, trace...))
end

"""
    real_pole_bath_fit(input, kernel::QuadratureKernel, partition)

Integrate each declared real-axis spectral bin without an optimization loop.
For matrix blocks every component contributes to the same raw matrix residue;
Hermiticity/PSD is intentionally checked later by `realize_bath` rather than
by deleting or diagonalizing off-diagonal data here.
"""
function real_pole_bath_fit(input::BathFitInput, kernel::QuadratureKernel,
                            partition::Partition)
    _validate_real_axis_fit(input, kernel.plan, partition)
    _fit_input_component(input) === :spectral ||
        throw(ArgumentError("QuadratureKernel requires component=:spectral input"))
    return _quadrature_expansion(
        input, kernel.plan, partition, :quadrature,
        (; rule=kernel.rule, algorithm=:direct_bin_integration,
           source_metadata=input.metadata),
    )
end

function _spectral_from_bins(poles::AbstractVector{<:Real},
                             residues::AbstractVector{<:AbstractMatrix},
                             frequency::Real, broadening::Float64)
    dimension = size(first(residues))
    value = zeros(ComplexF64, dimension...)
    for (pole, residue) in zip(poles, residues)
        kernel = broadening / pi / ((Float64(frequency) - pole)^2 + broadening^2)
        value .+= kernel .* residue
    end
    return value
end
