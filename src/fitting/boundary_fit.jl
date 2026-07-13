function _rescaled_block_plan(block_plan::BlockDiscretizationPlan,
                              scale::Float64)
    isempty(block_plan.intervals) &&
        throw(ArgumentError("BoundaryFitKernel needs nonempty support intervals"))
    intervals = collect(block_plan.intervals)
    lower, upper = something(
        block_plan.outer_bounds,
        (first(intervals).lower, last(intervals).upper),
    )
    center = (lower + upper) / 2
    scaled_lower = center + scale * (lower - center)
    scaled_upper = center + scale * (upper - center)
    scaled_lower < scaled_upper ||
        throw(ArgumentError("boundary scale collapses a support interval"))
    replacement = SpectralInterval[]
    for (index, interval) in enumerate(intervals)
        interval_lower = index == 1 ? scaled_lower : interval.lower
        interval_upper = index == length(intervals) ? scaled_upper : interval.upper
        push!(replacement, SpectralInterval(
            interval_lower, interval_upper, interval.modes;
            forced_poles=collect(interval.forced_poles),
        ))
    end
    return BlockDiscretizationPlan(
        replacement; outer_bounds=(scaled_lower, scaled_upper),
        discarded_weight=block_plan.discarded_weight,
        weight_measure=block_plan.weight_measure,
    )
end

function _rescaled_boundary_plan(plan::DiscretizationPlan, scale::Float64)
    return DiscretizationPlan(
        (block => _rescaled_block_plan(plan_block(plan, block), scale)
         for block in keys(plan.blocks))...;
        shared_grid=plan.shared_grid,
    )
end

function _spectral_discarded_weight(input::BathFitInput, block::Symbol,
                                    block_plan::BlockDiscretizationPlan)
    frequencies, samples = _sorted_real_axis_samples(input, block)
    grid, weights, measure = _spectral_weight_profile(frequencies, samples)
    total, retained = _spectral_support_weight(
        grid, weights,
        ((interval.lower, interval.upper) for interval in block_plan.intervals),
    )
    return max(total - retained, 0.0), measure
end

function _boundary_plan_with_coverage(input::BathFitInput,
                                      plan::DiscretizationPlan,
                                      component::Symbol)
    component === :spectral || return plan
    return DiscretizationPlan(
        (block => begin
            block_plan = plan_block(plan, block)
            bounds = something(block_plan.outer_bounds,
                               (first(block_plan.intervals).lower,
                                last(block_plan.intervals).upper))
            discarded, measure = _spectral_discarded_weight(
                input, block, block_plan,
            )
            BlockDiscretizationPlan(
                collect(block_plan.intervals); outer_bounds=bounds,
                discarded_weight=discarded, weight_measure=measure,
            )
        end for block in keys(plan.blocks))...;
        shared_grid=plan.shared_grid,
    )
end

function _components_for_block(components, block_index_value::Int)
    indices = findall(==(block_index_value), components.block_indices)
    return components.poles[indices], components.residues[indices], indices
end

function _spectral_candidate_error(input::BathFitInput, partition::Partition,
                                   components, broadening::Float64)
    error = 0.0
    for (block_index_value, block) in enumerate(block_names(partition))
        poles, residues, _ = _components_for_block(components, block_index_value)
        samples = _fit_block_samples(input, block)
        for (frequency, sample) in zip(input.frequencies, samples)
            error += norm(_spectral_from_bins(
                poles, residues, frequency, broadening,
            ) - sample)^2
        end
    end
    return error
end

function _least_squares_boundary_components(input::BathFitInput,
                                            plan::DiscretizationPlan,
                                            partition::Partition,
                                            broadening::Float64)
    provisional = _quadrature_components(input, plan, partition)
    fitted = Matrix{ComplexF64}[copy(residue) for residue in provisional.residues]
    for (block_index_value, block) in enumerate(block_names(partition))
        poles, _, indices = _components_for_block(provisional, block_index_value)
        samples = _fit_block_samples(input, block)
        design = ComplexF64[inv(frequency - pole + im * broadening)
                            for frequency in input.frequencies, pole in poles]
        dimension = length(block_flavors(partition, block))
        residues = [zeros(ComplexF64, dimension, dimension) for _ in poles]
        for row in 1:dimension, column in 1:dimension
            target = ComplexF64[sample[row, column] for sample in samples]
            coefficients = design \ target
            for pole_index in eachindex(poles)
                residues[pole_index][row, column] = coefficients[pole_index]
            end
        end
        for (global_index, residue) in zip(indices, residues)
            fitted[global_index] = residue
        end
    end
    return (; poles=provisional.poles, residues=fitted,
            block_indices=provisional.block_indices, bins=provisional.bins)
end

function _retarded_candidate_error(input::BathFitInput, partition::Partition,
                                   components, broadening::Float64)
    error = 0.0
    for (block_index_value, block) in enumerate(block_names(partition))
        poles, residues, _ = _components_for_block(components, block_index_value)
        samples = _fit_block_samples(input, block)
        for (frequency, sample) in zip(input.frequencies, samples)
            model = zeros(ComplexF64, size(sample)...)
            for (pole, residue) in zip(poles, residues)
                model .+= residue ./ (frequency - pole + im * broadening)
            end
            error += norm(model - sample)^2
        end
    end
    return error
end

function _boundary_components(input::BathFitInput, kernel::BoundaryFitKernel,
                              plan::DiscretizationPlan, partition::Partition,
                              component::Symbol)
    solver = kernel.residue_solver === :auto ?
        (component === :spectral ? :bin_integral : :least_squares) :
        kernel.residue_solver
    if component === :spectral
        solver === :bin_integral ||
            throw(ArgumentError(
                "spectral BoundaryFitKernel input requires residue_solver=:bin_integral",
            ))
        components = _quadrature_components(input, plan, partition)
        return components, _spectral_candidate_error(
            input, partition, components, kernel.broadening,
        ), solver
    elseif component === :retarded
        solver === :least_squares ||
            throw(ArgumentError(
                "retarded BoundaryFitKernel input requires residue_solver=:least_squares",
            ))
        components = _least_squares_boundary_components(
            input, plan, partition, kernel.broadening,
        )
        return components, _retarded_candidate_error(
            input, partition, components, kernel.broadening,
        ), solver
    end
    throw(ArgumentError("BoundaryFitKernel needs component=:spectral or :retarded"))
end

"""
    real_pole_bath_fit(input, kernel::BoundaryFitKernel, partition)

Run a deterministic real-axis outer boundary scan. For every candidate the
kernel retains complete matrix residues on a shared grid, records its error,
and places per-bin Hermiticity/PSD/ordered-LDL evidence in the expansion trace.
Use `realize_bath` to turn a non-mountable trace into the typed
`NonMountablePoleFit` result; no diagonal-only fallback exists here.
"""
function real_pole_bath_fit(input::BathFitInput, kernel::BoundaryFitKernel,
                            partition::Partition)
    _validate_real_axis_fit(input, kernel.plan, partition)
    component = _fit_input_component(input)
    component in (:spectral, :retarded) ||
        throw(ArgumentError("BoundaryFitKernel needs component=:spectral or :retarded"))
    candidates = NamedTuple[]
    best = nothing
    for scale in kernel.scan_scales
        plan = _rescaled_boundary_plan(kernel.plan, scale)
        candidate = try
            covered_plan = _boundary_plan_with_coverage(input, plan, component)
            components, error, solver = _boundary_components(
                input, kernel, covered_plan, partition, component,
            )
            raw = BlockRealPoles(input.layout, partition, components.poles,
                                 components.residues, components.block_indices;
                                 statistics=input.statistics)
            preflight = _attempt_factorization(
                raw; orbital_order=kernel.orbital_order,
            )
            (; scale, plan=covered_plan, components, error, solver,
             diagnostics=preflight.diagnostics,
             mountable=preflight.orbitals !== nothing,
             status=:evaluated, message=nothing)
        catch exception
            exception isa ArgumentError || rethrow()
            (; scale, plan, components=nothing, error=Inf, solver=nothing,
             diagnostics=PoleBinDiagnostic[], mountable=false,
             status=:invalid, message=sprint(showerror, exception))
        end
        push!(candidates, candidate)
        candidate.status === :evaluated || continue
        if best === nothing ||
           (candidate.mountable && !best.mountable) ||
           (candidate.mountable === best.mountable && candidate.error < best.error)
            best = candidate
        end
    end
    best === nothing && throw(ArgumentError(
        "BoundaryFitKernel has no candidate inside the supplied real-frequency mesh",
    ))
    raw = BlockRealPoles(input.layout, partition, best.components.poles,
                         best.components.residues, best.components.block_indices;
                         statistics=input.statistics)
    curve = [(; scale=candidate.scale, plan=candidate.plan,
               error=candidate.error, solver=candidate.solver,
               diagnostics=candidate.diagnostics, mountable=candidate.mountable,
               status=candidate.status, message=candidate.message)
             for candidate in candidates]
    return PoleExpansion(
        raw;
        kernel=:boundary_fit,
        trace=(; plan=best.plan, component, broadening=kernel.broadening,
               residue_solver=best.solver, boundary_curve=curve,
               selected_scale=best.scale, bins=best.components.bins,
               bin_diagnostics=best.diagnostics,
               orbital_order=kernel.orbital_order,
               selection_policy=kernel.selection_policy,
               source_metadata=input.metadata),
    )
end
