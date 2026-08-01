function _bathfit_audit_item(block::Union{Nothing,Symbol}, criterion::Symbol,
                             passed::Bool, observed, threshold,
                             message::AbstractString)
    observation = observed === nothing ? nothing : Float64(observed)
    limit = threshold === nothing ? nothing : Float64(threshold)
    observation === nothing || (isfinite(observation) || observation == Inf) ||
        throw(ArgumentError("BathFitAudit observed value must be finite or Inf"))
    limit === nothing || isfinite(limit) ||
        throw(ArgumentError("BathFitAudit threshold must be finite"))
    return BathFitAuditItem(block, criterion, passed, observation, limit,
                            String(message))
end

function _bathfit_upper_check!(items::Vector{BathFitAuditItem},
                               block::Union{Nothing,Symbol}, criterion::Symbol,
                               observed, threshold, description::AbstractString)
    threshold === nothing && return nothing
    passed = observed !== nothing && observed <= threshold
    message = observed === nothing ?
              "$description is unavailable" :
              "$description = $observed must be at most $threshold"
    push!(items, _bathfit_audit_item(block, criterion, passed, observed,
                                     threshold, message))
    return nothing
end

function _bathfit_lower_check!(items::Vector{BathFitAuditItem},
                               block::Union{Nothing,Symbol}, criterion::Symbol,
                               observed, threshold, description::AbstractString)
    threshold === nothing && return nothing
    passed = observed !== nothing && observed >= threshold
    message = observed === nothing ?
              "$description is unavailable" :
              "$description = $observed must be at least $threshold"
    push!(items, _bathfit_audit_item(block, criterion, passed, observed,
                                     threshold, message))
    return nothing
end

function _bathfit_spacing_ratio(block_report::BathFitBlockReport,
                                broadening::Union{Nothing,Float64})
    broadening === nothing && return nothing
    block_report.max_spacing === nothing && return nothing
    return block_report.max_spacing / broadening
end

function _bathfit_beta_spacing(block_report::BathFitBlockReport,
                               beta::Union{Nothing,Float64})
    beta === nothing && return nothing
    block_report.max_spacing === nothing && return nothing
    return beta * block_report.max_spacing
end

function _bathfit_horizon_ratio(block_report::BathFitBlockReport,
                                horizon::Union{Nothing,Float64})
    horizon === nothing && return nothing
    block_report.revival_time === nothing && return nothing
    return horizon / block_report.revival_time
end

"""
    audit_bathfit(report, criteria) -> BathFitAudit

Evaluate only caller-declared numerical acceptance criteria, block by block.
The sole fixed physical safety rule is that a requested real/complex-time
horizon must not exceed the reported revival time; when a horizon is supplied,
that check uses ratio one unless the caller asks for a more conservative ratio.
No residual, cone, spacing, or thermal threshold is inferred from fitter
internals.
"""
function audit_bathfit(report::BathFitReport, criteria::BathFitCriteria)
    items = BathFitAuditItem[]
    for block in keys(report.blocks)
        block_report = getproperty(report.blocks, block)
        residual = block_report.residual
        _bathfit_upper_check!(
            items, block, :absolute_residual,
            residual === nothing ? nothing : residual.absolute,
            criteria.max_absolute, "absolute reconstruction residual",
        )
        _bathfit_upper_check!(
            items, block, :l2_residual,
            residual === nothing ? nothing : residual.l2,
            criteria.max_l2, "L2 reconstruction residual",
        )
        _bathfit_upper_check!(
            items, block, :maximum_residual,
            residual === nothing ? nothing : residual.maximum,
            criteria.max_maximum, "maximum pointwise reconstruction residual",
        )
        _bathfit_upper_check!(
            items, block, :relative_l2_residual,
            residual === nothing ? nothing : residual.relative_l2,
            criteria.max_relative_l2, "relative L2 reconstruction residual",
        )
        _bathfit_upper_check!(
            items, block, :spectral_weight_error,
            block_report.spectral_weight_error,
            criteria.max_spectral_weight_error, "spectral-weight error",
        )
        _bathfit_lower_check!(
            items, block, :minimum_residue_eigenvalue,
            block_report.minimum_residue_eigenvalue,
            criteria.min_residue_eigenvalue, "minimum residue eigenvalue",
        )
        _bathfit_upper_check!(
            items, block, :psd_cone_distance,
            block_report.psd_cone_distance,
            criteria.max_psd_cone_distance, "PSD-cone distance",
        )
        _bathfit_upper_check!(
            items, block, :spacing_over_broadening,
            _bathfit_spacing_ratio(block_report, report.broadening),
            criteria.max_spacing_over_broadening,
            "maximum level spacing divided by broadening",
        )
        _bathfit_upper_check!(
            items, block, :beta_max_spacing,
            _bathfit_beta_spacing(block_report, criteria.beta),
            criteria.max_beta_spacing, "beta times maximum level spacing",
        )
        if criteria.request_horizon !== nothing
            ratio = _bathfit_horizon_ratio(block_report, criteria.request_horizon)
            limit = something(criteria.max_request_horizon_ratio, 1.0)
            _bathfit_upper_check!(
                items, block, :request_horizon_to_revival, ratio, limit,
                "request horizon divided by revival time",
            )
        end
    end
    if criteria.require_reconstruction
        reconstructed = report.reconstruction !== nothing
        push!(items, _bathfit_audit_item(
            nothing, :reconstruction_available, reconstructed,
            reconstructed ? 1.0 : 0.0, 1.0,
            reconstructed ? "a full-grid bath reconstruction is available" :
                            "a full-grid bath reconstruction is required but unavailable",
        ))
    end
    if criteria.require_mountable
        push!(items, _bathfit_audit_item(
            nothing, :mountable, report.mountable,
            report.mountable ? 1.0 : 0.0, 1.0,
            report.mountable ? "the pole expansion is Hamiltonian-mountable" :
                               "the pole expansion is not Hamiltonian-mountable",
        ))
    end
    violations = BathFitAuditItem[item for item in items if !item.passed]
    return BathFitAudit(isempty(violations), items, violations)
end
