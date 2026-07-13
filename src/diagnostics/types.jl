abstract type AbstractBathFitReport end

"""
    PoleBinDiagnostic

Numerical evidence for one raw pole residue at the common realization gate.
The original expansion is never changed to make this diagnostic pass.
"""
struct PoleBinDiagnostic
    pole_index::Int
    block::Symbol
    hermiticity_error::Float64
    minimum_eigenvalue::Float64
    tolerance::Float64
    pivots::Vector{Float64}
    reconstruction_error::Float64
    status::Symbol
end

"""Uniform full-grid residual measures for one named hybridization block."""
struct BathFitResidual
    absolute::Float64
    maximum::Float64
    l2::Float64
    relative_l2::Float64

    function BathFitResidual(absolute::Float64, maximum::Float64, l2::Float64,
                             relative_l2::Float64, ::Val{:validated})
        new(absolute, maximum, l2, relative_l2)
    end
end

function BathFitResidual(absolute::Real, maximum::Real, l2::Real, relative_l2::Real)
    values = Float64[absolute, maximum, l2, relative_l2]
    all(value -> isfinite(value) && value >= 0, values[1:3]) ||
        throw(ArgumentError(
            "BathFitResidual absolute, maximum, and L2 values must be finite and nonnegative",
        ))
    (isfinite(values[4]) || values[4] == Inf) && values[4] >= 0 ||
        throw(ArgumentError(
            "BathFitResidual relative L2 must be nonnegative and finite or Inf",
        ))
    return BathFitResidual(values..., Val(:validated))
end

"""Block-preserving report data for one named Partition block."""
struct BathFitBlockReport{C<:Tuple}
    block::Symbol
    residual::Union{Nothing,BathFitResidual}
    spectral_weight_error::Union{Nothing,Float64}
    discarded_weight::Float64
    weight_measure::Symbol
    pole_count::Int
    mode_count::Int
    bandwidth::Float64
    max_spacing::Union{Nothing,Float64}
    revival_time::Union{Nothing,Float64}
    minimum_residue_eigenvalue::Float64
    psd_cone_distance::Float64
    relative_psd_cone_distance::Float64
    boundary_curve::C

    function BathFitBlockReport(
        block::Symbol, residual::Union{Nothing,BathFitResidual},
        spectral_weight_error::Union{Nothing,Float64}, discarded_weight::Float64,
        weight_measure::Symbol, pole_count::Int, mode_count::Int, bandwidth::Float64,
        max_spacing::Union{Nothing,Float64}, revival_time::Union{Nothing,Float64},
        minimum_residue_eigenvalue::Float64, psd_cone_distance::Float64,
        relative_psd_cone_distance::Float64, boundary_curve::C,
        ::Val{:validated},
    ) where {C<:Tuple}
        new{C}(block, residual, spectral_weight_error, discarded_weight, weight_measure,
               pole_count, mode_count, bandwidth, max_spacing, revival_time,
               minimum_residue_eigenvalue, psd_cone_distance,
               relative_psd_cone_distance, boundary_curve)
    end
end

function BathFitBlockReport(
    block::Symbol; residual::Union{Nothing,BathFitResidual}=nothing,
    spectral_weight_error::Union{Nothing,Real}=nothing,
    discarded_weight::Real=0.0, weight_measure::Symbol=:none, pole_count::Integer=0,
    mode_count::Integer=0, bandwidth::Real=0.0,
    max_spacing::Union{Nothing,Real}=nothing,
    revival_time::Union{Nothing,Real}=nothing,
    minimum_residue_eigenvalue::Real=Inf, psd_cone_distance::Real=0.0,
    relative_psd_cone_distance::Real=0.0, boundary_curve=(),
)
    spectral = spectral_weight_error === nothing ? nothing : Float64(spectral_weight_error)
    spacing = max_spacing === nothing ? nothing : Float64(max_spacing)
    revival = revival_time === nothing ? nothing : Float64(revival_time)
    values = Float64[discarded_weight, bandwidth, psd_cone_distance,
                     relative_psd_cone_distance]
    all(isfinite, values) ||
        throw(ArgumentError("BathFitBlockReport scalar diagnostics must be finite"))
    (isfinite(minimum_residue_eigenvalue) || minimum_residue_eigenvalue == Inf) ||
        throw(ArgumentError(
            "BathFitBlockReport minimum_residue_eigenvalue must be finite or Inf",
        ))
    spectral === nothing || (isfinite(spectral) && spectral >= 0) ||
        throw(ArgumentError("BathFitBlockReport spectral_weight_error must be nonnegative"))
    spacing === nothing || (isfinite(spacing) && spacing > 0) ||
        throw(ArgumentError("BathFitBlockReport max_spacing must be positive"))
    revival === nothing || (isfinite(revival) && revival > 0) ||
        throw(ArgumentError("BathFitBlockReport revival_time must be positive"))
    discarded_weight >= 0 ||
        throw(ArgumentError("BathFitBlockReport discarded_weight must be nonnegative"))
    pole_count >= 0 && mode_count >= 0 ||
        throw(ArgumentError("BathFitBlockReport counts must be nonnegative"))
    psd_cone_distance >= 0 && relative_psd_cone_distance >= 0 ||
        throw(ArgumentError("BathFitBlockReport PSD cone distances must be nonnegative"))
    curve = Tuple(boundary_curve)
    return BathFitBlockReport(
        block, residual, spectral, Float64(discarded_weight), weight_measure,
        Int(pole_count), Int(mode_count), Float64(bandwidth), spacing, revival,
        Float64(minimum_residue_eigenvalue), Float64(psd_cone_distance),
        Float64(relative_psd_cone_distance), curve, Val(:validated),
    )
end

"""Measured fit, realization, and reconstruction durations in seconds."""
struct BathFitTiming
    fit_seconds::Union{Nothing,Float64}
    realization_seconds::Float64
    reconstruction_seconds::Union{Nothing,Float64}

    function BathFitTiming(fit_seconds::Union{Nothing,Float64},
                           realization_seconds::Float64,
                           reconstruction_seconds::Union{Nothing,Float64},
                           ::Val{:validated})
        new(fit_seconds, realization_seconds, reconstruction_seconds)
    end
end

function BathFitTiming(; fit_seconds::Union{Nothing,Real}=nothing,
                       realization_seconds::Real=0.0,
                       reconstruction_seconds::Union{Nothing,Real}=nothing)
    fit = fit_seconds === nothing ? nothing : Float64(fit_seconds)
    reconstruction = reconstruction_seconds === nothing ? nothing :
                     Float64(reconstruction_seconds)
    realization = Float64(realization_seconds)
    all(value -> value === nothing || (isfinite(value) && value >= 0),
        (fit, realization, reconstruction)) ||
        throw(ArgumentError("BathFitTiming values must be finite and nonnegative"))
    return BathFitTiming(fit, realization, reconstruction, Val(:validated))
end

"""An explicit nonfatal diagnostic emitted while building a BathFitReport."""
struct BathFitWarning
    code::Symbol
    block::Union{Nothing,Symbol}
    message::String
end

"""
    BathFitReport

Concrete immutable diagnostics for a fitted real-pole bath. It keeps the
layout-bearing source and optional reconstruction. A source adapted from
`GreenFunc.Gf` or `GreenFunc.BlockGf` keeps a copied original template in
`source.source_template`; its reconstructed counterpart keeps an equally typed
output template in `reconstruction.source_template`. The report also carries
named block reports, raw residue diagnostics, original kernel trace, warnings,
and measured timing.
"""
struct BathFitReport{B<:NamedTuple,T<:NamedTuple} <: AbstractBathFitReport
    source::BathFitInput
    reconstruction::Union{Nothing,BathFitInput}
    blocks::B
    plan::DiscretizationPlan
    kernel::Symbol
    mountable::Bool
    broadening::Union{Nothing,Float64}
    diagnostics::Vector{PoleBinDiagnostic}
    warnings::Vector{BathFitWarning}
    timing::BathFitTiming
    trace::T
end

"""Caller-declared, individually optional bath-fit acceptance thresholds."""
struct BathFitCriteria
    max_absolute::Union{Nothing,Float64}
    max_maximum::Union{Nothing,Float64}
    max_l2::Union{Nothing,Float64}
    max_relative_l2::Union{Nothing,Float64}
    max_spectral_weight_error::Union{Nothing,Float64}
    min_residue_eigenvalue::Union{Nothing,Float64}
    max_psd_cone_distance::Union{Nothing,Float64}
    max_spacing_over_broadening::Union{Nothing,Float64}
    beta::Union{Nothing,Float64}
    max_beta_spacing::Union{Nothing,Float64}
    request_horizon::Union{Nothing,Float64}
    max_request_horizon_ratio::Union{Nothing,Float64}
    require_reconstruction::Bool
    require_mountable::Bool
end

function _bathfit_optional_nonnegative(value, name::AbstractString)
    value === nothing && return nothing
    resolved = Float64(value)
    isfinite(resolved) && resolved >= 0 ||
        throw(ArgumentError("BathFitCriteria $name must be finite and nonnegative"))
    return resolved
end

function BathFitCriteria(; max_absolute=nothing, max_maximum=nothing,
                         max_l2=nothing,
                         max_relative_l2=nothing,
                         max_spectral_weight_error=nothing,
                         min_residue_eigenvalue=nothing,
                         max_psd_cone_distance=nothing,
                         max_spacing_over_broadening=nothing,
                         beta=nothing, max_beta_spacing=nothing,
                         request_horizon=nothing,
                         max_request_horizon_ratio=nothing,
                         require_reconstruction::Bool=false,
                         require_mountable::Bool=false)
    threshold = _bathfit_optional_nonnegative
    resolved_beta = beta === nothing ? nothing : Float64(beta)
    resolved_beta === nothing || (isfinite(resolved_beta) && resolved_beta > 0) ||
        throw(ArgumentError("BathFitCriteria beta must be finite and positive"))
    minimum_eigenvalue = min_residue_eigenvalue === nothing ? nothing :
                         Float64(min_residue_eigenvalue)
    minimum_eigenvalue === nothing || isfinite(minimum_eigenvalue) ||
        throw(ArgumentError(
            "BathFitCriteria min_residue_eigenvalue must be finite when supplied",
        ))
    horizon_ratio = threshold(max_request_horizon_ratio,
                              "max_request_horizon_ratio")
    horizon_ratio === nothing || horizon_ratio <= 1 || throw(ArgumentError(
        "BathFitCriteria max_request_horizon_ratio may not exceed one revival time",
    ))
    return BathFitCriteria(
        threshold(max_absolute, "max_absolute"),
        threshold(max_maximum, "max_maximum"), threshold(max_l2, "max_l2"),
        threshold(max_relative_l2, "max_relative_l2"),
        threshold(max_spectral_weight_error, "max_spectral_weight_error"),
        minimum_eigenvalue,
        threshold(max_psd_cone_distance, "max_psd_cone_distance"),
        threshold(max_spacing_over_broadening, "max_spacing_over_broadening"),
        resolved_beta, threshold(max_beta_spacing, "max_beta_spacing"),
        threshold(request_horizon, "request_horizon"),
        horizon_ratio,
        require_reconstruction, require_mountable,
    )
end

"""One independently auditable BathFitCriteria result."""
struct BathFitAuditItem
    block::Union{Nothing,Symbol}
    criterion::Symbol
    passed::Bool
    observed::Union{Nothing,Float64}
    threshold::Union{Nothing,Float64}
    message::String
end

"""Machine-readable separate BathFitCriteria checks and their violations."""
struct BathFitAudit
    passed::Bool
    items::Vector{BathFitAuditItem}
    violations::Vector{BathFitAuditItem}
end
