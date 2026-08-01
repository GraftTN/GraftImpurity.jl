"""
    DMFTBathBlockRecord

Passive diagnostics for one named bath block at one DMFT iteration. Updates
compare adjacent iterations in the input's native sample representation.
`measure_shape_update` is the one-dimensional Wasserstein distance between
normalized positive trace measures and therefore has pole-energy units;
`measure_mass_update` is a relative total-trace-mass change.
"""
struct DMFTBathBlockRecord
    block::Symbol
    data_update::Union{Nothing,Float64}
    prediction_update::Union{Nothing,Float64}
    fit_amplification::Union{Nothing,Float64}
    loop_amplification::Union{Nothing,Float64}
    measure_mass_update::Union{Nothing,Float64}
    measure_shape_update::Union{Nothing,Float64}
end

"""
    DMFTBathComplexity

Optional fit-complexity evidence retained without interpreting it as a DMFT
control parameter. The selected order's training/validation Q summaries,
returned-pole summary, and realized-mode summary are retained by identity from
an optional `BathFitHealthReport`; scalar pole and realized-mode counts fall
back to the ordinary fit report.
"""
struct DMFTBathComplexity{QT,QV,P,M}
    order::Union{Nothing,Int}
    q_train::QT
    q_validation::QV
    returned_poles::P
    mode_counts::M
    pole_count::Union{Nothing,Int}
    mode_count::Union{Nothing,Int}
end

"""
    DMFTBathIterationRecord

Immutable record appended by [`update!`](@ref). The monitor retains the target
and fitted prediction so that its evidence can be independently recomputed.
`u_k` is the signed adjacent target update and `v_k` is the signed adjacent
fitted-prediction update, both represented as layout-preserving
`BathFitInput`s; the scalar update fields are their relative Frobenius norms.
It does not retain a solver, a self-consistency callback, or mutable DMFT state.
"""
struct DMFTBathIterationRecord{B<:NamedTuple,C<:NamedTuple,H}
    iteration::Int
    target::BathFitInput
    prediction::BathFitInput
    blocks::B
    complexity::C
    health::H
    u_k::Union{Nothing,BathFitInput}
    v_k::Union{Nothing,BathFitInput}
    data_update::Union{Nothing,Float64}
    prediction_update::Union{Nothing,Float64}
    fit_amplification::Union{Nothing,Float64}
    loop_amplification::Union{Nothing,Float64}
    measure_mass_update::Union{Nothing,Float64}
    measure_shape_update::Union{Nothing,Float64}
end

"""One persisted monitor verdict together with its explicit calibration basis."""
struct DMFTBathVerdictEvidence
    verdict::Symbol
    calibration::Symbol
    observed::Float64
    threshold::Float64
    persistence::Int
    message::String
end

"""
    DMFTBathMonitorReport

Snapshot returned by [`dmft_bath_report`](@ref). With no supplied
health/bootstrap envelope, `calibration == :relative` means every threshold is
derived only from earlier monitor values by a rolling median/MAD rule. Such a
report makes no absolute noise or stability claim.
"""
struct DMFTBathMonitorReport
    records::Vector{DMFTBathIterationRecord}
    transitions::Int
    window::Int
    persistence::Int
    calibration::Symbol
    verdicts::Vector{Symbol}
    evidence::Vector{DMFTBathVerdictEvidence}
end

struct _DMFTPositiveMeasure
    energies::Vector{Float64}
    weights::Vector{Float64}
    mass::Float64
end

struct _DMFTMeasureSnapshot{B<:NamedTuple}
    blocks::B
end

"""
    DMFTBathMonitor(; window=5, persistence=3,
                    health_report=nothing, bootstrap_envelope=nothing)

Construct an opt-in, passive DMFT bath monitor. It observes only values passed
to [`update!`](@ref). Supplying a health report or bootstrap envelope enables
calibrated upper bounds when that object exposes a matching bound; all other
axes remain explicitly relative.
"""
mutable struct DMFTBathMonitor{C}
    window::Int
    persistence::Int
    calibration_source::C
    records::Vector{DMFTBathIterationRecord}
    measures::Vector{Union{Nothing,_DMFTMeasureSnapshot}}

    function DMFTBathMonitor(
        window::Int, persistence::Int, calibration_source::C,
        ::Val{:validated},
    ) where {C}
        new{C}(
            window, persistence, calibration_source,
            DMFTBathIterationRecord[],
            Union{Nothing,_DMFTMeasureSnapshot}[],
        )
    end
end

function DMFTBathMonitor(;
    window::Integer=5,
    persistence::Integer=3,
    health_report=nothing,
    bootstrap_envelope=nothing,
)
    window_value = Int(window)
    persistence_value = Int(persistence)
    window_value >= 2 || throw(ArgumentError(
        "DMFTBathMonitor window must contain at least two transitions",
    ))
    1 <= persistence_value <= window_value || throw(ArgumentError(
        "DMFTBathMonitor persistence must lie between one and window",
    ))
    health_report === nothing || bootstrap_envelope === nothing ||
        throw(ArgumentError(
            "supply either health_report or bootstrap_envelope, not both",
        ))
    source = bootstrap_envelope === nothing ? health_report : bootstrap_envelope
    return DMFTBathMonitor(
        window_value, persistence_value, source, Val(:validated),
    )
end

DMFTBathMonitor(health_report; window::Integer=5, persistence::Integer=3) =
    DMFTBathMonitor(; window, persistence, health_report)
