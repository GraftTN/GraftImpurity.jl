"""A finite empirical distribution and its central confidence interval."""
struct BathFitMetricSummary
    values::Vector{Float64}
    mean::Float64
    std::Float64
    median::Float64
    lower::Float64
    upper::Float64

    function BathFitMetricSummary(
        values::Vector{Float64}, mean::Float64, std::Float64, median::Float64,
        lower::Float64, upper::Float64, ::Val{:validated},
    )
        new(values, mean, std, median, lower, upper)
    end
end

function BathFitMetricSummary(
    values::AbstractVector{<:Real}, mean::Real, std::Real, median::Real,
    lower::Real, upper::Real,
)
    samples = Float64.(values)
    isempty(samples) && throw(ArgumentError("BathFitMetricSummary needs values"))
    all(value -> isfinite(value) || value == Inf, samples) || throw(ArgumentError(
        "BathFitMetricSummary values must be finite or Inf",
    ))
    scalars = Float64[mean, std, median, lower, upper]
    all(value -> isfinite(value) || value == Inf, scalars) || throw(ArgumentError(
        "BathFitMetricSummary statistics must be finite or Inf",
    ))
    std >= 0 || throw(ArgumentError("BathFitMetricSummary std must be nonnegative"))
    lower <= upper || throw(ArgumentError(
        "BathFitMetricSummary interval must be ordered",
    ))
    return BathFitMetricSummary(samples, scalars..., Val(:validated))
end

function _bathfit_health_quantile(sorted_values::Vector{Float64}, probability::Float64)
    length(sorted_values) == 1 && return only(sorted_values)
    position = clamp(probability, 0.0, 1.0) * (length(sorted_values) - 1) + 1
    lower_index = floor(Int, position)
    upper_index = ceil(Int, position)
    lower_index == upper_index && return sorted_values[lower_index]
    fraction = position - lower_index
    return (1 - fraction) * sorted_values[lower_index] +
           fraction * sorted_values[upper_index]
end

function BathFitMetricSummary(
    values::AbstractVector{<:Real}; confidence::Real=0.95,
)
    samples = Float64.(values)
    isempty(samples) && throw(ArgumentError("BathFitMetricSummary needs values"))
    all(value -> isfinite(value) || value == Inf, samples) || throw(ArgumentError(
        "BathFitMetricSummary values must be finite or Inf",
    ))
    0 < confidence < 1 || throw(ArgumentError(
        "BathFitMetricSummary confidence must lie strictly between zero and one",
    ))
    ordered = sort(copy(samples))
    average = sum(samples) / length(samples)
    deviation = length(samples) == 1 ? 0.0 :
                sqrt(sum(abs2(value - average) for value in samples) /
                     (length(samples) - 1))
    tail = (1 - Float64(confidence)) / 2
    return BathFitMetricSummary(
        samples, average, deviation, _bathfit_health_quantile(ordered, 0.5),
        _bathfit_health_quantile(ordered, tail),
        _bathfit_health_quantile(ordered, 1 - tail),
    )
end

"""Health metrics aggregated over all successful fits of one requested order."""
struct BathFitOrderHealth
    order::Int
    successful_replicas::Int
    failed_fits::Int
    returned_poles::BathFitMetricSummary
    mode_counts::BathFitMetricSummary
    q_train::BathFitMetricSummary
    q_validation::BathFitMetricSummary
    generalization_factor::BathFitMetricSummary
    predictive_distance::BathFitMetricSummary
    mass_instability::BathFitMetricSummary
    shape_instability::Union{Nothing,BathFitMetricSummary}
    zero_frequency_residual::Union{Nothing,BathFitMetricSummary}
    admissible_fraction::Float64
    rashomon_predictive_diameter::Float64
    rashomon_measure_diameter::Union{Nothing,Float64}
    one_se_difference::Float64
    verdicts::NamedTuple
    diagnostics::NamedTuple
    warnings::Vector{String}
end

"""Numerical thresholds and calibration rank used by one health analysis."""
struct BathFitHealthThresholds
    confidence::Float64
    q_floor::Float64
    covariance_rank::Int
    eigenvalue_floor::Float64
    shrinkage::Float64
    validation_envelope::Float64
    predictive_envelope::Float64
    measure_envelope::Float64
    noise_scale_sensitivity::Tuple{Vararg{Float64}}
    energy_window::Union{Nothing,Tuple{Float64,Float64}}
end

"""Pure, immutable result of `analyze_bathfit`."""
struct BathFitHealthReport
    layout::FlavorLayout
    statistics::Symbol
    calibration::NamedTuple
    orders::Vector{BathFitOrderHealth}
    selected_order::Union{Nothing,Int}
    selection_distribution::Dict{Int,Float64}
    selection_interval::Union{Nothing,Tuple{Int,Int}}
    selection_stability::Float64
    thresholds::BathFitHealthThresholds
    verdicts::NamedTuple
    warnings::Vector{String}
    provenance::NamedTuple

    function BathFitHealthReport(
        layout::FlavorLayout, statistics::Symbol, calibration::NamedTuple,
        orders::Vector{BathFitOrderHealth}, selected_order::Union{Nothing,Int},
        selection_distribution::Dict{Int,Float64},
        selection_interval::Union{Nothing,Tuple{Int,Int}},
        selection_stability::Float64, thresholds::BathFitHealthThresholds,
        verdicts::NamedTuple, warnings::Vector{String}, provenance::NamedTuple,
        ::Val{:validated},
    )
        new(layout, statistics, calibration, orders, selected_order,
            selection_distribution, selection_interval, selection_stability,
            thresholds, verdicts, warnings, provenance)
    end
end

function BathFitHealthReport(
    layout::FlavorLayout, statistics::Symbol, calibration::NamedTuple,
    orders::Vector{BathFitOrderHealth}, selected_order::Union{Nothing,Integer},
    selection_distribution::AbstractDict{<:Integer,<:Real},
    selection_interval::Union{Nothing,Tuple{<:Integer,<:Integer}},
    selection_stability::Real, thresholds::BathFitHealthThresholds,
    verdicts::NamedTuple, warnings::AbstractVector{<:AbstractString},
    provenance::NamedTuple,
)
    statistics in (:fermion, :boson) || throw(ArgumentError(
        "BathFitHealthReport statistics must be :fermion or :boson",
    ))
    hasproperty(calibration, :calibrated) &&
        getproperty(calibration, :calibrated) isa Bool ||
        throw(ArgumentError(
            "BathFitHealthReport calibration needs calibrated::Bool",
        ))
    hasproperty(verdicts, :overall) &&
        getproperty(verdicts, :overall) isa Symbol ||
        throw(ArgumentError(
            "BathFitHealthReport verdicts needs overall::Symbol",
        ))
    selected = selected_order === nothing ? nothing : Int(selected_order)
    selected === nothing || any(item -> item.order == selected, orders) ||
        throw(ArgumentError("selected bath-fit order is absent from orders"))
    order_set = Set(item.order for item in orders)
    distribution = Dict{Int,Float64}(
        Int(order) => Float64(probability)
        for (order, probability) in selection_distribution
    )
    all(order -> order in order_set, keys(distribution)) || throw(ArgumentError(
        "selection_distribution contains an order absent from orders",
    ))
    all(probability -> isfinite(probability) && probability >= 0,
        values(distribution)) || throw(ArgumentError(
        "selection probabilities must be finite and nonnegative",
    ))
    isempty(distribution) || isapprox(sum(values(distribution)), 1.0;
                                      atol=64eps(Float64), rtol=0.0) ||
        throw(ArgumentError("selection probabilities must sum to one"))
    stability = Float64(selection_stability)
    isfinite(stability) && 0 <= stability <= 1 || throw(ArgumentError(
        "selection_stability must lie between zero and one",
    ))
    interval = selection_interval === nothing ? nothing :
               (Int(first(selection_interval)), Int(last(selection_interval)))
    interval === nothing || interval[1] <= interval[2] || throw(ArgumentError(
        "selection_interval must be ordered",
    ))
    isempty(distribution) == (interval === nothing) || throw(ArgumentError(
        "selection_interval must be nothing exactly when selection_distribution is empty",
    ))
    if interval !== nothing
        all(endpoint -> endpoint in order_set, interval) || throw(ArgumentError(
            "selection_interval contains an endpoint absent from orders",
        ))
        all(endpoint -> haskey(distribution, endpoint), interval) ||
            throw(ArgumentError(
                "selection_interval endpoints must belong to selection_distribution",
            ))
    end
    return BathFitHealthReport(
        layout, statistics, calibration, copy(orders), selected, distribution,
        interval, stability, thresholds, verdicts, String.(warnings),
        provenance, Val(:validated),
    )
end
