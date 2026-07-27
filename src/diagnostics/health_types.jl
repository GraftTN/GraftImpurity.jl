"""
    AbstractBathFitPerturbation

Noise information used to calibrate bath-fit residuals. Concrete perturbations
are immutable input data; health analysis never mutates their covariance or
replicas.
"""
abstract type AbstractBathFitPerturbation end

"""
    CovariancePerturbation(covariance, scale=1)

Explicit covariance for the real embedding of a complex residual vector. The
embedding is `[real(residual); imag(residual)]`, so a prediction with `n`
complex scalar entries needs a `2n × 2n` covariance.
"""
struct CovariancePerturbation <: AbstractBathFitPerturbation
    covariance::Matrix{Float64}
    scale::Float64

    function CovariancePerturbation(
        covariance::Matrix{Float64}, scale::Float64, ::Val{:validated},
    )
        new(covariance, scale)
    end
end

function CovariancePerturbation(covariance::AbstractMatrix{<:Real}, scale::Real=1)
    size(covariance, 1) == size(covariance, 2) || throw(DimensionMismatch(
        "CovariancePerturbation covariance must be square",
    ))
    matrix = Matrix{Float64}(covariance)
    all(isfinite, matrix) || throw(ArgumentError(
        "CovariancePerturbation covariance must be finite",
    ))
    resolved_scale = Float64(scale)
    isfinite(resolved_scale) && resolved_scale > 0 || throw(ArgumentError(
        "CovariancePerturbation scale must be finite and positive",
    ))
    tolerance = sqrt(eps(Float64)) * max(opnorm(matrix), 1.0)
    norm(matrix - transpose(matrix)) <= tolerance || throw(ArgumentError(
        "CovariancePerturbation covariance must be symmetric",
    ))
    eigenvalues = eigvals(Hermitian((matrix + transpose(matrix)) / 2))
    minimum(eigenvalues; init=0.0) >= -tolerance || throw(ArgumentError(
        "CovariancePerturbation covariance must be positive semidefinite",
    ))
    return CovariancePerturbation(
        (matrix + transpose(matrix)) / 2, resolved_scale, Val(:validated),
    )
end

"""
    EmpiricalReplicaPerturbation(replicas)

Empirical target replicas used to estimate a shrunk covariance. Every replica
must have the same domain, statistics, grid, named blocks, and matrix shapes as
the analyzed target.
"""
struct EmpiricalReplicaPerturbation <: AbstractBathFitPerturbation
    replicas::Vector{BathFitInput}

    function EmpiricalReplicaPerturbation(
        replicas::Vector{BathFitInput}, ::Val{:validated},
    )
        new(replicas)
    end
end

function EmpiricalReplicaPerturbation(replicas::AbstractVector{<:BathFitInput})
    isempty(replicas) && throw(ArgumentError(
        "EmpiricalReplicaPerturbation needs at least one replica",
    ))
    return EmpiricalReplicaPerturbation(
        BathFitInput[replica for replica in replicas], Val(:validated),
    )
end

"""
    BathFitDiagnosticConfig(; kwargs...)

All statistical choices made by `analyze_bathfit`. Defaults are public and
stored verbatim in the returned provenance and threshold records.
"""
struct BathFitDiagnosticConfig
    n_replicas::Int
    confidence::Float64
    min_replicas::Int
    starts::Int
    seed::UInt64
    rashomon_tolerance::Float64
    noise_scale_sensitivity::Tuple{Vararg{Float64}}
    energy_window::Union{Nothing,Tuple{Float64,Float64}}

    function BathFitDiagnosticConfig(
        n_replicas::Int, confidence::Float64, min_replicas::Int, starts::Int,
        seed::UInt64, rashomon_tolerance::Float64,
        noise_scale_sensitivity::Tuple{Vararg{Float64}},
        energy_window::Union{Nothing,Tuple{Float64,Float64}},
        ::Val{:validated},
    )
        new(n_replicas, confidence, min_replicas, starts, seed,
            rashomon_tolerance, noise_scale_sensitivity, energy_window)
    end
end

function BathFitDiagnosticConfig(;
    n_replicas::Integer=200,
    confidence::Real=0.95,
    min_replicas::Integer=32,
    starts::Integer=1,
    seed::UInt64=UInt64(0x6a09e667f3bcc909),
    rashomon_tolerance::Real=sqrt(eps(Float64)),
    noise_scale_sensitivity=(0.5, 1.0, 2.0),
    energy_window=nothing,
)
    n_replicas > 0 || throw(ArgumentError(
        "BathFitDiagnosticConfig n_replicas must be positive",
    ))
    min_replicas > 0 || throw(ArgumentError(
        "BathFitDiagnosticConfig min_replicas must be positive",
    ))
    starts > 0 || throw(ArgumentError(
        "BathFitDiagnosticConfig starts must be positive",
    ))
    resolved_confidence = Float64(confidence)
    0 < resolved_confidence < 1 || throw(ArgumentError(
        "BathFitDiagnosticConfig confidence must lie strictly between zero and one",
    ))
    tolerance = Float64(rashomon_tolerance)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError(
        "BathFitDiagnosticConfig rashomon_tolerance must be finite and nonnegative",
    ))
    scales = Tuple(Float64(scale) for scale in noise_scale_sensitivity)
    isempty(scales) && throw(ArgumentError(
        "BathFitDiagnosticConfig noise_scale_sensitivity may not be empty",
    ))
    all(scale -> isfinite(scale) && scale > 0, scales) || throw(ArgumentError(
        "BathFitDiagnosticConfig noise scales must be finite and positive",
    ))
    window = if energy_window === nothing
        nothing
    else
        length(energy_window) == 2 || throw(ArgumentError(
            "BathFitDiagnosticConfig energy_window needs two endpoints",
        ))
        bounds = (Float64(first(energy_window)), Float64(last(energy_window)))
        all(isfinite, bounds) && bounds[1] < bounds[2] || throw(ArgumentError(
            "BathFitDiagnosticConfig energy_window must be finite and increasing",
        ))
        bounds
    end
    return BathFitDiagnosticConfig(
        Int(n_replicas), resolved_confidence, Int(min_replicas), Int(starts),
        seed, tolerance, scales, window, Val(:validated),
    )
end

"""
    BathFitHealthCandidate

One runner-produced fit attempt. `replica` and `start` are stable pairing keys
used by paired order selection. Failed candidates retain their message but
carry no predictions. Successful candidates must carry both predictions.
"""
struct BathFitHealthCandidate
    order::Int
    replica::Int
    start::Int
    status::Symbol
    expansion::Union{Nothing,PoleExpansion}
    returned_poles::Int
    mode_count::Int
    mountable::Bool
    training_target::Union{Nothing,BathFitInput}
    training_prediction::Union{Nothing,BathFitInput}
    validation_prediction::Union{Nothing,BathFitInput}
    message::String

    function BathFitHealthCandidate(
        order::Int, replica::Int, start::Int, status::Symbol,
        expansion::Union{Nothing,PoleExpansion}, returned_poles::Int,
        mode_count::Int, mountable::Bool,
        training_target::Union{Nothing,BathFitInput},
        training_prediction::Union{Nothing,BathFitInput},
        validation_prediction::Union{Nothing,BathFitInput}, message::String,
        ::Val{:validated},
    )
        new(order, replica, start, status, expansion, returned_poles, mode_count,
            mountable, training_target, training_prediction,
            validation_prediction, message)
    end
end

function BathFitHealthCandidate(
    order::Integer, replica::Integer, start::Integer, status::Symbol,
    expansion::Union{Nothing,PoleExpansion}, returned_poles::Integer,
    mode_count::Integer, mountable::Bool,
    training_target::Union{Nothing,BathFitInput},
    training_prediction::Union{Nothing,BathFitInput},
    validation_prediction::Union{Nothing,BathFitInput}, message::AbstractString="",
)
    order > 0 || throw(ArgumentError("bath-fit candidate order must be positive"))
    replica > 0 || throw(ArgumentError("bath-fit candidate replica must be positive"))
    start > 0 || throw(ArgumentError("bath-fit candidate start must be positive"))
    status in (:success, :nonconverged, :failed) || throw(ArgumentError(
        "bath-fit candidate status must be :success, :nonconverged, or :failed",
    ))
    returned_poles >= 0 && mode_count >= 0 || throw(ArgumentError(
        "bath-fit candidate counts must be nonnegative",
    ))
    mountable && expansion === nothing && throw(ArgumentError(
        "a mountable bath-fit candidate must carry its pole expansion",
    ))
    if status === :failed
        training_prediction === nothing && validation_prediction === nothing ||
            throw(ArgumentError("failed candidates may not carry predictions"))
    else
        training_target !== nothing && training_prediction !== nothing &&
            validation_prediction !== nothing ||
            throw(ArgumentError(
                "successful candidates need a training target and both predictions",
            ))
    end
    return BathFitHealthCandidate(
        Int(order), Int(replica), Int(start), status, expansion,
        Int(returned_poles), Int(mode_count), mountable, training_target,
        training_prediction, validation_prediction, String(message),
        Val(:validated),
    )
end

function BathFitHealthCandidate(
    order::Integer;
    replica::Integer=1,
    start::Integer=1,
    status::Symbol=:success,
    expansion::Union{Nothing,PoleExpansion}=nothing,
    returned_poles::Integer=order,
    mode_count::Integer=returned_poles,
    mountable::Bool=expansion !== nothing,
    training_target::Union{Nothing,BathFitInput}=nothing,
    training_prediction::Union{Nothing,BathFitInput}=nothing,
    validation_prediction::Union{Nothing,BathFitInput}=nothing,
    message::AbstractString="",
)
    return BathFitHealthCandidate(
        order, replica, start, status, expansion, returned_poles, mode_count,
        mountable, training_target, training_prediction,
        validation_prediction, message,
    )
end
