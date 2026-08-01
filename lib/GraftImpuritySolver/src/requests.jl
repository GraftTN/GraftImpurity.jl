function _require_complex_contour_evolver(request::ComplexTimeRequest)
    _complex_request_needs_general_steps(request) || return request
    supports_complex_step(typeof(request.evolver)) || throw(ArgumentError(
        "$(typeof(request.evolver)) supports only real nonpositive imaginary-time " *
        "steps and cannot evolve this mixed ComplexTimeRequest contour",
    ))
    return request
end

function _request_time_horizon(request::SolveRequest)
    horizon = nothing
    if request.real_time !== nothing
        horizon = maximum(request.real_time.times)
    end
    if request.complex_time !== nothing
        grid, _ = _complex_contour_grid(request.complex_time)
        contour_horizon = maximum(abs, grid)
        horizon = horizon === nothing ? contour_horizon : max(horizon, contour_horizon)
    end
    return horizon
end

function _request_beta(request::SolveRequest)
    request.imaginary_time === nothing && return nothing
    return request.imaginary_time.temperature.beta_eff
end

function _validate_solve_request_contract(request::SolveRequest)
    request.real_time === nothing || request.real_time.temperature isa ZeroTemperature ||
        throw(ArgumentError("RealTimeRequest must carry ZeroTemperature()"))
    request.complex_time === nothing || request.complex_time.temperature isa ZeroTemperature ||
        throw(ArgumentError("ComplexTimeRequest must carry ZeroTemperature()"))
    request.imaginary_time === nothing || request.imaginary_time.temperature isa FiniteTemperature ||
        throw(ArgumentError("ImaginaryTimeRequest must carry FiniteTemperature(beta_eff)"))
    if request.real_time !== nothing && any(>(0), request.real_time.times)
        supports_complex_step(typeof(request.real_time.evolver)) || throw(ArgumentError(
            "$(typeof(request.real_time.evolver)) supports only real nonpositive " *
            "imaginary-time steps and cannot run RealTimeRequest",
        ))
    end
    request.complex_time === nothing ||
        _require_complex_contour_evolver(request.complex_time)
    return request
end
