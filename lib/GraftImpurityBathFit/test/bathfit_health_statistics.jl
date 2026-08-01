function _pure_health_kernel(
    statistics::Symbol, frequency::Real, energy::Real,
)
    z = im * frequency
    if statistics === :fermion
        return inv(z - energy)
    elseif statistics === :boson
        return iszero(z) ? -1.0 + 0im : energy / (z - energy)
    end
    throw(ArgumentError("unsupported test statistics"))
end

function _pure_health_kernel_report(statistics::Symbol)
    fixture = _pure_health_fixture(
        1; basis=Symbol(:health_kernel_, statistics),
    )
    frequencies = statistics === :fermion ?
                  [-3.0, -1.0, 1.0, 3.0] :
                  [0.0, 1.0, 2.0, 3.0]
    first_energy = 0.4
    second_energy = 0.8
    target_values = ComplexF64[
        _pure_health_kernel(statistics, frequency, first_energy)
        for frequency in frequencies
    ]
    target = _pure_health_input(
        fixture.layout, frequencies, target_values;
        statistics,
    )
    second_values = ComplexF64[
        _pure_health_kernel(statistics, frequency, second_energy)
        for frequency in frequencies
    ]
    if statistics === :boson
        second_values[findfirst(iszero, frequencies)] = -0.5
    end
    second_prediction = _pure_health_input(
        fixture.layout, frequencies, second_values;
        statistics,
    )
    first_expansion = _pure_health_expansion(
        fixture, [first_energy], [1.0];
        statistics,
    )
    second_expansion = _pure_health_expansion(
        fixture, [second_energy], [1.0];
        statistics,
    )
    candidates = BathFitHealthCandidate[
        _pure_health_candidate(
            1, target, target;
            replica=1, expansion=first_expansion,
        ),
        _pure_health_candidate(
            1, target, second_prediction;
            replica=2, expansion=second_expansion,
        ),
    ]
    return analyze_bathfit(
        candidates, target, target;
        covariance=_pure_health_covariance(target),
        config=BathFitDiagnosticConfig(
            n_replicas=2, min_replicas=1,
        ),
    )
end

@testset "pure bath-fit health fermion and boson kernels" begin
    fermion = _pure_health_kernel_report(:fermion)
    boson = _pure_health_kernel_report(:boson)
    fermion_order = only(fermion.orders)
    boson_order = only(boson.orders)

    @test fermion.statistics === :fermion
    @test boson.statistics === :boson
    @test fermion_order.shape_instability !== nothing
    @test boson_order.shape_instability !== nothing
    @test all(isfinite, something(fermion_order.shape_instability).values)
    @test all(isfinite, something(boson_order.shape_instability).values)
    @test maximum(something(fermion_order.shape_instability).values) > 0
    @test maximum(something(boson_order.shape_instability).values) > 0
    @test fermion_order.zero_frequency_residual === nothing
    @test boson_order.zero_frequency_residual !== nothing
    @test something(boson_order.zero_frequency_residual).values ≈ [0.0, 0.5]
end
