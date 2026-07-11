using Test
using Random
using GreenFunc
using SparseIR

@testset "SparseIR GreenFunc adapter" begin
    beta = 7.0
    basis = FiniteTempBasis(Fermionic(), beta, 3.0, 1e-8)
    tau_sampling = TauSampling(basis)
    tau_mesh = ImTime(beta, FERMION;
        Euv=3.0, rtol=1e-8, grid=collect(sampling_points(tau_sampling)))

    Random.seed!(0x51a7)
    reference = randn(ComplexF64, 2, 2, 3, length(basis))
    tau_data = Array{ComplexF64}(undef, 2, 2, 3, length(tau_mesh))
    evaluate!(tau_data, tau_sampling, reference; dim=4)
    gf = Gf(1:3, tau_mesh;
        target_shape=(2, 2), data=tau_data, statistics=FERMION,
        component=:self_energy,
        target_labels=((:a, :b), (:a, :b)),
        metadata=(source=:adapter_test,),
    )

    coefficients = fit_ir(gf, basis; axis=2)
    @test coefficients isa IRCoefficients
    @test coefficients.axis == 2
    @test size(coefficients.coefficients) == size(reference)
    @test coefficients.coefficients ≈ reference rtol=2e-9 atol=2e-10

    tau_result = to_imtime_ir(coefficients; grid=range(0, beta; length=17))
    @test tau_result.target_shape == (2, 2)
    @test tau_result.mesh[1] == 1:3
    @test tau_result.mesh[2] isa ImTime
    @test size(tau_result.data) == (2, 2, 3, 17)
    @test tau_result.component == :self_energy
    @test tau_result.target_labels == ((:a, :b), (:a, :b))
    @test tau_result.metadata == (source=:adapter_test,)

    default_tau = to_imtime_ir(coefficients)
    @test collect(default_tau.mesh[2]) ≈ collect(sampling_points(TauSampling(basis)))

    default_iw = to_imfreq_ir(coefficients)
    expected_n = Int.(sampling_points(MatsubaraSampling(basis; positive_only=false)))
    expected_m = div.(expected_n .- 1, 2)
    actual_m = matfreq_to_int.(Ref(default_iw.mesh[2]), matfreq(default_iw.mesh[2]))
    @test actual_m == expected_m

    # A descending GreenFunc m-grid must become SparseIR's odd n-grid in the
    # same data order, including negative frequencies.
    requested_m = [3, 1, 0, -2]
    iw_result = to_imfreq_ir(coefficients; grid=requested_m)
    @test matfreq_to_int.(Ref(iw_result.mesh[2]), matfreq(iw_result.mesh[2])) == requested_m
    direct_sampling = MatsubaraSampling(basis;
        sampling_points=2 .* requested_m .+ 1, positive_only=false)
    direct = Array{ComplexF64}(undef, 2, 2, 3, length(requested_m))
    evaluate!(direct, direct_sampling, reference; dim=4)
    @test iw_result.data ≈ direct rtol=2e-9 atol=2e-10

    iw_coefficients = fit_ir(default_iw, basis; axis=2)
    @test iw_coefficients.coefficients ≈ reference rtol=2e-9 atol=2e-10

    # Automatic selection is allowed only for a unique imaginary physical axis.
    scalar_gf = Gf(tau_mesh;
        data=vec(tau_data[1, 1, 1, :]), statistics=FERMION)
    @test fit_ir(scalar_gf, basis).axis == 1
    two_tau = Gf(tau_mesh, tau_mesh;
        data=zeros(ComplexF64, length(tau_mesh), length(tau_mesh)),
        statistics=FERMION)
    @test_throws ArgumentError fit_ir(two_tau, basis)

    wrong_beta = ImTime(beta + 1, FERMION;
        Euv=3.0, rtol=1e-8, grid=[0.0, beta + 1])
    @test_throws ArgumentError evaluate_ir(coefficients, wrong_beta)
    boson_basis = FiniteTempBasis(Bosonic(), beta, 3.0, 1e-8)
    @test_throws ArgumentError fit_ir(gf, boson_basis; axis=2)
    @test_throws MethodError evaluate_ir(coefficients, ReFreq(-1.0, 1.0, 3))
end
