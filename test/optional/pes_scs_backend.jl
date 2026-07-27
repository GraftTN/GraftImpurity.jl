using Test
using LinearAlgebra: Hermitian, eigmin, norm
using GraftImpurity
using SCS

fermion_kernel(z, pole) = inv(z - pole)

function matrix_samples(poles, weights, frequencies)
    return [sum(fermion_kernel(z, poles[k]) .* weights[k]
                for k in eachindex(poles))
            for z in frequencies]
end

@testset "optional SCS PES backend" begin
    beta = 20.0
    frequencies = im .* (vcat(collect(-24:-1), collect(1:24)) .* π / beta)
    poles = [-1.0, 0.2, 1.3]
    vectors = [
        ComplexF64[0.7, 0.2im],
        ComplexF64[0.3 + 0.1im, 0.8],
        ComplexF64[0.4, -0.2 + 0.3im],
    ]
    weights = [vector * vector' for vector in vectors]
    values = matrix_samples(poles, weights, frequencies)

    fit = pes_fit(
        values, frequencies; n_poles=3, solver=:sdp,
        conic_solver=:scs, maxiter=0)

    @test fit.diagnostics.requested_conic_solver == :scs
    @test fit.diagnostics.used_conic_solver == :scs
    @test fit.diagnostics.sdp_solves == 1
    @test all(weight -> norm(weight - weight') < 1e-10, fit.weights)
    @test all(weight -> eigmin(Hermitian(weight)) >= -1e-8, fit.weights)
    @test fit.diagnostics.max_abs_error < 2e-6
end
