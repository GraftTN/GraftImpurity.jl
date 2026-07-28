using JLD2
using Graft
using Graft.TestUtils
using LinearAlgebra: Hermitian, eigen

# Cross-check against the committed one-time TRIQS CTSEG/pyed reference
# (scratch/ctseg_reference/convert.jl regenerated it from the TRIQS benchmark
# HDF5; CTSEG itself is never executed by this package).

const CTSEG_REFERENCE =
    load(joinpath(@__DIR__, "data", "ctseg_pyed_reference.jld2"))["artifact"]

function _ctseg_reference_hamiltonian(params, S, B)
    H = OpSum()
    H += Term(-params.mu, SiteOp(:d_up, :N, S.N))
    H += Term(-params.mu, SiteOp(:d_dn, :N, S.N))
    H += Term(params.U, SiteOp(:d_up, :N, S.N), SiteOp(:d_dn, :N, S.N))
    for (d, b) in ((:d_up, :bath_up), (:d_dn, :bath_dn))
        H += Term(params.V, SiteOp(d, :Cd, S.Sm), SiteOp(b, :C, S.Sp))
        H += Term(conj(params.V), SiteOp(d, :C, S.Sp), SiteOp(b, :Cd, S.Sm))
    end
    H += Term(params.omega0, SiteOp(:ph, :N, B.N))
    H += Term(params.g, SiteOp(:d_up, :N, S.N), SiteOp(:ph, :X, B.X))
    H += Term(params.g, SiteOp(:d_dn, :N, S.N), SiteOp(:ph, :X, B.X))
    return H
end

function _lehmann_matsubara(Hd, Cd, beta, iw)
    decomposition = eigen(Hermitian(Matrix{ComplexF64}(Hd)))
    energies = real.(decomposition.values)
    vectors = decomposition.vectors
    weights = exp.(-beta .* (energies .- minimum(energies)))
    partition = sum(weights)
    A = vectors' * Matrix{ComplexF64}(Cd) * vectors
    return ComplexF64[
        sum((weights[m] + weights[n]) / partition * abs2(A[m, n]) /
            (z + energies[m] - energies[n])
            for m in eachindex(energies), n in eachindex(energies))
        for z in iw]
end

@testset "CTSEG pyed reference: kernels and Dyson" begin
    artifact = CTSEG_REFERENCE
    params = artifact.parameters
    ns = 0:(params.n_iw - 1)
    iw = im .* (2 .* ns .+ 1) .* pi ./ params.beta

    action = FiniteModeAction(;
        beta=params.beta, mu=params.mu, static_interaction=params.U, n0=0.0,
        bath_energies=[params.eps_bath], bath_couplings=[params.V],
        boson_frequencies=[params.omega0], boson_couplings=[params.g],
        orbital_convention=:spinful, density_convention=:shifted)
    delta = hybridization_iw(action, ns).values
    g0inv = iw .+ params.mu .- delta

    @test only(retarded_interaction_iv(action, [0]).values) ≈
        params.D0_iw0 rtol=1e-12

    residual = 1 ./ artifact.pyed_G_iw .-
        (g0inv .- artifact.pyed_Sigma_iw)
    @test maximum(abs.(residual)) < 1e-10
end

@testset "CTSEG pyed reference: dense ED and Monte Carlo" begin
    artifact = CTSEG_REFERENCE
    params = artifact.parameters
    ns = 0:(params.n_iw - 1)
    iw = im .* (2 .* ns .+ 1) .* pi ./ params.beta

    S = spin_ops()
    B = boson_ops(params.pyed_nb_states - 1)
    topo = TreeTopology(
        :d_up,
        [:d_up => :bath_up, :bath_up => :d_dn,
         :d_dn => :bath_dn, :bath_dn => :ph])
    phys = Dict(
        :d_up => S.P, :bath_up => S.P,
        :d_dn => S.P, :bath_dn => S.P, :ph => B.P)
    Hd = dense_hamiltonian(
        _ctseg_reference_hamiltonian(params, S, B), topo, phys)
    C_up = dense_hamiltonian(
        OpSum() + Term(1.0, SiteOp(:d_up, :C, S.Sp)), topo, phys)
    G_ed = _lehmann_matsubara(Hd, C_up, params.beta, iw)

    @test maximum(abs.(G_ed .- artifact.pyed_G_iw)) < 1e-8

    # The CTSEG Monte Carlo blocks bracket the exact curve; up/down are
    # paramagnetic, so their mean estimates G with only statistical error.
    half = length(artifact.ctseg_G_iw_up) ÷ 2
    G_mc = (artifact.ctseg_G_iw_up[(half + 1):end] .+
            artifact.ctseg_G_iw_down[(half + 1):end]) ./ 2
    @test maximum(abs.(G_mc[1:8] .- artifact.pyed_G_iw[1:8])) < 2e-2
end
