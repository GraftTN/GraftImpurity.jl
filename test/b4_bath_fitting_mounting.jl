using Test
using GraftImpurity
using Graft: TreeTopology, nnodes, spin_ops, boson_ops, ttno_from_opsum
using Graft.TestUtils: random_ttns, dense_hamiltonian, to_dense
using Graft.Backend: ℂ
using LinearAlgebra: norm
using Random: Xoshiro

const BATH_RNG = Xoshiro(20260709)

@testset "boson bath fitting and mounting" begin
    S = spin_ops()
    B = boson_ops(2)
    P = Partition([[:imp]])
    J(ω) = 0.2 * ω
    bath = fit_bath(J, P; nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath isa RealPoles
    @test bath.diagnostics.domain == :real_axis
    @test bath.blocks == P.blocks
    @test bath.block_ranges == [1:3]
    @test bath.poles ≈ [0.75, 1.25, 1.75]
    @test bath.residues ≈ 0.2 .* bath.poles .* 0.5
    @test couplings(bath) ≈ sqrt.(bath.residues)
    @test bath.diagnostics.block_diagnostics[1].rel_weight_change < 0.2
    frozen_bath = @test_logs (:warn, r"FROZEN") fit_bath(
        J, P; T=1.0, nmodes=2, ωmin=0.1, ωmax=1.0)
    @test frozen_bath === nothing

    νs = [0.0, 0.4, 1.0, 2.0, 4.0, 8.0]
    exact_poles = [0.75, 1.25, 1.75]
    exact_residues = [0.04, 0.02, 0.01]
    Uν = [sum(2 * ω * r / (ν^2 + ω^2)
              for (ω, r) in zip(exact_poles, exact_residues)) for ν in νs]
    bath_m = fit_bath((; frequencies=im .* νs, values=Uν), P;
                      domain=:matsubara, nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath_m isa RealPoles
    @test bath_m.diagnostics.domain == :matsubara
    @test bath_m.poles ≈ exact_poles
    @test bath_m.residues ≈ exact_residues atol = 1e-10
    @test matsubara_reconstruct(bath_m, im .* νs) ≈ Uν atol = 1e-10
    @test bath_m.diagnostics.block_diagnostics[1].relative_residual < 1e-10

    P2 = Partition([[:a], [:b]])
    Uν2 = 0.5 .* Uν
    bath_m2 = fit_bath((; frequencies=νs, values=hcat(Uν, Uν2)), P2;
                       domain=:matsubara, nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath_m2.block_ranges == [1:3, 4:6]
    @test matsubara_reconstruct(bath_m2, νs; block=1) ≈ Uν atol = 1e-10
    @test matsubara_reconstruct(bath_m2, νs; block=2) ≈ Uν2 atol = 1e-10

    topo = TreeTopology(:imp, Pair{Symbol,Symbol}[])
    mounted = mount_bath(topo, bath, P; prefix=:ph)
    @test nnodes(mounted.topology) == 4
    @test mounted.sites == [:ph1_1_1, :ph1_2_1, :ph1_3_1]
    @test all(==(:imp), mounted.anchors)
    mounted_chain = mount_bath(topo, bath, P; mode=:chain, prefix=:ch)
    @test nnodes(mounted_chain.topology) == 4
    @test mounted_chain.sites == [:ch1_1, :ch1_2, :ch1_3]
    @test mounted_chain.block_sites == [mounted_chain.sites]

    bb = BosonBath(J; partition=P, topology=topo, matter_ops=S, boson_ops=B,
                   nmodes=2, ωmin=0.5, ωmax=1.5, prefix=:bfit, density=:Z)
    @test bb.bath.poles ≈ [0.75, 1.25]
    @test bb.sites == [:bfit1_1_1, :bfit1_2_1]
    @test Set(keys(bb.phys)) == Set(bb.sites)
    phys = merge(Dict(:imp => S.P), bb.phys)
    O = ttno_from_opsum(bb.H, bb.topology, phys; hermitian=true)
    ψ = random_ttns(BATH_RNG, ComplexF64, bb.topology, phys, ℂ^3)
    @test norm(to_dense(O) - dense_hamiltonian(bb.H, ψ)) < 1e-12
end
