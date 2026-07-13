using Graft
using Graft.Backend: FermionParity
using Graft.TestUtils: dense_hamiltonian, exact_groundstate, product_ttns
using LinearAlgebra: Hermitian, diag, eigvals, norm

@testset "fermionic Anderson real-pole star" begin
    F = fermion_ops_z2()
    partition = Partition([[:d1, :d2]])
    poles = [-0.6, 0.8]
    vectors = [ComplexF64[0.45, 0.2im],
               ComplexF64[0.1 + 0.15im, 0.35]]
    residues = [vector * vector' for vector in vectors]
    fit = PESPoleFit(poles, residues, :fermion, (; source=:synthetic);
                     residue_constraint=:psd)
    topology0 = TreeTopology(:root, [:root => :d1, :root => :d2])
    phys0 = Dict(:d1 => F.P, :d2 => F.P)
    problem = AndersonBath(fit, partition; topology=topology0, phys=phys0,
                           ops=F)

    @test problem.real_poles isa AndersonRealPoles
    @test problem.real_poles.energies ≈ poles
    @test length(problem.bath_sites) == 2
    @test all(site -> haskey(problem.phys, site), problem.bath_sites)
    @test all(anchor -> anchor in (:d1, :d2), problem.anchors)
    @test length(problem.H) == 10

    reconstructed = [zeros(ComplexF64, 2, 2) for _ in poles]
    for (coupling, pole_index) in
            zip(problem.real_poles.couplings,
                problem.real_poles.pole_indices)
        reconstructed[pole_index] .+= coupling * coupling'
    end
    @test all(norm(reconstructed[k] - residues[k]) < 1e-12
              for k in eachindex(poles))

    dense = dense_hamiltonian(problem.H, problem.topology, problem.phys)
    one_particle = zeros(ComplexF64, 4, 4)
    for k in eachindex(problem.real_poles.energies)
        mode = 2 + k
        one_particle[mode, mode] = problem.real_poles.energies[k]
        for a in 1:2
            coupling = problem.real_poles.couplings[k][a]
            one_particle[a, mode] = coupling
            one_particle[mode, a] = conj(coupling)
        end
    end
    number = OpSum()
    for site in vcat([:d1, :d2], problem.bath_sites)
        number += Term(1.0, SiteOp(site, :N, F.N))
    end
    dense_number = dense_hamiltonian(
        number, problem.topology, problem.phys)
    one_particle_indices = findall(
        value -> abs(value - 1) < 1e-12, real.(diag(dense_number)))
    dense_one_particle = dense[one_particle_indices, one_particle_indices]
    @test norm(dense - dense') < 1e-12
    @test eigvals(Hermitian(dense_one_particle)) ≈
          eigvals(Hermitian(one_particle)) atol=1e-11

    @test_throws ArgumentError AndersonRealPoles(
        PESPoleFit(poles, residues, :boson, (;); residue_constraint=:psd),
        partition)
    @test_throws ArgumentError AndersonRealPoles(
        fit, Partition([[:d1], [:d2]]))
end

@testset "solve finite Anderson real-pole star" begin
    F = fermion_ops_z2()
    partition = Partition([[:imp]])
    fit = PESPoleFit([0.5], [reshape(ComplexF64[0.36], 1, 1)],
                     :fermion, (; source=:synthetic);
                     residue_constraint=:psd)
    problem = AndersonBath(fit, partition;
                           topology=TreeTopology(:imp, Pair{Symbol,Symbol}[]),
                           phys=Dict(:imp => F.P), ops=F)
    Hloc = OpSum() + Term(-0.2, SiteOp(:imp, :N, F.N))
    sectors = Dict(:imp => FermionParity(1),
                   only(problem.bath_sites) => FermionParity(0))
    psi0 = product_ttns(ComplexF64, problem.topology, problem.phys, sectors)
    Nimp = OpSum() + Term(1.0, SiteOp(:imp, :N, F.N))
    result = solve(problem, Hloc; psi0, observables=(Nimp=Nimp,),
                   times=[0.0, 0.02],
                   evolver=TDVP2(trunc=TruncationScheme(maxdim=4,
                                                        atol=1e-12),
                                 verbose=false),
                   taus=[0.0, 1.0], beta_eff=1.0,
                   thermal_evolver=TDVP2(
                       trunc=TruncationScheme(maxdim=8, atol=1e-12),
                       verbose=false),
                   thermal_nsteps=1, thermal_prop_nsteps=1,
                   trunc=TruncationScheme(maxdim=4, atol=1e-12),
                   nsweeps=4, krylovdim=12, verbose=false)

    dense = dense_hamiltonian(Hloc + problem.H,
                              problem.topology, problem.phys)
    exact_energy, _ = exact_groundstate(dense)
    @test result.energy ≈ exact_energy atol=1e-10
    @test result.state === psi0
    @test 0 <= real(result.observables.Nimp) <= 1
    @test result.real_time.temperature == 0.0
    @test result.real_time.convention == :raw_correlator
    @test size(result.real_time.particle) == (1, 1)
    @test size(result.real_time.hole) == (1, 1)
    @test length(result.real_time.particle[1, 1]) == 2
    @test all(isfinite, result.real_time.particle[1, 1].values)
    @test all(isfinite, result.real_time.hole[1, 1].values)
    @test result.imaginary_time.beta_eff == 1.0
    @test result.imaginary_time.convention == :raw_correlator
    @test result.imaginary_time.taus == [0.0, 1.0]
    @test size(result.imaginary_time.particle) == (1, 1)
    @test all(isfinite, result.imaginary_time.particle[1, 1].values)
    @test all(isfinite, result.imaginary_time.hole[1, 1].values)
    @test result.imaginary_time.particle[1, 1].values[1] +
          result.imaginary_time.hole[1, 1].values[1] ≈ 1 atol=1e-10

    wrong = product_ttns(
        ComplexF64, TreeTopology(:other, Pair{Symbol,Symbol}[]),
        Dict(:other => F.P), Dict(:other => FermionParity(0)))
    @test_throws ArgumentError solve(problem, Hloc; psi0=wrong,
                                     verbose=false)
    @test_throws ArgumentError solve(problem, Hloc; psi0,
                                     times=[0.0], verbose=false)
    @test_throws ArgumentError solve(problem, Hloc; psi0,
                                     taus=[0.0], verbose=false)
    @test_throws ArgumentError solve(problem, Hloc; psi0,
                                     beta_eff=2.0, verbose=false)
    @test_throws ArgumentError solve(
        problem, Hloc; psi0, times=[0.0],
        evolver=TDVP2(trunc=TruncationScheme(maxdim=4), verbose=false),
        beta_eff=2.0, verbose=false)
end
