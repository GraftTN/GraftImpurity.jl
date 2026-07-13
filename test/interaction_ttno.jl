using Test
using Graft
using GraftImpurity
using Graft.TestUtils: to_dense
using LinearAlgebra: Hermitian, eigvals

function _m6_ttno_layout()
    flavors_ = [:a, :b, :c, :d]
    return FlavorLayout(
        flavors_, Dict(flavor => Symbol(:imp_, flavor) for flavor in flavors_),
        Dict(Symbol(:imp_, flavor) => [flavor] for flavor in flavors_);
        basis=:m6_ttno,
    )
end

function _m6_ttno_bath(layout::FlavorLayout)
    partition = Partition(:all => collect(flavors(layout)))
    orbitals = BathOrbitals(
        [0.27], [ComplexF64[0.17, 0.09im, -0.05, 0.03im]], [1], [1], [:a];
        layout, partition,
    )
    return partition, DiscreteBath(layout, partition, orbitals; statistics=:fermion)
end

function _m6_ttno_interaction(layout::FlavorLayout)
    tensor = zeros(ComplexF64, 4, 4, 4, 4)
    coefficient = 0.23 - 0.11im
    tensor[1, 2, 3, 4] = 2 * coefficient
    tensor[3, 4, 1, 2] = 2 * conj(coefficient)
    return FullCoulombInteraction(tensor, BareCoulombTensor(), layout)
end

@testset "M6 full-Coulomb TTNO integration" begin
    layout = _m6_ttno_layout()
    partition, bath = _m6_ttno_bath(layout)
    interaction = _m6_ttno_interaction(layout)
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    soc = ImpurityOneBody(
        ComplexF64[0 0.04im 0 0; -0.04im 0 0 0; 0 0 0 0; 0 0 0 0],
        layout; label=:soc,
    )
    spectra = Vector{Vector{Float64}}()
    for plan in (T3NS(layout), FTPS(layout))
        topology = impurity_topology(plan, partition, bath)
        mounted = mount_bath(topology, bath; sector=ParticleNumberSector())
        assembled = lower_hamiltonian(
            mounted, interaction, operators;
            soc, compression_atol=1e-12,
        )
        physical = Dict(site => getproperty(mounted.phys, site)
                        for site in propertynames(mounted.phys))
        uncompressed = ttno_from_opsum(
            assembled.opsum, mounted.topology, physical; hermitian=true,
        )
        @test to_dense(assembled.operator) ≈ to_dense(uncompressed) atol=1e-10
        @test assembled.compression.sector_aware
        @test assembled.compression.mode === :exact_rank
        @test !isempty(assembled.compression.edges)
        @test all(edge -> !isempty(edge.sectors), assembled.compression.edges)
        push!(spectra, eigvals(Hermitian(to_dense(assembled.operator))))
    end
    @test spectra[1] ≈ spectra[2] atol=1e-10
end
