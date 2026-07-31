using Test
using Graft
using GraftImpurity
using Graft.TestUtils: spin_ops

function _residual_driven_consumer_fixture()
    layout = FlavorLayout(
        [:d], Dict(:d => :imp), Dict(:imp => [:d]);
        basis=:residual_driven_consumer)
    partition = Partition(:d => [:d])
    site = spin_ops()
    orbitals = BathOrbitals(
        Float64[], Vector{Vector{ComplexF64}}(), Int[], Int[], Symbol[];
        layout, partition)
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    topology = TreeTopology(:imp, Pair{Symbol,Symbol}[])
    physical = Dict(:imp => site.P)
    hamiltonian = OpSum() +
        Term(0.2, SiteOp(:imp, :N, site.N))
    mounted = AndersonBath(
        bath, topology, physical, Symbol[], Symbol[], hamiltonian)
    interaction = DensityDensityInteraction(zeros(1, 1), layout)
    operator = ttno_from_opsum(
        hamiltonian, topology, physical; hermitian=true)
    lowered = LoweredImpurityHamiltonian(
        mounted, interaction, hamiltonian, operator,
        LegacyTTNOBuilder(), nothing, nothing,
        SymmetryAudit(:certified, (), ()), nothing, (;))
    channel = LocalCorrelator(
        :particle,
        :imp => site.Sp,
        :imp => site.Sm)
    return (; lowered, channel)
end

@testset "ImaginaryTimeRequest forwards Residual Driven Expansion" begin
    fixture = _residual_driven_consumer_fixture()
    expansion = ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        residual_trunc=TruncationScheme(maxdim=2),
        max_add=0,
        max_total_add=0,
        max_edges=0,
        max_rounds=0)
    evolver = ImplicitLogTime(
        scheme=LogBackwardEuler(),
        krylovdim=2,
        maxiter=2,
        tol=1e-8,
        fit_nsweeps=1,
        fit_tol=0.0,
        expansion=expansion)
    beta = 0.2
    taus = [0.0, beta]
    imaginary_time = ImaginaryTimeRequest(
        taus, GraftImpurity.FiniteTemperature(beta);
        evolver,
        thermal_nsteps=1,
        propagation_nsteps=1)

    @test imaginary_time.evolver === evolver
    @test imaginary_time.evolver.expansion === expansion

    result = GraftImpurity._solver_imaginary_time(
        fixture.lowered, imaginary_time, (fixture.channel,))
    raw = result.correlators.particle
    standard_gtau = CorrelatorSeries(
        taus, -raw.values, (; beta, convention=:fermionic_gtau))

    @test raw.convention === :raw_correlator
    @test raw.metadata.coordinate === :tau
    @test raw.metadata.evolver_type === typeof(evolver)
    @test result.trajectory.metadata.evolver_type === typeof(evolver)
    @test raw.z_grid == ComplexF64.(taus)
    @test all(value -> isfinite(real(value)) && isfinite(imag(value)), raw.values)
    @test all(
        value -> isfinite(real(value)) && isfinite(imag(value)),
        standard_gtau.values)
    @test standard_gtau.values == -raw.values
    @test real(first(raw.values)) > 0
    @test real(first(standard_gtau.values)) < 0
end
