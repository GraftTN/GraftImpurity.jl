using Test
using GraftImpurityFoundations
using GraftImpurityBaths
using GraftImpurityBathFit

@testset "BathFit-only imaginary-time ESPRIT" begin
    @test_throws ArgumentError ESPRITTauKernel(n_poles=0)

    beta = 6.0
    energy = 0.4
    residue = 0.75
    taus = collect(range(0.0, beta; length=33))
    samples = ComplexF64[
        -residue * exp(-tau * energy) / (1 + exp(-beta * energy))
        for tau in taus
    ]
    layout = FlavorLayout(
        [:orbital], Dict(:orbital => :impurity),
        Dict(:impurity => [:orbital]); basis=:bathfit_esprit_tau,
    )
    partition = Partition(:bath => [:orbital])
    input = BathFitInput(
        layout, taus, :bath => samples;
        domain=:imaginary_time, statistics=:fermion,
    )

    expansion = real_pole_bath_fit(
        input,
        ESPRITTauKernel(
            n_poles=1, pole_tolerance=1e-8,
            projection_tolerance=1e-10, fit_tolerance=1e-7,
        ),
        partition,
    )

    @test expansion.kernel === :esprit_tau
    @test expansion.trace.method === :imaginary_time_esprit
    @test only(expansion.poles.poles) ≈ energy atol=1e-7 rtol=1e-7
    @test only(expansion.poles.residues)[1, 1] ≈ residue atol=1e-7 rtol=1e-7
end
