@testset "M2 finite-mode action" begin
    action = FiniteModeAction(
        beta=5.0,
        mu=0.3,
        static_interaction=2.0,
        n0=1.0,
        bath_energies=[-0.7, 0.4],
        bath_couplings=ComplexF64[0.25 + 0.1im, -0.2im],
        boson_frequencies=[0.6, 1.1],
        boson_couplings=[0.3, -0.2],
        orbital_convention=:spinful,
        density_convention=:shifted)

    delta = hybridization_iw(action, -2:2)
    @test delta.frequencies ==
        [(2n + 1) * pi / action.beta for n in -2:2]
    @test delta.values ≈ [
        sum(abs2(V) / (im * omega - epsilon)
            for (epsilon, V) in zip(
                action.bath_energies, action.bath_couplings))
        for omega in delta.frequencies
    ]

    interaction = retarded_interaction_iv(action, 0:3)
    @test interaction.frequencies ==
        [2n * pi / action.beta for n in 0:3]
    @test interaction.values ≈ [
        -sum(2g^2 * omega0 / (nu^2 + omega0^2)
             for (omega0, g) in zip(
                 action.boson_frequencies, action.boson_couplings))
        for nu in interaction.frequencies
    ]
    @test finite_mode_hash(action) == finite_mode_hash(action)
    changed = FiniteModeAction(
        beta=action.beta,
        mu=action.mu,
        static_interaction=action.static_interaction,
        n0=action.n0,
        bath_energies=action.bath_energies,
        bath_couplings=action.bath_couplings,
        boson_frequencies=action.boson_frequencies,
        boson_couplings=[0.3, -0.21],
        orbital_convention=:spinful,
        density_convention=:shifted)
    @test finite_mode_hash(action) != finite_mode_hash(changed)

    @test_throws ArgumentError FiniteModeAction(
        beta=0, bath_energies=[], bath_couplings=[])
    @test_throws ArgumentError FiniteModeAction(
        beta=1, bath_energies=[0.0], bath_couplings=[])
    @test_throws ArgumentError FiniteModeAction(
        beta=1, boson_frequencies=[0.0], boson_couplings=[1.0])
end
