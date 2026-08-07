function problem_fixture(; basis::Symbol=:problem_fixture)
    layout = FlavorLayout(
        :up => :impurity, :down => :impurity;
        site_modes=Dict(:impurity => [:up, :down]), basis,
    )
    partition = Partition(:all => [:up, :down])
    orbitals = BathOrbitals(
        [-0.5, 0.75],
        [ComplexF64[0.4, 0.1im], ComplexF64[0.2im, 0.3]],
        [1, 2], [1, 1], [:up, :down]; layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    h_loc = ImpurityOneBody(
        ComplexF64[0.2 0.1im; -0.1im -0.3], layout; label=:h_loc,
    )
    interaction = DensityDensityInteraction(
        ComplexF64[0 2; 2 0], layout,
    )
    return ImpurityProblem(bath, h_loc, interaction)
end
