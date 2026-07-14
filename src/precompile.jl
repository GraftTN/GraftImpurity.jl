import PrecompileTools

PrecompileTools.@setup_workload begin
    layout = FlavorLayout(
        :impurity => :impurity_site;
        site_modes=Dict(:impurity_site => [:impurity]),
        basis=:precompile,
    )
    partition = Partition(:hybridization => [:impurity])
    input = BathFitInput(
        layout, [-1.0, 0.0, 1.0],
        :hybridization => ComplexF64[0.0, 0.5, 0.0];
        domain=:real_axis,
        statistics=:fermion,
    )
    plan = DiscretizationPlan(
        :hybridization => BlockDiscretizationPlan(
            (-1.0, 1.0), [(-1.0, 1.0)], 1,
        );
        shared_grid=false,
    )
    kernel = QuadratureKernel(plan)

    PrecompileTools.@compile_workload begin
        expansion = real_pole_bath_fit(input, kernel, partition)
        realize_bath(input, expansion, partition)
    end
end
