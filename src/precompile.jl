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
    h_loc = ImpurityOneBody(zeros(ComplexF64, 1, 1), layout)
    interaction = DensityDensityInteraction(zeros(ComplexF64, 1, 1), layout)
    symmetry = ImpuritySymmetryDeclaration(layout)
    preparation = HybridizationPreparationInput(input, partition; h_loc)
    policy = ImpurityPreparationPolicy(
        kernel, BathFitCriteria(require_mountable=true),
    )

    PrecompileTools.@compile_workload begin
        outcome = prepare_impurity_problem(
            preparation, interaction, symmetry, policy,
        )
        if outcome isa PreparedImpurityProblem
            problem = outcome.problem
            action = only(symmetry_actions(problem.symmetry))
            target_type = only(category_product(action_identity(action)))
            manifold = TargetIrrep(action, target_type(0))
            validate_manifold(problem, manifold)
            problem_identity(problem)
            manifold_identity(manifold)
        end
    end
end
