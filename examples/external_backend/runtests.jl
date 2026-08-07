using Test
using GraftFoundation: U1Irrep
using GraftImpurityBaths: BathOrbitals, DiscreteBath
import GraftImpurityFoundations
using GraftImpurityFoundations: AbstractImpuritySolver,
    AbstractImpurityWorkspace, AbstractImpuritySolveRequest,
    AbstractImpuritySolveResult, FlavorLayout, Partition, solve!
using GraftImpurityInteractions: DensityDensityInteraction, ImpurityOneBody
using GraftImpurityProblems: AbstractImpurityManifold, ImpurityProblem,
    TargetIrrep, manifold_targets, problem_identity, symmetry_actions,
    validate_manifold

struct ExternalSolver <: AbstractImpuritySolver
    offset::Int
end

mutable struct ExternalWorkspace <: AbstractImpurityWorkspace
    executions::Int
    last_result::Union{Nothing,AbstractImpuritySolveResult}
end

ExternalWorkspace() = ExternalWorkspace(0, nothing)

struct ExternalRequest{M<:AbstractImpurityManifold} <:
        AbstractImpuritySolveRequest
    manifold::M
    value::Int
end

struct ExternalResult <: AbstractImpuritySolveResult
    problem_identity::UInt
    targets::Tuple
    value::Int
end

function GraftImpurityFoundations.solve!(
        workspace::ExternalWorkspace, solver::ExternalSolver,
        problem::ImpurityProblem, request::ExternalRequest)
    validate_manifold(problem, request.manifold)
    result = ExternalResult(
        problem_identity(problem), manifold_targets(request.manifold),
        solver.offset + request.value,
    )
    workspace.executions += 1
    workspace.last_result = result
    return result
end

function external_problem()
    layout = FlavorLayout(
        :d => :impurity;
        site_modes=Dict(:impurity => [:d]), basis=:external_backend,
    )
    partition = Partition(:d => [:d])
    orbitals = BathOrbitals(
        [0.25], [ComplexF64[0.4]], [1], [1], [:d]; layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    h_loc = ImpurityOneBody(zeros(ComplexF64, 1, 1), layout)
    interaction = DensityDensityInteraction(zeros(ComplexF64, 1, 1), layout)
    return ImpurityProblem(bath, h_loc, interaction)
end

@testset "external backend uses only common impurity owners" begin
    problem = external_problem()
    charge = only(symmetry_actions(problem.symmetry))
    manifold = TargetIrrep(charge, U1Irrep(0))
    solver = ExternalSolver(7)
    workspace = ExternalWorkspace()
    result = solve!(workspace, solver, problem, ExternalRequest(manifold, 11))

    @test result.problem_identity == problem_identity(problem)
    @test result.targets == (U1Irrep(0),)
    @test result.value == 18
    @test workspace.executions == 1
    @test workspace.last_result === result

    loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
    @test "GraftImpurity" ∉ loaded
    @test "GraftImpuritySolver" ∉ loaded
    @test "GraftGroundState" ∉ loaded
    @test "GraftEvolution" ∉ loaded
    @test "GraftThermal" ∉ loaded
end
