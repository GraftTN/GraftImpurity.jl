using Test
using LinearAlgebra: I
using GraftImpurityFoundations
using GraftImpurityFoundations: _nnls

struct SyntheticRealPoleKernel <: AbstractRealPoleBathFitKernel end

function GraftImpurityFoundations.real_pole_bath_fit(
        input::Int, ::SyntheticRealPoleKernel, partition::Partition)
    return input, block_names(partition)
end

struct ExternalRequest <: AbstractImpuritySolveRequest
    value::Int
end

struct ExternalResult <: AbstractImpuritySolveResult
    value::Int
end

struct ExternalSolver <: AbstractImpuritySolver
    offset::Int
end

mutable struct ExternalWorkspace <: AbstractImpurityWorkspace
    result::Union{Nothing, ExternalResult}
end

ExternalWorkspace() = ExternalWorkspace(nothing)

function GraftImpurityFoundations.solve!(
        workspace::ExternalWorkspace, solver::ExternalSolver,
        problem::Int, request::ExternalRequest)
    result = ExternalResult(problem + request.value + solver.offset)
    workspace.result = result
    return result
end

function local_matrix(operator)
    array = convert(Array, operator)
    return ndims(array) == 2 ? Matrix{ComplexF64}(array) :
        Matrix{ComplexF64}(selectdim(array, 3, 1))
end

@testset "layout and partition" begin
    layout = FlavorLayout(
        :up => :impurity, :down => :impurity;
        site_modes=Dict(:impurity => [:up, :down]), basis=:spin_orbital,
    )
    equivalent = FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis=:spin_orbital,
    )

    @test layout == equivalent
    @test flavors(layout) == (:up, :down)
    @test flavor_index(layout, :down) == 2
    @test physical_site(layout, :up) === :impurity
    @test site_modes(layout, :impurity) == (:up, :down)
    @test layout_sites(layout) == (:impurity,)
    @test basis_identity(layout) === :spin_orbital
    @test_throws KeyError flavor_index(layout, :missing)
    @test_throws ArgumentError FlavorLayout(
        [:up, :up], Dict(:up => :impurity), Dict(:impurity => [:up]); basis=:bad,
    )

    partition = Partition(:up => [:up], :down => [:down])
    @test validate_partition(partition, layout) === partition
    @test block_names(partition) == (:up, :down)
    @test block_flavors(partition, :down) == (:down,)
    @test block_index(partition, :down) == 2
    @test partition_flavors(partition) == (:up, :down)
    @test hash(partition) == hash(Partition(:up => [:up], :down => [:down]))
    @test_throws ArgumentError Partition(:left => [:up], :right => [:up])
    @test_throws ArgumentError validate_partition(
        Partition(:down => [:down], :up => [:up]), layout,
    )
end

@testset "external impurity solver protocol" begin
    @test AbstractImpuritySolver ===
        GraftImpurityFoundations.AbstractImpuritySolver
    @test AbstractImpuritySolveRequest ===
        GraftImpurityFoundations.AbstractImpuritySolveRequest
    @test AbstractImpuritySolveResult ===
        GraftImpurityFoundations.AbstractImpuritySolveResult
    @test AbstractImpurityWorkspace ===
        GraftImpurityFoundations.AbstractImpurityWorkspace
    @test solve! === GraftImpurityFoundations.solve!
    @test !isdefined(GraftImpurityFoundations, :set_weiss!)
    @test !isdefined(GraftImpurityFoundations, :set_hybridization!)
    @test ExternalSolver <: AbstractImpuritySolver
    @test ExternalWorkspace <: AbstractImpurityWorkspace
    @test ExternalRequest <: AbstractImpuritySolveRequest
    @test ExternalResult <: AbstractImpuritySolveResult

    solver = ExternalSolver(13)
    workspace = ExternalWorkspace()
    result = solve!(workspace, solver, 7, ExternalRequest(11))
    @test result == ExternalResult(31)
    @test workspace.result === result

    loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
    @test "GraftImpuritySolver" ∉ loaded
end

@testset "generic identity and local fermions" begin
    layout = FlavorLayout(
        :up => :impurity, :down => :impurity;
        site_modes=Dict(:impurity => [:up, :down]),
    )
    partition = Partition(:all => [:up, :down])

    @test real_pole_bath_fit === GraftImpurityFoundations.real_pole_bath_fit
    @test real_pole_bath_fit(7, SyntheticRealPoleKernel(), partition) ==
        (7, (:all,))

    operators = FermionSiteOperators(
        layout, :impurity; sector=FermionParitySector(),
    )
    @test operators.modes == (:up, :down)
    @test fermion_sector(operators) isa FermionParitySector
    annihilator = local_matrix(local_annihilator(operators, :up))
    creator = local_matrix(local_creator(operators, :up))
    number = local_matrix(local_number(operators, :up))
    @test annihilator * creator + creator * annihilator ≈
        Matrix{ComplexF64}(I, 4, 4)
    @test creator * annihilator ≈ number
    @test_throws KeyError local_number(operators, :missing)

    number_operators = FermionSiteOperators(
        [:up, :down]; sector=ParticleNumberSector(),
    )
    @test fermion_sector(number_operators) isa ParticleNumberSector
end

@testset "NNLS basics" begin
    identity_matrix = Matrix{Float64}(I, 2, 2)
    coefficients, iterations = _nnls(identity_matrix, [2.0, -1.0])
    @test coefficients ≈ [2.0, 0.0]
    @test iterations == 1
    @test all(>=(0.0), coefficients)
    @test_throws DimensionMismatch _nnls(ones(2, 1), ones(3))
    @test_throws ErrorException _nnls(identity_matrix, [1.0, 1.0]; maxiter=0)
end
