using Test

@testset "GraftImpuritySolver" begin
    include("onebody_symmetry.jl")
    include("interactions_lowering.jl")
    include("interaction_ttno.jl")
    include("ttno_builders.jl")
    include("solver.jl")
    include("residual_driven_implicit_consumer.jl")
    include("cayley_mapping.jl")
    include("esprit_tau.jl")
end

@testset "owner-local load graph" begin
    loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
    @test "Graft" ∉ loaded
    @test "GraftImpurity" ∉ loaded
    @test "GraftImpurityValidation" ∉ loaded
end
