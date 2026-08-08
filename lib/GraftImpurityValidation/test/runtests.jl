using Test

using GraftImpurityValidation

@testset "GraftImpurityValidation" begin
    include("matsubara.jl")
    include("finite_mode_action.jl")
    include("kondo_scaling.jl")
    include("m2_metal_bath_gtau.jl")
    include("finite_mode_benchmark.jl")
    include("p4_finite_mode_anderson_holstein.jl")
    include("ctseg_reference.jl")
end

@testset "owner-local load graph" begin
    loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
    @test "Graft" ∉ loaded
    @test "GraftImpurity" ∉ loaded
    @test isempty(intersect(loaded, Set([
        "GraftImpurityFoundations",
        "GraftImpurityInteractions",
        "GraftImpurityBaths",
        "GraftImpurityPoleFits",
        "GraftImpurityBathFit",
        "GraftTTNSSolver",
    ])))
end
