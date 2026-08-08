@testset "common-owner load isolation" begin
    active_project = dirname(Base.active_project())
    script = raw"""
        using GraftImpurityProblems
        banned = Set([
            "GraftImpurityBathFit", "GreenFunc", "GraftTTNSSolver",
            "GraftTTNOBuild", "GraftStateDiagram", "GraftGroundState",
            "GraftEvolution", "GraftThermal", "GraftImpurityCI",
        ])
        loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
        isempty(intersect(banned, loaded)) || error(
            "forbidden common-owner modules loaded: $(intersect(banned, loaded))",
        )
    """
    command = `$(Base.julia_cmd()) --startup-file=no --project=$active_project -e $script`
    @test success(pipeline(command; stdout=devnull, stderr=devnull))
    @test :ImpurityOneBody ∉ names(GraftImpurityProblems)
end
