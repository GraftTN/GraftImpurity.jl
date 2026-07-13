using Test
using GraftImpurity

const TEST_VERBOSE = lowercase(get(ENV, "GRAFT_TEST_VERBOSE", "false")) in
    ("1", "true", "yes", "on")

@testset "GraftImpurity.jl" begin
    include(joinpath(@__DIR__, "foundations.jl"))
    include(joinpath(@__DIR__, "realization.jl"))
    include(joinpath(@__DIR__, "bathfit_report.jl"))
    include(joinpath(@__DIR__, "real_pole_kernels.jl"))
    include(joinpath(@__DIR__, "complex_poles.jl"))
    include(joinpath(@__DIR__, "coupling_fit.jl"))
    include(joinpath(@__DIR__, "pes_pole_fitting.jl"))
    include(joinpath(@__DIR__, "lorentzian_psd.jl"))
    include(joinpath(@__DIR__, "sparseir_adapter.jl"))
end
