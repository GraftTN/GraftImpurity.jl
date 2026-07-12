using Test
using GraftImpurity

const TEST_VERBOSE = lowercase(get(ENV, "GRAFT_TEST_VERBOSE", "false")) in
    ("1", "true", "yes", "on")

@testset "GraftImpurity.jl" begin
    include("b4_bath_fitting_mounting.jl")
    include("bath_matrix_fitting.jl")
    include("pes_pole_fitting.jl")
    include("lorentzian_psd.jl")
    include("sparseir_adapter.jl")
    include("b4_finite_temperature_bath.jl")
    include("b4_bath_tdvp_e2e.jl")
end
