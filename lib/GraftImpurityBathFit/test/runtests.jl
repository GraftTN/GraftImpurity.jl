using Test

@testset "GraftImpurityBathFit" begin
    include("bathfit_health_fixtures.jl")
    include("bathfit_health_scalar.jl")
    include("bathfit_health_sensitivity.jl")
    include("bathfit_health_matrix.jl")
    include("bathfit_health_statistics.jl")
    include("bathfit_health_profiles.jl")
    include("bathfit_health_spectral_details.jl")
    include("bathfit_health_integration.jl")
    include("bathfit_health_regressions.jl")
    include("bathfit_runner.jl")
    include("dmft_monitor.jl")
    include("real_pole_kernels.jl")
    include("complex_poles.jl")
    include("coupling_fit.jl")
    include("esprit_tau_bathfit.jl")
    include("realization.jl")
    include("bathfit_report.jl")
    include("preparation.jl")
    include("sparseir_adapter.jl")
end
