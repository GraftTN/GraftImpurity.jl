using GraftImpurityFoundations: bath_orbitals
using GraftImpurityPoleFits
using Test

@testset "GraftImpurityPoleFits.jl" begin
    include("pes_pole_fitting.jl")
    include("lorentzian_psd.jl")
end
