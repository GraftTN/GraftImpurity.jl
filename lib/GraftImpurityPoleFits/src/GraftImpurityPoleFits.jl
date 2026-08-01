"""
Pole-estimation, semidefinite residue fitting, and experimental Lorentzian
spectral models for the GraftImpurity package family.
"""
module GraftImpurityPoleFits

using LinearAlgebra: Diagonal, Hermitian, eigen, eigvals, norm, svd, tr
import Clarabel
import GraftImpurityFoundations: bath_orbitals, _nnls
import JuMP
import Optim

export PESPoleFit, pes_fit, evaluate_poles
export LorentzianPSD, MatrixLorentzianPSD, lorentzian_fit, spectral_density,
    complex_poles

include("pes_pole_fitting.jl")
include("lorentzian_psd.jl")

end # module GraftImpurityPoleFits
