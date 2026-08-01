"""
Impurity-validation records, finite-mode reference kernels, Matsubara
transforms, and Kondo/bath scaling analysis for the GraftImpurity package
family.
"""
module GraftImpurityValidation

using GraftEvolution: CorrelatorSeries
using LinearAlgebra: dot, norm

export MatsubaraSeries, matsubara_transform
export FiniteModeAction, finite_mode_hash, fermionic_frequency,
    bosonic_frequency, hybridization_iw, retarded_interaction_iv
export KondoScalingResult, fit_kondo_scaling
export SemicircularBathReport, semicircular_hybridization,
    gauss_semicircular_bath, discrete_bath_hybridization,
    validate_semicircular_bath, read_bath_csv
export ThermalBenchmarkDatum, FiniteModeBenchmarkCell, BosonCutoffReport,
    assess_boson_cutoff, RepresentationComparison, compare_representations

include("fourier.jl")
include("finite_mode.jl")
include("kondo.jl")
include("benchmarks.jl")

end # module GraftImpurityValidation
