# GraftImpurity.jl

Impurity-solver companion package for
[Graft.jl](https://github.com/GraftTN/Graft.jl). `GraftImpurity.jl` owns impurity
partitions, bath fitting and mounting, solver-facing interfaces, and
postprocessing of correlators and Green's functions; the tree tensor-network kernels is in `Graft.jl`.

The core of GraftImpurity.jl contains three types of features:

1. Various bath fitting algorithms from published/verified and experimental methods to fit the hybridization function of fermions and bosons. Whereas the kernel constructions are independent but share common optimisation algorithms. In the future, Lindblad/HEOM-type complex weight+complex pole modes and/or Lorentzian type modes may be extended, subject to the availability of TTNDO features from Graft.jl, i.e., my personal effort toward the TTNDO.
2. Mapping the effective Hamiltonian through constructing the tree tensor network operator; for T3NS (MT3N) and FTPS, we use star geometry. In the case of Cayley tree or in other cases, star geometry to chain geometry mapping is applied.
3. Providing the post-processing of Green's functions which may be required for DMFT calculations, such as linear predictor and PSD projection through Lorentzian.
## Features

- Hamiltonian bath fitting, realization, and mounting with `Partition`,
  `BlockRealPoles`, `DiscreteBath`, `real_pole_bath_fit`, `realize_bath`, and
  `mount_bath`.
- PES/ADAPOL-style pole fitting with `pes_fit`, `evaluate_poles`, and
  `bath_orbitals`, including scalar NNLS and matrix-valued Hermitian PSD fits.
- SparseIR adapters for GreenFunc target-first `Gf` objects in imaginary time
  and Matsubara frequency; they do not perform real-axis continuation.
- An Anderson-star workflow using `AndersonBath`, two-site DMRG, observables,
  zero-temperature real-time correlators, and opt-in thermal imaginary-time
  correlators.
- Experimental scalar and matrix-valued real-axis Lorentzian fits through
  `LorentzianPSD`, `MatrixLorentzianPSD`, and `lorentzian_fit`.

Thermofield is not a finite-temperature solver or bath-fitting contract in this
package. It is reserved for EDMFT bosonic-bath chain mapping: reorganizing the
emission and absorption sectors before tree layout. Physical finite-temperature
impurity-plus-bath states use Graft's ordinary purification route. This refactor
snapshot therefore does not expose the legacy `ThermofieldRealPoles` or
temperature-driven `fit_bath` interface.

In `pes_fit`, `solver=:sdp` enforces Hermitian PSD residues, while
`solver=:least_squares` is unconstrained. Its optional conic diagnostic reports
the distance to the PSD cone without changing the fit, and `bath_orbitals`
rejects materially indefinite residues. PES fits may contain positive and
negative real poles and are distinct from the typed `BlockRealPoles`
Hamiltonian-bath representation.

The BFGS branch in the `green-jl-counterterms` reference code belongs to the
same direct coupling-fit model and objective family as `CouplingFitKernel`; it
is not an extension of `ESPRITTauKernel`, which is the imaginary-time
ESPRIT route. A corresponding unweighted real-coupling configuration is:

```julia
CouplingFitKernel(
    n_modes=N,
    alpha=0.0,
    components=RealComponents(),
    energy_bounds=(emin, emax),
)
```

This mapping is not a literal port. The reference implementation initializes
randomly and uses `lambda_range` only to initialize energies, leaving them
unconstrained afterwards. `CouplingFitKernel` instead uses deterministic moment
initialization, and `energy_bounds` is a true closed feasible interval enforced
throughout the fit.

The Lorentzian interface accepts real-axis nonnegative or Hermitian PSD data
only. It is experimental and makes no convergence or uniqueness guarantee.
The default minimum width is half the smallest input-grid spacing; set
`minimum_width` to choose another resolution contract. Its full-axis
`1 / omega^2` tails can make higher spectral moments diverge. The current
Anderson workflow supports one partition block and requires an explicit warm
start `psi0`. Real-time series use the zero-temperature DMRG ground state;
imaginary-time series require finite `beta_eff`, a `thermal_evolver`, and
thermal purification. Correlator output is raw: fermionic signs, retarded
assembly, Fourier transforms, and self-energies are left to the caller.

## Local development

Keep the package next to its local dependencies:

```text
~/tmp/
├── GRAFT.jl/
├── GraftImpurity.jl/
└── GreenFunc.jl/
```

Then develop the dependencies and run the test suite:

```julia
using Pkg
Pkg.develop(path="../GRAFT.jl")
Pkg.develop(path="../GreenFunc.jl")
Pkg.test()
```

## References

Background for bath discretization, impurity topologies, and solver design:

- C. Gramsch, K. Balzer, M. Eckstein, and M. Kollar, *Hamiltonian-based
  impurity solver for nonequilibrium dynamical mean-field theory*,
  [Phys. Rev. B 88, 235106 (2013)](https://doi.org/10.1103/PhysRevB.88.235106).
  Matrix-valued Hamiltonian-bath discretization.
- D. Bauernfeind, M. Zingl, R. Triebl, M. Aichhorn, and H. G. Evertz,
  *Fork Tensor-Product States: Efficient Multiorbital Real-Time DMFT Solver*,
  [Phys. Rev. X 7, 031013 (2017)](https://doi.org/10.1103/PhysRevX.7.031013).
  FTPS geometry, star-bath discretization, and residue-factorization convention.
- K. Gunst, F. Verstraete, S. Wouters, Ö. Legeza, and D. Van Neck,
  *T3NS: Three-Legged Tree Tensor Network States*,
  [J. Chem. Theory Comput. 14, 2026 (2018)](https://doi.org/10.1021/acs.jctc.8b00098).
  Three-legged tree representation.
- N.-O. Linden, M. Zingl, C. Hubig, O. Parcollet, and U. Schollwöck,
  *Imaginary-time matrix product state impurity solver in a real material
  calculation: Spin-orbit coupling in Sr₂RuO₄*,
  [Phys. Rev. B 101, 041101(R) (2020)](https://doi.org/10.1103/PhysRevB.101.041101).
  Matrix-valued hybridization and spin-orbit-coupled block structure.
- H. Schnait, D. Bauernfeind, T. Saha-Dasgupta, and M. Aichhorn,
  *Small moments without long-range magnetic ordering in the zero-temperature
  ground state of the double-perovskite iridate Ba₂YIrO₆*,
  [Phys. Rev. B 106, 035132 (2022)](https://doi.org/10.1103/PhysRevB.106.035132)
  ([arXiv:2202.10794](https://arxiv.org/abs/2202.10794)). Off-diagonal and
  spin-orbit-coupled hybridization in fork geometries.
- M. Grundner, P. Westhoff, F. B. Kugler, O. Parcollet, and U. Schollwöck,
  *Complex time evolution in tensor networks and time-dependent Green's
  functions*,
  [Phys. Rev. B 109, 155124 (2024)](https://doi.org/10.1103/PhysRevB.109.155124).
  Complex-time evolution for Green-function calculations.
- X. Cao, E. M. Stoudenmire, and O. Parcollet, *Finite temperature minimal
  entangled typical thermal states impurity solver*,
  [Phys. Rev. B 109, 245113 (2024)](https://doi.org/10.1103/PhysRevB.109.245113)
  ([arXiv:2312.13668](https://arxiv.org/abs/2312.13668)). Direct coupling-space
  Matsubara bath fitting and METTS.
- S. Paeckel, *Spectral decomposition and high-accuracy Greens functions:
  Overcoming the Nyquist-Shannon limit via complex-time Krylov expansion*,
  [arXiv:2411.09680](https://arxiv.org/abs/2411.09680). Complex-time Krylov
  augmentation of real-time correlators.
- B. Zhan, J.-L. Chen, Z. Fan, and T. Xiang, *Tree tensor network impurity
  solver based on Cayley-tree mapping*,
  [Phys. Rev. B 113, 195144 (2026)](https://doi.org/10.1103/ycty-d5f9).
  Cayley-tree mapping for scalar-hybridization bath Hamiltonians.

Related theses:

- D. Bauernfeind, *Fork Tensor Product States: Efficient Multi-Orbital
  Impurity Solver for Dynamical Mean Field Theory*, PhD thesis, Graz
  University of Technology (2018). Detailed FTPS construction.
- S. Mardazad, *Simulating real molecules with tensor network techniques*, PhD
  thesis, LMU Munich (2022). T3N framework.
- M. Grundner, *Tensor Network Impurity Solvers: Simulating Quantum Materials*,
  PhD thesis, LMU Munich (2025). T3N, FT3N, and MT3N impurity topologies.

## License

GraftImpurity.jl is licensed under the [Apache License 2.0](LICENSE).
