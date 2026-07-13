# GraftImpurity.jl

Optional impurity-solver companion package for
[Graft.jl](https://github.com/GraftTN/Graft.jl).

`GraftImpurity.jl` owns impurity partitions, Hamiltonian bath fitting and
mounting, and solver-facing interfaces. The migrated bath API includes
`Partition`, `RealPoles`, `MatrixRealPoles`, `ThermofieldRealPoles`,
`ComplexPoles`, `fit_bath`, `factorize_residues`, `matsubara_reconstruct`,
`mount_bath`, and `BosonBath`. Matrix-valued Matsubara fits use grouped
Hermitian PSD residues; `solver=:psd` (or the `:sdp` alias) handles off-diagonal
blocks, while `solver=:nnls` is reserved for scalar or diagonal data.
`fit_ir`/`evaluate_ir` connect GreenFunc target-first `Gf` objects to SparseIR
without dropping complex off-diagonal coefficients; this adapter covers
imaginary time and Matsubara frequency only, not real-axis continuation.
Tensor-network kernels remain in `Graft.jl`; `solve` is currently an
intentional zero-method interface.

The ADAPOL-style PES pole fitter is available separately as `pes_fit`. It uses
a shared matrix AAA approximation for real-pole estimation, optional
`Optim.LBFGS` pole refinement, scalar NNLS, and an explicit real-block
Hermitian PSD residue fit through `JuMP` and `Clarabel`. The result is a
`PESPoleFit`; use `evaluate_poles` for
reconstruction and `bath_orbitals` for the PSD residue factorization. This
general representation permits both positive and negative real poles, so it
is intentionally distinct from the positive-frequency bosonic `RealPoles`
Hamiltonian-bath type.

`solver=:least_squares` is strictly unconstrained and never calls NNLS, SDP,
or another conic solver. Set `conic_diagnostic=:distance` to compute the exact
Frobenius distance of its Hermitian residues to the product PSD cone by
eigendecomposition; the diagnostic does not modify the fitted poles or
residues. `bath_orbitals` validates the current residue values and refuses
materially indefinite least-squares results.

`LorentzianPSD` and `MatrixLorentzianPSD` are separate, highly experimental
representations for real-axis continuous spectra. `lorentzian_fit` directly
fits nonnegative scalar samples or complex Hermitian PSD matrix samples with
shared positive Lorentzian components. The matrix path parameterizes every
residue as `R = B * B'`, so positive widths and factorized residues guarantee
pointwise matrix positivity without an SDP or a global frequency-grid
positivity solve. Matsubara and general non-Hermitian complex-valued data are
not accepted by this interface. Although positivity of the parameterization
is exact, the package currently claims no rigorous convergence, uniqueness,
finite-mixture completeness, or off-grid error theorem for the nonconvex
reconstruction.
The default minimum width is half the smallest input-grid spacing, preventing
an unresolved component from collapsing into a sample-point delta peak; use
`minimum_width` to state a different resolution contract explicitly.
Because every Lorentzian has a full-axis `1 / omega^2` tail, higher spectral
moments may not exist even when the fitted window looks accurate.

For a finite zero-temperature Anderson star, fit a scalar or matrix Matsubara
hybridization with `pes_fit(...; statistics=:fermion, solver=:sdp)`, declare
the residue ordering with a one-block `Partition`, and construct
`AndersonBath(fit, partition; topology, phys)`. PSD residues are factorized
into repeated real-energy bath orbitals with complex impurity coupling
vectors, mounted as graded `fZ2` fermion sites, and lowered to the Hermitian
star Hamiltonian `sum(epsilon * n) + sum(V * d' * c + conj(V) * d * c')`.
`solve(problem, H_loc; psi0, ...)` runs two-site DMRG and evaluates requested
`OpSum` observables. `times` plus `evolver` returns raw particle/hole real-time
series strictly from the zero-temperature DMRG ground state. Imaginary-time
series are a separate opt-in path: `taus` requires a finite `beta_eff` and a
`thermal_evolver`, and uses thermal purification. Both outputs carry
`convention=:raw_correlator`: no fermionic minus sign or retarded assembly is
implicit. This first implementation supports one partition block and an
explicit warm-start `psi0`; it does not manufacture a Fourier transform or
self-energy.

Representative scalar-NNLS, matrix-SDP, evaluation, bath-factorization, and
LBFGS paths are cached with `PrecompileTools` to reduce first-call latency.
A custom `PackageCompiler` sysimage remains a deployment option once the
package family and dependency versions are stable.

## Local development

The three repositories are kept as siblings:

```text
~/tmp/
├── GRAFT.jl/
├── GraftImpurity.jl/
└── GreenFunc.jl/
```

Develop the local dependencies from this environment with:

```julia
using Pkg
Pkg.develop(path="../GRAFT.jl")
Pkg.develop(path="../GreenFunc.jl")
```

Then run the focused B4 bath tests from the package environment:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

For a one-shot setup from a fresh checkout:

```sh
julia --project -e 'using Pkg; Pkg.develop(path="../GRAFT.jl"); Pkg.develop(path="../GreenFunc.jl"); Pkg.test()'
```

## References

Bath discretization, impurity topologies, and the solver design follow these
works. Theses are credited separately from the papers.

Papers:

- C. Gramsch, K. Balzer, M. Eckstein, and M. Kollar, *Hamiltonian-based
  impurity solver for nonequilibrium dynamical mean-field theory*,
  [Phys. Rev. B 88, 235106 (2013)](https://doi.org/10.1103/PhysRevB.88.235106).
  Matrix-valued bath discretization background.
- D. Bauernfeind, M. Zingl, R. Triebl, M. Aichhorn, and H. G. Evertz,
  *Fork Tensor-Product States: Efficient Multiorbital Real-Time DMFT Solver*,
  [Phys. Rev. X 7, 031013 (2017)](https://doi.org/10.1103/PhysRevX.7.031013).
  FTPS geometry, star-basis interval discretization, and the Cholesky
  convention for matrix-valued residues.
- K. Gunst, F. Verstraete, S. Wouters, Ö. Legeza, and D. Van Neck,
  *T3NS: Three-Legged Tree Tensor Network States*,
  [J. Chem. Theory Comput. 14, 2026 (2018)](https://doi.org/10.1021/acs.jctc.8b00098).
  Origin of the three-legged (rank-3 branching) tree representation.
- N.-O. Linden, M. Zingl, C. Hubig, O. Parcollet, and U. Schollwöck,
  *Imaginary-time matrix product state impurity solver in a real material
  calculation: Spin-orbit coupling in Sr₂RuO₄*,
  [Phys. Rev. B 101, 041101(R) (2020)](https://doi.org/10.1103/PhysRevB.101.041101).
  Matrix-valued hybridization and SOC block structure.
- H. Schnait, D. Bauernfeind, T. Saha-Dasgupta, and M. Aichhorn,
  *Small moments without long-range magnetic ordering in the zero-temperature
  ground state of the double-perovskite iridate Ba₂YIrO₆*,
  [Phys. Rev. B 106, 035132 (2022)](https://doi.org/10.1103/PhysRevB.106.035132)
  ([arXiv:2202.10794](https://arxiv.org/abs/2202.10794)). Off-diagonal/SOC
  hybridization terms routed through the existing fork structure.
- M. Grundner, P. Westhoff, F. B. Kugler, O. Parcollet, and U. Schollwöck,
  *Complex time evolution in tensor networks and time-dependent Green's
  functions*,
  [Phys. Rev. B 109, 155124 (2024)](https://doi.org/10.1103/PhysRevB.109.155124).
- X. Cao, E. M. Stoudenmire, and O. Parcollet, *Finite temperature minimal
  entangled typical thermal states impurity solver*,
  [Phys. Rev. B 109, 245113 (2024)](https://doi.org/10.1103/PhysRevB.109.245113)
  ([arXiv:2312.13668](https://arxiv.org/abs/2312.13668)). Direct
  coupling-space Matsubara bath fitting with off-diagonal couplings; METTS.
- S. Paeckel, *Spectral decomposition and high-accuracy Greens functions:
  Overcoming the Nyquist-Shannon limit via complex-time Krylov expansion*,
  [arXiv:2411.09680](https://arxiv.org/abs/2411.09680). Complex-time Krylov
  augmentation of real-time correlator series.
- B. Zhan, J.-L. Chen, Z. Fan, and T. Xiang, *Tree tensor network impurity
  solver based on Cayley-tree mapping*,
  [Phys. Rev. B 113, 195144 (2026)](https://doi.org/10.1103/ycty-d5f9).
  Cayley-tree bath-Hamiltonian mapping (scalar hybridization).

Theses:

- D. Bauernfeind, *Fork Tensor Product States: Efficient Multi-Orbital
  Impurity Solver for Dynamical Mean Field Theory*, PhD thesis, Graz
  University of Technology (2018).
- S. Mardazad, *Simulating real molecules with tensor network techniques*,
  PhD thesis, LMU Munich (2022). T3N framework.
- M. Grundner, *Tensor Network Impurity Solvers: Simulating Quantum
  Materials*, PhD thesis, LMU Munich (2025). Chapter 6 defines the
  T3N/FT3N/MT3N impurity topologies.

## License

GraftImpurity.jl is licensed under the [Apache License 2.0](LICENSE).
