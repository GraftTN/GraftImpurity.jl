# GraftImpurity.jl

Impurity-solver companion package for
[Graft.jl](https://github.com/GraftTN/Graft.jl). `GraftImpurity.jl` owns impurity
partitions, bath fitting and mounting, solver-facing interfaces, and
postprocessing of correlators and Green's functions; the tree tensor-network kernels is in `Graft.jl`.

The core of GraftImpurity.jl contains three types of features:

1. Various bath fitting algorithms from published/verified and experimental methods to fit the hybridization function of fermions and bosons. Whereas the kernel constructions are independent but share common optimisation algorithms. In the future, Lindblad/HEOM-type complex weight+complex pole modes and/or Lorentzian type modes may be extended, subject to the availability of TTNDO features from Graft.jl, i.e., my personal effort toward the TTNDO.
2. Mapping the effective Hamiltonian through constructing the tree tensor network operator; for T3NS (MT3N) and FTPS, we use star geometry. In the case of Cayley tree or in other cases, star geometry to chain geometry mapping is applied.
3. Providing the post-processing of Green's functions which may be required for DMFT calculations, such as linear predictor and PSD projection through Lorentzian.

#### Implemented Features

A quick bath fit example is:

```julia
CouplingFitKernel(
    n_modes=N,                    # Bath modes per independently fitted block.
    alpha=0.0,                    # Matsubara weight exponent; 0 is unweighted.
    components=RealComponents(),  # Restrict all fitted couplings to be real.
    energy_bounds=(emin, emax),   # Closed feasible interval for bath energies.
)
```

#### Known Limitations
The experimental Lorentzian interface computes a PSD-constrained real-axis
approximation of finite scalar or Hermitian matrix-valued spectral-density
samples; the input need not already be PSD. The returned finite Lorentzian
mixture is nonnegative or Hermitian PSD at every real frequency by construction.
Here “projection” means an initialization-dependent, unweighted least-squares
fit into the chosen finite Lorentzian family—not pointwise clipping or a
guaranteed global nearest-cone projection. It does not accept raw retarded or
Matsubara Green functions and makes no convergence or uniqueness guarantee.
The default minimum width is half the smallest input-grid spacing; its full-axis
`1 / omega^2` tails can make higher spectral moments diverge.

## Local development

Keep the package next to its local dependencies:

```text
~/tmp/
├── Graft.jl/
├── GraftImpurity.jl/
└── GreenFunc.jl/
```

Then develop the dependencies and run the test suite:

```julia
using Pkg
Pkg.develop(path="../Graft.jl")
Pkg.develop(path="../GreenFunc.jl")
Pkg.test()
```

## Algorithmic References and Provenance

References are grouped by the GraftImpurity functionality they inform. Each
entry identifies its relation to the package; citing a method does not imply
that every solver or physical setting in the paper is implemented.

### Bath Discretization and Impurity Geometries

1. **Matrix-valued Hamiltonian bath** — *implemented; algorithmic basis*

   C. Gramsch, K. Balzer, M. Eckstein, and M. Kollar, “Hamiltonian-based impurity solver for nonequilibrium dynamical mean-field theory,” *Physical Review B* **88**, 235106 (2013).
   [DOI](https://doi.org/10.1103/PhysRevB.88.235106)

   **Provenance:** Basis for matrix-valued Hamiltonian-bath discretization.

2. **FTPS geometry** — *implemented; algorithmic basis*

   D. Bauernfeind, M. Zingl, R. Triebl, M. Aichhorn, and H. G. Evertz, “Fork Tensor-Product States: Efficient Multiorbital Real-Time DMFT Solver,” *Physical Review X* **7**, 031013 (2017).
   [DOI](https://doi.org/10.1103/PhysRevX.7.031013)

   **Provenance:** Basis for FTPS geometry, star-bath discretization, and the residue-factorization convention.

3. **T3NS geometry** — *implemented; algorithmic basis*

   K. Gunst, F. Verstraete, S. Wouters, Ö. Legeza, and D. Van Neck, “T3NS: Three-Legged Tree Tensor Network States,” *Journal of Chemical Theory and Computation* **14**, 2026 (2018).
   [DOI](https://doi.org/10.1021/acs.jctc.8b00098)

   **Provenance:** Basis for the three-legged tree representation.

4. **Cayley-tree bath mapping** — *implemented; algorithmic basis*

   B. Zhan, J.-L. Chen, Z. Fan, and T. Xiang, “Tree tensor network impurity solver based on Cayley-tree mapping,” *Physical Review B* **113**, 195144 (2026).
   [DOI](https://doi.org/10.1103/ycty-d5f9)

   **Provenance:** Basis for scalar Cayley-tree mapping; GraftImpurity extends the realization to number-conserving block baths.

### Spin-Orbit Coupling and Matrix Structure

1. **SOC-aware block hybridization** — *implemented; validation reference*

   N.-O. Linden, M. Zingl, C. Hubig, O. Parcollet, and U. Schollwöck, “Imaginary-time matrix product state impurity solver in a real material calculation: Spin-orbit coupling in Sr₂RuO₄,” *Physical Review B* **101**, 041101(R) (2020).
   [DOI](https://doi.org/10.1103/PhysRevB.101.041101)

   **Provenance:** Reference for matrix-valued hybridization and spin-orbit-coupled block structure.

2. **SOC hybridization in fork geometries** — *validation reference*

   H. Schnait, D. Bauernfeind, T. Saha-Dasgupta, and M. Aichhorn, “Small moments without long-range magnetic ordering in the zero-temperature ground state of the double-perovskite iridate Ba₂YIrO₆,” *Physical Review B* **106**, 035132 (2022).
   [DOI](https://doi.org/10.1103/PhysRevB.106.035132) ·
   [arXiv](https://arxiv.org/abs/2202.10794)

   **Provenance:** Reference for off-diagonal, spin-orbit-coupled hybridization in fork geometries.

3. **SOC Hamiltonian in TTNS/TTNO** — *implemented; validation reference*

   X. Cao, Y. Lu, P. Hansmann, and M. W. Haverkort, “Tree tensor-network real-time multiorbital impurity solver: Spin-orbit coupling and correlation functions in Sr₂RuO₄,” *Physical Review B* **104**, 115119 (2021).
   [DOI](https://doi.org/10.1103/PhysRevB.104.115119) ·
   [arXiv](https://arxiv.org/abs/2103.05545)

   **Provenance:** Validation reference for representing spin-orbit-coupled multiorbital impurity Hamiltonians and their dynamics with TTNS/TTNO.

### Time Evolution, Bath Fitting, and Thermal Methods

1. **Complex-time Green functions** — *design reference*

   M. Grundner, P. Westhoff, F. B. Kugler, O. Parcollet, and U. Schollwöck, “Complex time evolution in tensor networks and time-dependent Green's functions,” *Physical Review B* **109**, 155124 (2024).
   [DOI](https://doi.org/10.1103/PhysRevB.109.155124)

   **Provenance:** Design reference for complex-time evolution in Green-function calculations.

2. **Direct coupling-space bath fitting**

   X. Cao, E. M. Stoudenmire, and O. Parcollet, “Finite temperature minimal entangled typical thermal states impurity solver,” *Physical Review B* **109**, 245113 (2024).
   [DOI](https://doi.org/10.1103/PhysRevB.109.245113) ·
   [arXiv](https://arxiv.org/abs/2312.13668)

   **Provenance:** Basis for direct coupling-space Matsubara bath fitting.

3. **ESPRIT imaginary-time bath fitting** — *implemented; validation reference*

   Y. Yu, G. Harsha, L. Zhang, A. Jażdżewska, D. Zgid, X. Dong, and E. Gull, “Hybrid Hamiltonian-diagrammatic quantum impurity solver,” arXiv:2606.11095
   (2026).
   [arXiv](https://arxiv.org/abs/2606.11095) ·
   [Supplement](https://arxiv.org/src/2606.11095v1/anc/supplement.pdf)

   **Provenance:** Matrix-valued imaginary-time ESPRIT construction.

   **Note:** We only use this bath fitting method, we still solve the Hamiltonian not hybrid.

4. **Complex-time Krylov augmentation** — *design reference*

   S. Paeckel, “Spectral decomposition and high-accuracy Greens functions: Overcoming the Nyquist-Shannon limit via complex-time Krylov expansion,”
   arXiv:2411.09680 (2024).
   [arXiv](https://arxiv.org/abs/2411.09680)

   **Provenance:** Design reference for complex-time Krylov augmentation of real-time correlators.

### Theses and Implementation Guides

1. **FTPS construction** — *implementation guide*

   D. Bauernfeind, *Fork Tensor Product States: Efficient Multi-Orbital Impurity Solver for Dynamical Mean Field Theory*, PhD thesis, Graz University of Technology (2018).

   **Provenance:** Detailed guide to the FTPS construction.

2. **T3N framework** — *implementation guide*

   S. Mardazad, *Simulating real molecules with tensor network techniques*, PhD thesis, LMU Munich (2022).

   **Provenance:** Detailed guide to the T3N framework.

3. **T3N-family impurity topologies** — *implementation guide*

   M. Grundner, *Tensor Network Impurity Solvers: Simulating Quantum Materials*,
   PhD thesis, LMU Munich (2025).

   **Provenance:** Detailed guide to T3N, FT3N, and MT3N impurity topologies.

## License

GraftImpurity.jl is licensed under the [Apache License 2.0](LICENSE).
