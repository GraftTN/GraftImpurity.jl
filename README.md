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
