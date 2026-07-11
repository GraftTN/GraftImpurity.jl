# GraftImpurity.jl

Optional impurity-solver companion package for
[Graft.jl](https://github.com/GraftTN/Graft.jl).

`GraftImpurity.jl` will own impurity partitions, bath fitting and mounting,
Green's-function measurement orchestration, and solver-facing interfaces. The
tensor-network kernels remain in `Graft.jl`.

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
