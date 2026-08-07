using Test
using Serialization
using GraftFoundation: U1Irrep, SU2Irrep
import GraftImpurityFoundations
using GraftImpurityFoundations: FlavorLayout, Partition,
    AbstractImpurityInteraction, interaction_layout, interaction_identity
using GraftImpurityInteractions: ImpurityOneBody, DensityDensityInteraction
using GraftImpurityBaths: BathOrbitals, DiscreteBath
using GraftImpurityProblems

include("fixtures.jl")
include("problem.jl")
include("manifolds.jl")
include("load_isolation.jl")
