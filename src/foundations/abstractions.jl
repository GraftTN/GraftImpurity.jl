# Stable semantic boundaries. Concrete algorithms are deliberately introduced
# in their assigned milestones rather than by forwarding old interfaces.

abstract type AbstractRealPoleBathFitKernel end
abstract type AbstractBathParametrization end
abstract type AbstractBCFParametrization <: AbstractBathParametrization end
abstract type AbstractHamiltonianBath <: AbstractBathParametrization end
abstract type AbstractBathMappingKernel end
abstract type AbstractImpurityTopologyPlan end
abstract type AbstractMountedBath end
abstract type AbstractImpurityInteraction end
abstract type AbstractImpuritySolver end

"""
    real_pole_bath_fit(input, kernel, partition)

Fit a real-pole expansion through an executable bath-fit kernel.
"""
function real_pole_bath_fit end

"""
    fit_complex_bcf(input, kernel, partition)

Fit a typed time-domain bath-correlation-function exponential sum. Concrete
methods return BCF data only; they never create a Hamiltonian bath.
"""
function fit_complex_bcf end

"""
    evaluate_bcf(poles, times, block)

Evaluate a typed complex BCF exponential sum for one named partition block.
"""
function evaluate_bcf end

"""
    realize_quasi_lindblad(poles; kwargs...)

TODO(M5+/CG-005) — BCF-preserving quasi-Lindblad realization requires the
missing core Liouvillian/TTNDO lowering contract.
"""
# TODO(M5+/CG-005): implement only after core Liouvillian/TTNDO semantics land.
function realize_quasi_lindblad end

"""
    realize_coupled_lindblad(poles; kwargs...)

TODO(M5+/CG-005) — coupled-Lindblad realization requires the missing core
Liouvillian/TTNDO lowering contract.
"""
# TODO(M5+/CG-005): implement only after core Liouvillian/TTNDO semantics land.
function realize_coupled_lindblad end

"""
    realize_bath(input, expansion, partition)

Validate real-pole Hamiltonian realizability and form a canonical DiscreteBath
or typed non-mountable result with a concrete BathFitReport.
"""
function realize_bath end

"""
    mount_bath(topology, bath; kwargs...)

Mount a canonical Hamiltonian bath onto an impurity topology while preserving
declared flavor ownership.  The fermionic `DiscreteBath` route is implemented
in the M5 topology/mounting layer; bosonic values require an explicit local
cutoff/operator convention.
"""
function mount_bath end

"""
    map_bath(kernel, bath)

Apply an explicit bath-Hamiltonian basis-mapping kernel.

M5 implements the typed `ScalarCayley` route. The full-matrix `BlockCayley`
route remains an explicit M5b extension and no mapping infers ownership from
coupling magnitude.
"""
function map_bath end

"""
    impurity_topology(plan, partition, bath)

Build a topology from an impurity topology plan and explicit bath ownership.

M5 implements the `T3NS` and `FTPS` `DiscreteBath` methods.  Other geometry
plans remain explicit extension points rather than inferred aliases.
"""
function impurity_topology end

"""
    lower_interaction(interaction, ops, sector_spec)

Lower a typed impurity interaction into Graft symbolic operators.

M6 provides concrete layout-owned interaction lowering through
`ImpurityOperators`; later interaction families extend this semantic boundary
without reviving a bare-`OpSum` solver interface.
"""
function lower_interaction end

"""
    audit_partition(state, partition)

Audit a declared named partition against a converged impurity state.

TODO(M5) — no methods yet.
"""
# TODO(M5): cross-block entanglement/MI audit is topology-stage work.
function audit_partition end

"""
    factorize_residues(expansion; kwargs...)

Factor validated PSD residues into canonical BathOrbitals with explicit
within-block ownership order. `realize_bath` retains the associated typed
factorization diagnostics in its BathFitReport.
"""
function factorize_residues end

"""
    reconstruct_hybridization(bath, mesh; broadening=nothing)

Reconstruct a hybridization from a canonical discrete bath on a typed
`BathFitInput`, `GreenFunc.Gf`, or `GreenFunc.BlockGf` mesh/template. Real-axis
reconstruction requires an explicit positive broadening; Matsubara
reconstruction rejects nonzero broadening.
"""
function reconstruct_hybridization end

"""
    audit_bathfit(report, criteria)

Evaluate explicit bath-fit acceptance criteria.
"""
function audit_bathfit end

"""
    audit_symmetry(hamiltonian, sector_spec)

Audit symmetry from the complete lowered impurity Hamiltonian.

M6 supplies explicit abelian-generator and non-abelian-unsupported audit
methods. Candidate generators are typed input; no spin or angular-momentum
sector is inferred from a flavor label.
"""
function audit_symmetry end

"""
    set_weiss!(solver, G0_iw; h_loc0)

Set a mutually exclusive Weiss-field input on a stateful impurity solver.

The implemented M6 Solver conversion requires an explicit layout-owned
`h_loc0`; a Weiss propagator alone cannot determine a hybridization under an
arbitrary chemical-potential convention.
"""
function set_weiss! end

"""
    set_hybridization!(solver, Delta; h_loc0)

Set a mutually exclusive hybridization-plus-one-body input on a stateful
impurity solver.
"""
function set_hybridization! end

"""
    solve!(solver, interaction, request; initial_state=nothing)

Execute typed impurity solving and record the result on the solver.
"""
function solve! end

"""
    bath_layout(bath)

Return the FlavorLayout carried by a Hamiltonian bath.

TODO(M2 concrete bath methods are defined below) — no generic methods.
"""
# TODO(M2): concrete DiscreteBath query method is defined below.
function bath_layout end

"""
    bath_partition(bath)

Return the named Partition carried by a Hamiltonian bath.

TODO(M2 concrete bath methods are defined below) — no generic methods.
"""
# TODO(M2): concrete DiscreteBath query method is defined below.
function bath_partition end

"""
    bath_orbitals(bath)

Return canonical BathOrbitals for a Hamiltonian bath.

TODO(M2 concrete bath methods are defined below) — no generic methods.
"""
# TODO(M2): concrete DiscreteBath query method is defined below.
function bath_orbitals end

"""
    bath_statistics(bath)

Return the particle statistics declared by a canonical Hamiltonian bath.

TODO(M2 concrete bath method is defined below) — no generic methods.
"""
# TODO(M2): concrete DiscreteBath query method is defined below.
function bath_statistics end
