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

"""
Backend-neutral marker for impurity solver implementations.
"""
abstract type AbstractImpuritySolver end

"""
Backend-neutral marker for mutable backend execution workspaces.

A workspace owns reusable build state, warm starts, and last-execution caches;
solver values describe policy and remain separate from this mutable state.
"""
abstract type AbstractImpurityWorkspace end

"""
Backend-neutral marker for typed impurity solve requests.
"""
abstract type AbstractImpuritySolveRequest end

"""
Backend-neutral marker for typed impurity solve results.
"""
abstract type AbstractImpuritySolveResult end

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
    interaction_layout(interaction)

Return the authoritative `FlavorLayout` of an impurity interaction. Concrete
interaction owners implement this accessor so common problem construction does
not inspect implementation fields.
"""
function interaction_layout end

"""
    interaction_identity(interaction)

Return an immutable structural description of every Hamiltonian coefficient,
convention, and basis choice carried by an impurity interaction. Problem and
workspace identities use this value to invalidate cached lowerings. Extension
interactions must implement this protocol; there is deliberately no fallback
to object-identity hashing for mutable values.
"""
function interaction_identity end

"""
    solve!(workspace, solver, problem, request)

Execute a typed impurity solve using an explicit mutable workspace, immutable
backend policy, canonical finite problem, and typed request. Backend packages
provide concrete methods; the common protocol supplies no catch-all fallback.
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
