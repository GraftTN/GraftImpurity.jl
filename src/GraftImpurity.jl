"""
GraftImpurity: optional impurity-solver companion package for Graft.jl.
Owns boson bath fitting/mounting and finite zero-temperature Anderson stars
lowered from fermionic PSD real-pole fits. Finite-temperature solver glue
remains a later milestone.

Never referenced by any lower layer (§9.10). Owns *no* private geometry code
(§0.1): geometry builders emit plain `Trees.TreeTopology`.
"""
module GraftImpurity

using LinearAlgebra: Diagonal, Hermitian, diag, eigen, eigvals, norm, opnorm,
    svd, tr
using Graft.Trees: TreeTopology, mount_chain, nodeindex
using Graft.Symbolic: OpSum, SiteOp, Term, boson_modes, BosonCoupling,
    fermion_ops_z2
using Graft: Purified, TTNS, TruncationScheme, correlator_series, dmrg2!,
    expect, normalize!, purification_problem, thermal_correlator, thermalize,
    topology, ttno_from_opsum

export Partition, audit_partition, BathParametrization, RealPoles,
    MatrixRealPoles, ThermofieldRealPoles, ComplexPoles, couplings,
    factorize_residues, matsubara_reconstruct, mount_bath, fit_bath,
    BosonBath, solve,
    IRCoefficients, fit_ir, evaluate_ir, to_imtime_ir, to_imfreq_ir,
    PESPoleFit, pes_fit, evaluate_poles, bath_orbitals,
    LorentzianPSD, MatrixLorentzianPSD, lorentzian_fit, spectral_density,
    complex_poles,
    AndersonRealPoles, AndersonBath

# ---------------------------------------------------------------------------
# §6.2 Partition: a *user declaration* on the impurity orbitals; H_bath never
# partitions independently — it inherits the block structure through Δ(ω).
# Dependency chain (one-way):
#   physics priors → Partition P → block structure of Δ → blockwise bath fit
#   → modes mounted on the block's branch.
# ---------------------------------------------------------------------------

"""
    Partition(blocks::Vector{Vector{Symbol}})

Immutable grouping of impurity orbitals into blocks (eg/t2g, j_eff, d+ligand…).
First-class *input*: fixed partition ⇒ fixed topology ⇒ warm starts across the
self-consistency loop stay valid (`==`/`hash` are value-based, §9.4/§10.9).
Automatic partitioning is deliberately **not** offered — `audit_partition` is
the diagnostic ("人分区、库验收").
"""
struct Partition{B<:Tuple}
    _blocks::B
    function Partition(blocks::Vector{Vector{Symbol}})
        isempty(blocks) && throw(ArgumentError("Partition needs at least one block"))
        any(isempty, blocks) && throw(ArgumentError("Partition blocks may not be empty"))
        owned = Tuple(Tuple(block) for block in blocks)
        orbs = reduce(vcat, owned; init=Symbol[])
        allunique(orbs) || throw(ArgumentError("orbitals appear in more than one block"))
        return new{typeof(owned)}(owned)
    end
end

# Preserve the public vector-of-vectors view while keeping the owned storage
# immutable, so a Partition remains safe as fixed-topology/hash key material.
function Base.getproperty(p::Partition, name::Symbol)
    name === :blocks && return [collect(block) for block in getfield(p, :_blocks)]
    return getfield(p, name)
end
Base.propertynames(::Partition, private::Bool=false) =
    private ? (:_blocks, :blocks) : (:blocks,)
Base.:(==)(a::Partition, b::Partition) =
    getfield(a, :_blocks) == getfield(b, :_blocks)
Base.hash(p::Partition, h::UInt) =
    hash(getfield(p, :_blocks), hash(:Partition, h))

# TODO(M5): cross-block vs in-block mutual-information audit; warn when
# cross-block MI ≳ in-block MI ("the partition may be cut wrong"). Rare-event
# handling stays manual: re-partition + loop restart (§6.2).
"""
    audit_partition(ψ_converged, P::Partition) -> report

Partition diagnostic ("人分区、库验收"). TODO(M5) — no methods yet.
"""
function audit_partition end

# ---------------------------------------------------------------------------
# §6.3 bath discretization / hybridization fitting
# ---------------------------------------------------------------------------

abstract type BathParametrization end

"""
    RealPoles(poles, residues, blocks, block_ranges, diagnostics)

Real-pole bath parametrization. For bosons, `poles[k] = ω_k > 0` and
`residues[k] = g_k^2 >= 0`, so the mode coupling is `g_k = sqrt(residues[k])`.
`block_ranges[i]` indexes the modes fitted for `blocks[i]`. Diagnostics are a
typed `NamedTuple` produced by [`fit_bath`](@ref).
"""
struct RealPoles{D<:NamedTuple} <: BathParametrization
    poles::Vector{Float64}
    residues::Vector{Float64}
    blocks::Vector{Vector{Symbol}}
    block_ranges::Vector{UnitRange{Int}}
    diagnostics::D

    function RealPoles(poles::AbstractVector{<:Real}, residues::AbstractVector{<:Real},
                       blocks::Vector{Vector{Symbol}},
                       block_ranges::Vector{UnitRange{Int}},
                       diagnostics::D) where {D<:NamedTuple}
        length(poles) == length(residues) ||
            throw(ArgumentError("RealPoles needs one residue per pole"))
        length(blocks) == length(block_ranges) ||
            throw(ArgumentError("RealPoles needs one mode range per partition block"))
        p = Float64.(poles)
        r = Float64.(residues)
        all(isfinite, p) && all(x -> x > 0, p) ||
            throw(ArgumentError("RealPoles poles must be finite and positive"))
        all(isfinite, r) && all(x -> x >= 0, r) ||
            throw(ArgumentError("RealPoles residues must be finite and nonnegative"))
        expected = _contiguous_ranges(block_ranges)
        expected == block_ranges ||
            throw(ArgumentError("RealPoles block ranges must be contiguous and ordered"))
        sum(length, block_ranges; init=0) == length(p) ||
            throw(ArgumentError("RealPoles block ranges must cover every pole exactly"))
        return new{D}(p, r, deepcopy(blocks), copy(block_ranges), diagnostics)
    end
end
Base.length(b::RealPoles) = length(b.poles)
couplings(b::RealPoles) = sqrt.(b.residues)

"""
    MatrixRealPoles(poles, residues, blocks, block_ranges, diagnostics)

Grouped real-pole bath parametrization for matrix-valued Matsubara data.
Each `residues[k]` is a Hermitian positive-semidefinite matrix local to the
partition block owning pole `k`. A rank-`r` residue represents `r` bath modes
at the same energy; use [`factorize_residues`](@ref) to obtain those repeated
energies and their coupling vectors.
"""
struct MatrixRealPoles{D<:NamedTuple} <: BathParametrization
    poles::Vector{Float64}
    residues::Vector{Matrix{ComplexF64}}
    blocks::Vector{Vector{Symbol}}
    block_ranges::Vector{UnitRange{Int}}
    diagnostics::D

    function MatrixRealPoles(poles::AbstractVector{<:Real}, residues,
                             blocks::Vector{Vector{Symbol}},
                             block_ranges::Vector{UnitRange{Int}},
                             diagnostics::D) where {D<:NamedTuple}
        length(poles) == length(residues) ||
            throw(ArgumentError("MatrixRealPoles needs one residue matrix per pole"))
        length(blocks) == length(block_ranges) ||
            throw(ArgumentError("MatrixRealPoles needs one mode range per partition block"))
        p = Float64.(poles)
        all(isfinite, p) && all(>(0), p) ||
            throw(ArgumentError("MatrixRealPoles poles must be finite and positive"))
        expected = _contiguous_ranges(block_ranges)
        expected == block_ranges ||
            throw(ArgumentError("MatrixRealPoles block ranges must be contiguous and ordered"))
        sum(length, block_ranges; init=0) == length(p) ||
            throw(ArgumentError("MatrixRealPoles block ranges must cover every pole exactly"))
        mats = Matrix{ComplexF64}[]
        for (bi, r) in enumerate(block_ranges)
            d = length(blocks[bi])
            d > 0 || throw(ArgumentError("MatrixRealPoles blocks may not be empty"))
            for k in r
                R = Matrix{ComplexF64}(residues[k])
                size(R) == (d, d) ||
                    throw(ArgumentError("residue $k must be a $d×$d matrix for block $bi"))
                push!(mats, _validated_psd_residue(R, k))
            end
        end
        return new{D}(p, mats, deepcopy(blocks), copy(block_ranges), diagnostics)
    end
end
Base.length(b::MatrixRealPoles) = length(b.poles)

"""
    factorize_residues(bath::MatrixRealPoles; atol=0, rtol=sqrt(eps()))

Factor each PSD residue as `Rₖ = sum(v * v')`. The returned `energies` repeats
`bath.poles[k]` once per retained factor column. `factors`, `pole_indices`, and
`block_indices` use the same flattened mode ordering.
"""
function factorize_residues(bath::MatrixRealPoles; atol::Real=0.0,
                            rtol::Real=sqrt(eps(Float64)))
    atol >= 0 || throw(ArgumentError("factorization atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("factorization rtol must be nonnegative"))
    energies = Float64[]
    factors = Vector{ComplexF64}[]
    pole_indices = Int[]
    block_indices = Int[]
    for (bi, r) in enumerate(bath.block_ranges), k in r
        F = eigen(Hermitian(bath.residues[k]))
        scale = isempty(F.values) ? 0.0 : maximum(F.values)
        cutoff = max(Float64(atol), Float64(rtol) * scale)
        for j in eachindex(F.values)
            λ = F.values[j]
            λ > cutoff || continue
            push!(energies, bath.poles[k])
            push!(factors, sqrt(λ) .* ComplexF64.(F.vectors[:, j]))
            push!(pole_indices, k)
            push!(block_indices, bi)
        end
    end
    return (; energies, factors, pole_indices, block_indices)
end

"""
    ThermofieldRealPoles(source, emission, absorption; beta, temperature, diagnostics)

Finite-temperature Hamiltonian bath parametrization for thermofield stars.
`source` is the real-pole fit to the retarded/Matsubara source object, while
`emission` and `absorption` are positive real-pole baths with bosonic thermal
weights `(n_B(ω)+1)` and `n_B(ω)`. The Hamiltonian realization is two stars in a
vacuum product state; this is deliberately separate from `ComplexPoles` BCF
pseudomodes.
"""
struct ThermofieldRealPoles{S<:RealPoles,E<:RealPoles,A<:RealPoles,D<:NamedTuple} <: BathParametrization
    source::S
    emission::E
    absorption::A
    blocks::Vector{Vector{Symbol}}
    beta::Float64
    temperature::Float64
    statistics::Symbol
    diagnostics::D

    function ThermofieldRealPoles(source::S, emission::E, absorption::A;
                                  beta::Real, temperature::Real,
                                  statistics::Symbol=:boson,
                                  diagnostics::D) where {S<:RealPoles,E<:RealPoles,A<:RealPoles,D<:NamedTuple}
        statistics == :boson ||
            throw(ArgumentError("ThermofieldRealPoles currently supports statistics=:boson"))
        source.blocks == emission.blocks ||
            throw(ArgumentError("thermofield emission blocks must match the source bath"))
        source.blocks == absorption.blocks ||
            throw(ArgumentError("thermofield absorption blocks must match the source bath"))
        source.block_ranges == emission.block_ranges ||
            throw(ArgumentError("thermofield emission block ranges must match the source bath"))
        source.block_ranges == absorption.block_ranges ||
            throw(ArgumentError("thermofield absorption block ranges must match the source bath"))
        source.poles == emission.poles ||
            throw(ArgumentError("thermofield emission poles must match the source bath"))
        source.poles == absorption.poles ||
            throw(ArgumentError("thermofield absorption poles must match the source bath"))
        β = Float64(beta)
        T = Float64(temperature)
        isfinite(β) && β > 0 ||
            throw(ArgumentError("thermofield beta must be finite and positive"))
        isfinite(T) && T > 0 ||
            throw(ArgumentError("thermofield temperature must be finite and positive"))
        return new{S,E,A,D}(source, emission, absorption, deepcopy(source.blocks),
                            β, T, statistics, diagnostics)
    end
end
Base.length(b::ThermofieldRealPoles) = length(b.emission) + length(b.absorption)
couplings(b::ThermofieldRealPoles) = (;
    source = couplings(b.source),
    emission = couplings(b.emission),
    absorption = couplings(b.absorption),
)

"""
    matsubara_reconstruct(bath::RealPoles, frequencies; block=1, kind=:boson)

Project a real-pole Hamiltonian bath back to the supplied Matsubara grid. For
the forwarded boson path, `kind=:boson` uses
`sum_k 2ω_k g_k^2 / (ν_n^2 + ω_k^2)` for each partition block.
"""
function matsubara_reconstruct(bath::RealPoles, frequencies; block::Integer=1,
                               kind::Symbol=:boson)
    kind == :boson ||
        throw(ArgumentError("matsubara_reconstruct currently supports kind=:boson"))
    1 <= block <= length(bath.block_ranges) ||
        throw(ArgumentError("block index out of range"))
    νs = _matsubara_frequencies(frequencies)
    r = bath.block_ranges[block]
    A = _boson_matsubara_kernel(νs, bath.poles[r])
    return A * bath.residues[r]
end

"""
    matsubara_reconstruct(bath::MatrixRealPoles, frequencies; block=1, kind=:boson)

Reconstruct grouped matrix-valued Matsubara samples. The output layout is
`(frequency, row, column)`.
"""
function matsubara_reconstruct(bath::MatrixRealPoles, frequencies;
                               block::Integer=1, kind::Symbol=:boson)
    kind == :boson ||
        throw(ArgumentError("matsubara_reconstruct currently supports kind=:boson"))
    1 <= block <= length(bath.block_ranges) ||
        throw(ArgumentError("block index out of range"))
    νs = _matsubara_frequencies(frequencies)
    r = bath.block_ranges[block]
    A = _boson_matsubara_kernel(νs, bath.poles[r])
    d = length(bath.blocks[block])
    out = zeros(ComplexF64, length(νs), d, d)
    for n in eachindex(νs), (j, k) in enumerate(r)
        @views out[n, :, :] .+= A[n, j] .* bath.residues[k]
    end
    return out
end

"""
    matsubara_reconstruct(bath::ThermofieldRealPoles, frequencies; block=1,
                          kind=:boson, channel=:source)

Project a finite-temperature thermofield bath to a Matsubara grid. `channel` may
be `:source`/`:retarded` for the original Hamiltonian source object,
`:emission`, `:absorption`, or `:thermal_sum` for the positive channel sum.
"""
function matsubara_reconstruct(bath::ThermofieldRealPoles, frequencies;
                               block::Integer=1, kind::Symbol=:boson,
                               channel::Symbol=:source)
    kind == :boson ||
        throw(ArgumentError("matsubara_reconstruct currently supports kind=:boson"))
    if channel in (:source, :retarded, :spectral)
        return matsubara_reconstruct(bath.source, frequencies; block, kind)
    elseif channel == :emission
        return matsubara_reconstruct(bath.emission, frequencies; block, kind)
    elseif channel == :absorption
        return matsubara_reconstruct(bath.absorption, frequencies; block, kind)
    elseif channel in (:thermal_sum, :sum)
        return matsubara_reconstruct(bath.emission, frequencies; block, kind) +
               matsubara_reconstruct(bath.absorption, frequencies; block, kind)
    else
        throw(ArgumentError("unknown thermofield reconstruction channel $channel"))
    end
end

"""
    ComplexPoles

Type slot ONLY (§6.3 / `03_GRAFT_complex_pole.md`): BCF exponential-sum
pseudomode baths (complex poles). These belong to the Lindbladian/TTNDO route
and are deliberately not mountable as Hamiltonian `TTNS` bath sites.
"""
struct ComplexPoles <: BathParametrization end

"""
    fit_bath(J, P::Partition; T=0, beta=nothing, kind=:boson, nmodes, ωmin, ωmax,
             grid=:linear, method=:midpoint, solver=:nnls, domain=:real_axis,
             crossblock=:highmount, pole_family=:real) -> RealPoles | ThermofieldRealPoles

Blockwise real-pole Hamiltonian-bath fitting for the forwarded boson path.
`domain=:real_axis` consumes a continuous boson spectral density `J(ω)` and
uses deterministic midpoint quadrature: each bin produces one pole at the bin
midpoint and a residue `g_k^2 = ∫_bin J(ω)dω` approximated by the midpoint
rule. Matrix-valued spectral densities return a `d×d` Hermitian PSD matrix per
block and are discretized as the full residue `Rₖ = Δωₖ J(ωₖ)`; no
entry-by-entry square root is taken. `domain=:matsubara` consumes samples
`(; frequencies, values)` (or one such object per partition block) and solves
for nonnegative real-pole residues whose Matsubara reconstruction matches the
input grid.

For matrix-valued samples, provide `values` as a vector of square matrices,
one per frequency. `solver=:nnls` is valid for scalar or diagonal targets;
genuine off-diagonal targets require `solver=:psd` (`:sdp` is an alias) and
return [`MatrixRealPoles`](@ref). `method=:adaptive` performs deterministic,
data-driven bounded pole-coordinate refinement rather than changing the input
frequency grid.

* `T = 0`: Δ(iωₙ)/Δ(ω) → real-pole fit per block.
* `T > 0`, `kind=:boson`: fit the source real poles and split them into
  thermofield emission/absorption `RealPoles`; the bath initial state is the
  vacuum product state.
* `crossblock = :highmount | :rotate` (§6.2): high mounting near the tree
  center, or a pre-fit single-particle rotation (returned with results).
* `pole_family=:complex`: TODO(C3) BCF/Lindbladian fitting. Complex BCF poles
  are never routed into Hamiltonian bath mounting.

Mandatory self-checks to implement with it (§6.3): (1) β·δε ≪ 1 resolution
check; (2) loop-bath vs final-bath both projected back to Δ(iωₙ) and compared.
Global fitting across the whole Δ matrix while ignoring `P` is a forbidden
path — the interface does not offer it. TODO(B4+ future refinement):
replace/augment midpoint with adapol-style AAA initialization plus nonlinear
refinement.
"""
function fit_bath(J, P::Partition; T::Real=0, kind::Symbol=:boson,
                  nmodes::Integer=8, wmin=nothing, wmax=nothing,
                  ωmin=nothing, ωmax=nothing, grid::Symbol=:linear,
                  method::Symbol=:midpoint, domain::Symbol=:real_axis,
                  crossblock::Symbol=:highmount, beta=nothing, β=nothing,
                  pole_family::Symbol=:real, solver::Symbol=:nnls)
    if pole_family == :complex
        throw(ArgumentError("fit_bath pole_family=:complex is TODO(C3): ComplexPoles are BCF/Lindbladian fits and are not Hamiltonian-mountable"))
    elseif pole_family != :real
        throw(ArgumentError("fit_bath pole_family must be :real or :complex"))
    end
    kind == :boson ||
        throw(ArgumentError("fit_bath currently implements only kind=:boson; fermionic finite-T Hamiltonian stars need a charged hybridization mounting surface"))
    crossblock in (:highmount, :rotate) ||
        throw(ArgumentError("crossblock must be :highmount or :rotate"))
    nmodes > 0 || throw(ArgumentError("nmodes must be positive"))
    lo = _bound("ωmin", wmin, ωmin)
    hi = _bound("ωmax", wmax, ωmax)
    lo < hi || throw(ArgumentError("ωmin must be smaller than ωmax"))
    dom = _canonical_domain(domain)
    fitmethod = _canonical_fit_method(method, dom)
    fitsolver = _canonical_solver(solver)
    temp = _temperature_spec(T, beta, β)

    if dom == :real_axis && J isa Function && length(P.blocks) > 1
        probe = first(_pole_midpoints(lo, hi, Int(nmodes), grid))
        J(probe) isa AbstractMatrix &&
            throw(ArgumentError("matrix-valued real-axis fitting with multiple partition blocks requires one spectral-density function per block"))
    end

    targets = dom == :real_axis ? _block_targets(J, P) : _matsubara_targets(J, P)
    source = _fit_realpoles_from_targets(targets, P; kind, T=temp.T, dom,
                                         fitmethod, lo, hi, nmodes=Int(nmodes),
                                         grid, crossblock, channel=:source,
                                         solver=fitsolver)
    temp.finite && source isa MatrixRealPoles &&
        throw(ArgumentError("finite-temperature matrix thermofield splitting is not implemented"))
    temp.finite || return source
    return _thermofield_from_source(source; beta=temp.beta, temperature=temp.T,
                                    kind, dom, fitmethod, grid,
                                    nmodes=Int(nmodes), lo, hi, crossblock)
end

function _fit_realpoles_from_targets(targets, P::Partition; kind::Symbol, T::Float64,
                                     dom::Symbol, fitmethod::Symbol, lo::Float64,
                                     hi::Float64, nmodes::Int, grid::Symbol,
                                     crossblock::Symbol, channel::Symbol,
                                     solver::Symbol)
    matrix_flags = [_is_matrix_target(target, dom, lo, hi, nmodes, grid)
                    for target in targets]
    if any(matrix_flags)
        all(matrix_flags) ||
            throw(ArgumentError("matrix-valued fitting requires matrix samples for every partition block"))
        return _fit_matrixrealpoles_from_targets(
            targets, P; kind, T, dom, fitmethod, lo, hi, nmodes, grid,
            crossblock, channel, solver)
    end
    poles = Float64[]
    residues = Float64[]
    ranges = UnitRange{Int}[]
    block_diags = map(enumerate(targets)) do (i, target)
        start = length(poles) + 1
        p, r, diag = if dom == :real_axis
            p, r = _midpoint_modes(target, lo, hi, Int(nmodes), grid)
            p, r, _fit_diagnostics(target, p, r, lo, hi, Int(nmodes), grid, i)
        else
            _fit_matsubara_modes(target, lo, hi, Int(nmodes), grid, i;
                                 fitmethod, solver)
        end
        append!(poles, p)
        append!(residues, r)
        stop = length(poles)
        push!(ranges, start:stop)
        diag
    end
    diagnostics = (;
        kind,
        T,
        method = fitmethod,
        domain = dom,
        grid,
        nmodes,
        ωmin = lo,
        ωmax = hi,
        crossblock,
        channel,
        solver,
        block_diagnostics = block_diags,
    )
    return RealPoles(poles, residues, deepcopy(P.blocks), ranges, diagnostics)
end

function _fit_matrixrealpoles_from_targets(targets, P::Partition;
                                           kind::Symbol, T::Float64,
                                           dom::Symbol,
                                           fitmethod::Symbol, lo::Float64,
                                           hi::Float64, nmodes::Int,
                                           grid::Symbol, crossblock::Symbol,
                                           channel::Symbol, solver::Symbol)
    poles = Float64[]
    residues = Matrix{ComplexF64}[]
    ranges = UnitRange{Int}[]
    block_diags = map(enumerate(targets)) do (i, target)
        start = length(poles) + 1
        p, r, diag = if dom == :real_axis
            _midpoint_matrix_modes(target, lo, hi, nmodes, grid,
                                   length(P.blocks[i]), i)
        else
            _validate_matrix_target(target, length(P.blocks[i]), i)
            _fit_matrix_matsubara_modes(
                target, lo, hi, nmodes, grid, i; fitmethod, solver)
        end
        append!(poles, p)
        append!(residues, r)
        push!(ranges, start:length(poles))
        diag
    end
    diagnostics = (;
        kind,
        T,
        method = fitmethod,
        domain = dom,
        grid,
        nmodes,
        ωmin = lo,
        ωmax = hi,
        crossblock,
        channel,
        solver,
        residue_constraint = :positive_semidefinite,
        block_diagnostics = block_diags,
    )
    return MatrixRealPoles(poles, residues, deepcopy(P.blocks), ranges, diagnostics)
end

"""
    mount_bath(topo, bath::RealPoles, P::Partition; mode=:star, prefix=:bath, attach=nothing)

Return a named tuple with a new topology and the mode-site labels created for
`bath`. `mode=:star` mounts one boson leaf per fitted mode under the block
anchor; `mode=:chain` mounts one chain per block for cross-validation only.
`attach` may be omitted (first orbital in each block), a `Symbol` for a
single-block partition, a vector of symbols, a dictionary keyed by block index,
or a function `(block, i) -> site`.
"""
function mount_bath(topo::TreeTopology, bath::RealPoles, P::Partition;
                    mode::Symbol=:star, prefix::Symbol=:bath, attach=nothing)
    P.blocks == bath.blocks ||
        throw(ArgumentError("mount_bath: partition does not match RealPoles blocks"))
    mode in (:star, :chain) || throw(ArgumentError("mount_bath mode must be :star or :chain"))
    top = topo
    sites = Symbol[]
    anchors = Symbol[]
    block_sites = Vector{Vector{Symbol}}()
    for (bi, block) in enumerate(P.blocks)
        anchor = _block_anchor(block, bi, attach)
        try
            nodeindex(top, anchor)
        catch err
            err isa KeyError || rethrow()
            throw(ArgumentError("mount_bath anchor $anchor is not present in the topology"))
        end
        r = bath.block_ranges[bi]
        local_sites = Symbol[]
        if mode == :chain
            pref = Symbol(prefix, bi, :_)
            top = mount_chain(top, anchor, length(r); prefix=pref)
            for j in eachindex(r)
                site = Symbol(pref, j)
                push!(sites, site); push!(anchors, anchor); push!(local_sites, site)
            end
        else
            for j in eachindex(r)
                pref = Symbol(prefix, bi, :_, j, :_)
                top = mount_chain(top, anchor, 1; prefix=pref)
                site = Symbol(pref, 1)
                push!(sites, site); push!(anchors, anchor); push!(local_sites, site)
            end
        end
        push!(block_sites, local_sites)
    end
    return (; topology=top, sites, anchors, block_sites)
end

"""
    mount_bath(topo, bath::ThermofieldRealPoles, P::Partition; mode=:star, prefix=:bath, attach=nothing)

Mount the emission and absorption thermofield stars as ordinary boson bath
branches. The return value keeps the two channel mount records separate and
also exposes concatenated `sites`, `anchors`, and per-block `block_sites`.
"""
function mount_bath(topo::TreeTopology, bath::ThermofieldRealPoles, P::Partition;
                    mode::Symbol=:star, prefix::Symbol=:bath, attach=nothing)
    emission = mount_bath(topo, bath.emission, P;
                          mode, prefix=Symbol(prefix, :em_), attach)
    absorption = mount_bath(emission.topology, bath.absorption, P;
                            mode, prefix=Symbol(prefix, :abs_), attach)
    block_sites = [vcat(emission.block_sites[i], absorption.block_sites[i])
                   for i in eachindex(emission.block_sites)]
    return (;
        topology = absorption.topology,
        emission,
        absorption,
        sites = vcat(emission.sites, absorption.sites),
        anchors = vcat(emission.anchors, absorption.anchors),
        block_sites,
        channel_sites = (; emission=emission.block_sites, absorption=absorption.block_sites),
    )
end

"""
    BosonBath(J; partition, topology, matter_ops, boson_ops, mode=:star, kwargs...)

Continuous boson-bath entry point. Fits `J(ω)` with [`fit_bath`](@ref),
mounts explicit boson sites with [`mount_bath`](@ref), and emits ordinary
symbolic terms via `boson_modes` and `BosonCoupling` for zero-temperature
`RealPoles`. Finite-temperature `ThermofieldRealPoles` fitting/mounting is
available through `fit_bath`/`mount_bath`; lowering the two thermofield coupling
channels to a single `OpSum` is a later solver-convention step. Returns
`(; bath, topology, sites, anchors, phys, H)`, where `phys` contains the new
bath-site physical spaces and `H` is an `OpSum`.
"""
function BosonBath(J; partition::Partition, topology::TreeTopology, matter_ops,
                   boson_ops, mode::Symbol=:star, density::Symbol=:N,
                   prefix::Symbol=:bath, attach=nothing, kwargs...)
    bath = fit_bath(J, partition; kwargs...)
    bath isa ThermofieldRealPoles &&
        throw(ArgumentError("BosonBath finite-T thermofield OpSum lowering is not implemented; use fit_bath/mount_bath for typed thermofield star data"))
    bath isa MatrixRealPoles &&
        throw(ArgumentError("BosonBath matrix-residue OpSum lowering is not implemented; use factorize_residues to obtain repeated-energy coupling vectors"))
    mounted = mount_bath(topology, bath, partition; mode, prefix, attach)
    g = couplings(bath)
    modes = [site => bath.poles[k] for (k, site) in enumerate(mounted.sites)]
    coupls = [(mounted.anchors[k], mounted.sites[k]) => g[k] for k in eachindex(g)]
    H = boson_modes(modes; ops=boson_ops) +
        BosonCoupling(coupls, :density; matter_ops, boson_ops, density)
    phys = Dict(site => boson_ops.P for site in mounted.sites)
    return (; bath, topology=mounted.topology, sites=mounted.sites,
            anchors=mounted.anchors, phys, H)
end

# ---------------------------------------------------------------------------
# §6.1 geometry constructors — thin wrappers over Trees.Geometries that mount
# bath branches according to the Partition (mechanical expansion, §6.2).
# Star/chain branch assembly from (Partition, BathParametrization) is implemented
# by `mount_bath`; generic fork-layout construction from partitions remains a
# future M5+ topology-planning surface. Trees already provides:
# mps/star/binary/fork topologies and the is_t3ns predicate.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §6.4 measurements — impurity-solver data products beyond the forwarded B5
# local correlator snapshots: sparse IR/DLR grids, F(t) improved estimators
# (Σ = F·G⁻¹), χ_ch(τ) two-particle output, and TRIQS BlockGf round-trip.
# TODO(future impurity-measurement milestone). §6.5 spectral post-processing
# consumes Evolver snapshots but is a separate M1+ post-processing surface.
# ---------------------------------------------------------------------------

# TODO(future impurity-solver integration): the single self-consistency-facing
# entry point (§6.6). Graft does NOT implement the DMFT/EDMFT loop itself.
# Contract: `ψ0` warm starts are first-class (topology hash validated — refuse
# silently rebuilt geometry); basis rotations `U` are returned with the results,
# loop side stays oblivious.
"""
    solve(bath, H_loc; partition, T, observables, ψ0=nothing) -> (; G, Σ, χ, ψ, U)

Impurity-solver extension point. The finite zero-temperature
`AndersonBath`/real-pole method is implemented in `anderson_real_poles.jl`;
finite-temperature and self-energy-producing methods remain later extensions.
"""
function solve end

function _contiguous_ranges(rs::Vector{UnitRange{Int}})
    out = UnitRange{Int}[]
    next = 1
    for r in rs
        isempty(r) && throw(ArgumentError("RealPoles block ranges may not be empty"))
        first(r) == next || return out
        push!(out, r)
        next = last(r) + 1
    end
    return out
end

function _bound(name::String, ascii, unicode)
    v = unicode === nothing ? ascii : unicode
    ascii_name = name == "ωmin" ? "wmin" : "wmax"
    v === nothing && throw(ArgumentError("fit_bath requires `$name`/`$ascii_name`"))
    x = Float64(v)
    isfinite(x) || throw(ArgumentError("$name must be finite"))
    return x
end

function _temperature_spec(T::Real, beta, β)
    Tval = Float64(T)
    isfinite(Tval) && Tval >= 0 ||
        throw(ArgumentError("temperature T must be finite and nonnegative"))
    if beta !== nothing && β !== nothing && Float64(beta) != Float64(β)
        throw(ArgumentError("beta and β keywords disagree"))
    end
    βarg = β === nothing ? beta : β
    if βarg === nothing
        Tval == 0 && return (; finite=false, T=0.0, beta=Inf)
        return (; finite=true, T=Tval, beta=inv(Tval))
    end
    βval = Float64(βarg)
    isfinite(βval) && βval > 0 ||
        throw(ArgumentError("beta must be finite and positive for finite-temperature bath fitting"))
    implied_T = inv(βval)
    if Tval > 0 && !isapprox(Tval, implied_T; rtol=1e-12, atol=0.0)
        throw(ArgumentError("T and beta specify different finite temperatures"))
    end
    return (; finite=true, T=implied_T, beta=βval)
end

function _thermofield_from_source(source::RealPoles; beta::Float64,
                                  temperature::Float64, kind::Symbol,
                                  dom::Symbol, fitmethod::Symbol, grid::Symbol,
                                  nmodes::Int, lo::Float64, hi::Float64,
                                  crossblock::Symbol)
    occupations = [_bose_occupation(beta, ω) for ω in source.poles]
    emission_residues = (occupations .+ 1.0) .* source.residues
    absorption_residues = occupations .* source.residues
    emission_diag = _thermofield_channel_diagnostics(
        source, emission_residues, occupations, :emission; beta, temperature,
        kind, dom, fitmethod, grid, nmodes, lo, hi, crossblock)
    absorption_diag = _thermofield_channel_diagnostics(
        source, absorption_residues, occupations, :absorption; beta, temperature,
        kind, dom, fitmethod, grid, nmodes, lo, hi, crossblock)
    emission = RealPoles(source.poles, emission_residues, source.blocks,
                         source.block_ranges, emission_diag)
    absorption = RealPoles(source.poles, absorption_residues, source.blocks,
                           source.block_ranges, absorption_diag)
    diagnostics = _thermofield_diagnostics(
        source, emission, absorption; beta, temperature, kind, dom, fitmethod,
        grid, nmodes, lo, hi, crossblock)
    return ThermofieldRealPoles(source, emission, absorption; beta,
                                temperature, diagnostics)
end

function _bose_occupation(beta::Float64, ω::Float64)
    x = beta * ω
    x > 700 && return 0.0
    return inv(expm1(x))
end

function _thermofield_channel_diagnostics(source::RealPoles, residues,
                                          occupations, channel::Symbol;
                                          beta::Float64, temperature::Float64,
                                          kind::Symbol, dom::Symbol,
                                          fitmethod::Symbol, grid::Symbol,
                                          nmodes::Int, lo::Float64,
                                          hi::Float64, crossblock::Symbol)
    block_diags = map(enumerate(source.block_ranges)) do (i, r)
        factors = channel == :emission ? occupations[r] .+ 1.0 : occupations[r]
        (;
            block_index = i,
            source_domain = dom,
            channel,
            spectral_weight = sum(residues[r]),
            source_spectral_weight = sum(source.residues[r]),
            thermal_factor_min = minimum(factors),
            thermal_factor_max = maximum(factors),
        )
    end
    return (;
        kind,
        T = temperature,
        beta,
        method = :thermofield_split,
        source_method = fitmethod,
        domain = dom,
        grid,
        nmodes,
        ωmin = lo,
        ωmax = hi,
        crossblock,
        channel,
        block_diagnostics = block_diags,
    )
end

function _thermofield_diagnostics(source::RealPoles, emission::RealPoles,
                                  absorption::RealPoles; beta::Float64,
                                  temperature::Float64, kind::Symbol,
                                  dom::Symbol, fitmethod::Symbol, grid::Symbol,
                                  nmodes::Int, lo::Float64, hi::Float64,
                                  crossblock::Symbol)
    recon = _thermofield_source_reconstruction(source, emission, absorption)
    βδε = _beta_delta_energy(source, beta)
    return (;
        kind,
        T = temperature,
        beta,
        method = :thermofield_split,
        source_method = fitmethod,
        domain = dom,
        grid,
        nmodes,
        ωmin = lo,
        ωmax = hi,
        crossblock,
        statistics = :boson,
        representation = :thermofield_star,
        channels = (:emission, :absorption),
        beta_delta_energy = βδε,
        resolution_ok = βδε < 1.0,
        source_diagnostics = source.diagnostics,
        source_reconstruction = recon,
    )
end

function _thermofield_source_reconstruction(source::RealPoles, emission::RealPoles,
                                            absorption::RealPoles)
    block_diags = map(enumerate(source.block_ranges)) do (i, r)
        reconstructed = emission.residues[r] .- absorption.residues[r]
        expected = source.residues[r]
        err = reconstructed .- expected
        rel = norm(err) / max(norm(expected), eps(Float64))
        maxabs = isempty(err) ? 0.0 : maximum(abs.(err))
        (;
            block_index = i,
            source_domain = source.diagnostics.domain,
            relative_residual = rel,
            max_abs_error = maxabs,
            spectral_weight = sum(reconstructed),
            source_spectral_weight = sum(expected),
        )
    end
    return (;
        channel_combination = :emission_minus_absorption,
        relative_residual = maximum(d.relative_residual for d in block_diags),
        max_abs_error = maximum(d.max_abs_error for d in block_diags),
        block_diagnostics = block_diags,
    )
end

function _beta_delta_energy(source::RealPoles, beta::Float64)
    poles = sort(unique(source.poles))
    length(poles) < 2 && return Inf
    return beta * minimum(diff(poles))
end

function _block_targets(J, P::Partition)
    n = length(P.blocks)
    if J isa Function
        return [J for _ in 1:n]
    elseif J isa AbstractVector
        length(J) == n || throw(ArgumentError("vector-valued spectral density needs one function per partition block"))
        all(f -> f isa Function, J) || throw(ArgumentError("spectral-density vector entries must be functions"))
        return collect(J)
    else
        throw(ArgumentError("fit_bath expects a spectral-density function or one function per partition block"))
    end
end

function _canonical_domain(domain::Symbol)
    domain in (:real_axis, :real, :spectral) && return :real_axis
    domain in (:matsubara, :iw, :iω, :iν) && return :matsubara
    throw(ArgumentError("fit_bath domain must be :real_axis or :matsubara"))
end

function _canonical_fit_method(method::Symbol, domain::Symbol)
    if domain == :real_axis
        method in (:midpoint, :equal_spacing) ||
            throw(ArgumentError("fit_bath method=$method is unavailable for domain=:real_axis; TODO(B4+ future) adapol/AAA refinement"))
        return :midpoint
    end
    method == :adaptive && return :adaptive
    method in (:midpoint, :least_squares, :lsq) ||
        throw(ArgumentError("fit_bath method=$method is unavailable for domain=:matsubara; use :least_squares or :adaptive"))
    return :least_squares
end

function _canonical_solver(solver::Symbol)
    solver == :sdp && return :psd
    solver in (:nnls, :psd) ||
        throw(ArgumentError("fit_bath solver must be :nnls, :psd, or :sdp"))
    return solver
end

function _matsubara_targets(data, P::Partition)
    n = length(P.blocks)
    if data isa NamedTuple
        νs, vals = _matsubara_dataset(data)
        if _is_matrix_series(vals)
            n == 1 ||
                throw(ArgumentError("one matrix-valued Matsubara series fits one partition block; provide one NamedTuple per block"))
            return [(νs, vals)]
        elseif vals isa AbstractMatrix
            size(vals, 1) == length(νs) ||
                throw(ArgumentError("Matsubara value matrix must have one row per frequency"))
            size(vals, 2) == n ||
                throw(ArgumentError("Matsubara value matrix needs one column per partition block"))
            return [(νs, collect(vals[:, i])) for i in 1:n]
        elseif vals isa AbstractVector && length(vals) == n &&
               all(v -> v isa AbstractVector, vals)
            return [(νs, _matsubara_values(v)) for v in vals]
        else
            n == 1 ||
                throw(ArgumentError("single Matsubara value vector only fits a one-block partition"))
            return [(νs, _matsubara_values(vals))]
        end
    elseif data isa AbstractVector && length(data) == n &&
           all(x -> x isa NamedTuple, data)
        return [_matsubara_dataset(x) for x in data]
    else
        throw(ArgumentError("domain=:matsubara expects `(; frequencies, values)` or one such NamedTuple per partition block"))
    end
end

function _matsubara_dataset(data::NamedTuple)
    fkey = _first_present(data, (:frequencies, :freqs, :ω, :omega, :ν, :nu,
                                 :iω, :iw, :iν, :iv, :iwn))
    vkey = _first_present(data, (:values, :data, :U, :u, :Δ, :Delta, :delta))
    νs = _matsubara_frequencies(getfield(data, fkey))
    vals = getfield(data, vkey)
    if _is_matrix_series(vals)
        length(vals) == length(νs) ||
            throw(ArgumentError("matrix-valued Matsubara data needs one matrix per frequency"))
        return νs, _matsubara_matrices(vals)
    end
    if vals isa AbstractMatrix
        return νs, vals
    end
    ys = _matsubara_values(vals)
    length(ys) == length(νs) ||
        throw(ArgumentError("Matsubara data needs one value per frequency"))
    return νs, ys
end

_is_matrix_series(values) = values isa AbstractVector &&
    !isempty(values) && all(v -> v isa AbstractMatrix, values)
function _is_matrix_target(target, dom::Symbol, lo::Float64, hi::Float64,
                           nmodes::Int, grid::Symbol)
    if dom == :matsubara
        return target isa Tuple && length(target) == 2 &&
               _is_matrix_series(target[2])
    end
    target isa Function || return false
    probe = first(_pole_midpoints(lo, hi, nmodes, grid))
    return target(probe) isa AbstractMatrix
end

function _matsubara_matrices(values)
    first_size = size(first(values))
    length(first_size) == 2 && first_size[1] == first_size[2] ||
        throw(ArgumentError("matrix-valued Matsubara samples must be square"))
    out = Matrix{ComplexF64}[]
    for (i, value) in enumerate(values)
        size(value) == first_size ||
            throw(ArgumentError("matrix-valued Matsubara sample $i has inconsistent size"))
        M = Matrix{ComplexF64}(value)
        all(z -> isfinite(real(z)) && isfinite(imag(z)), M) ||
            throw(ArgumentError("matrix-valued Matsubara samples must be finite"))
        norm(M - M') <= 100sqrt(eps(Float64)) * max(norm(M), 1.0) ||
            throw(ArgumentError("matrix-valued Matsubara sample $i must be Hermitian"))
        push!(out, (M + M') / 2)
    end
    return out
end

function _validate_matrix_target(target, blockdim::Int, block_index::Int)
    _, values = target
    _is_matrix_series(values) ||
        throw(ArgumentError("block $block_index does not contain matrix-valued samples"))
    size(first(values)) == (blockdim, blockdim) ||
        throw(ArgumentError("block $block_index matrix samples must be $blockdim×$blockdim"))
    return nothing
end

function _first_present(data::NamedTuple, keys)
    for k in keys
        haskey(data, k) && return k
    end
    throw(ArgumentError("Matsubara data is missing one of $(keys)"))
end

function _matsubara_frequencies(xs)
    νs = Float64[_matsubara_frequency(x) for x in xs]
    all(isfinite, νs) || throw(ArgumentError("Matsubara frequencies must be finite"))
    all(x -> x >= 0, νs) || throw(ArgumentError("Matsubara frequencies must be nonnegative"))
    return νs
end

function _matsubara_frequency(x)
    z = complex(x)
    if abs(imag(z)) > 100eps(Float64) &&
       abs(real(z)) <= 100eps(Float64) * max(abs(imag(z)), 1)
        return abs(Float64(imag(z)))
    elseif abs(imag(z)) <= 100eps(Float64) * max(abs(real(z)), 1)
        return abs(Float64(real(z)))
    else
        throw(ArgumentError("Matsubara frequencies must be real magnitudes or pure-imaginary points"))
    end
end

_matsubara_values(xs) = Float64[_matsubara_value(x) for x in xs]

function _matsubara_value(x)
    z = complex(x)
    abs(imag(z)) <= 100eps(Float64) * max(abs(real(z)), 1) ||
        throw(ArgumentError("bosonic Matsubara samples must be real-valued"))
    return Float64(real(z))
end

function _fit_matsubara_modes(target, lo::Float64, hi::Float64, nmodes::Int,
                              grid::Symbol, block_index::Int;
                              fitmethod::Symbol, solver::Symbol)
    νs, values = target
    initial_poles = _pole_midpoints(lo, hi, nmodes, grid)
    poles, initial_residual = if fitmethod == :adaptive
        _adaptive_scalar_poles(νs, values, initial_poles, lo, hi, solver)
    else
        initial_poles, nothing
    end
    A = _boson_matsubara_kernel(νs, poles)
    residues, iterations = _nnls(A, values)
    reconstructed = A * residues
    err = reconstructed - values
    rel = norm(err) / max(norm(values), eps(Float64))
    maxabs = isempty(err) ? 0.0 : maximum(abs.(err))
    maxrel = isempty(err) ? 0.0 :
        maximum(abs(err[i]) / max(abs(values[i]), eps(Float64)) for i in eachindex(err))
    diag = (;
        block_index,
        source_domain = :matsubara,
        npoints = length(νs),
        relative_residual = rel,
        max_abs_error = maxabs,
        max_rel_error = maxrel,
        frequencies = copy(νs),
        values = copy(values),
        reconstructed,
        solver,
        pole_selection = fitmethod == :adaptive ? :bounded_coordinate_refinement : :equal_spacing,
        initial_relative_residual = initial_residual,
        solver_iterations = iterations,
    )
    return poles, residues, diag
end

function _fit_matrix_matsubara_modes(target, lo::Float64, hi::Float64,
                                     nmodes::Int, grid::Symbol,
                                     block_index::Int; fitmethod::Symbol,
                                     solver::Symbol)
    νs, values = target
    offdiag = _has_genuine_offdiag(values)
    solver == :nnls && offdiag &&
        throw(ArgumentError("solver=:nnls only supports scalar or diagonal Matsubara targets; use solver=:psd or :sdp for off-diagonal data"))
    initial_poles = _pole_midpoints(lo, hi, nmodes, grid)
    poles, initial_residual = if fitmethod == :adaptive
        _adaptive_matrix_poles(νs, values, initial_poles, lo, hi, solver)
    else
        initial_poles, nothing
    end
    A = _boson_matsubara_kernel(νs, poles)
    residues, iterations = if solver == :nnls
        _diagonal_nnls_residues(A, values)
    else
        _psd_least_squares(A, values)
    end
    reconstructed = _matrix_reconstruct_series(A, residues)
    errnorm = _matrix_series_distance(reconstructed, values)
    valnorm = _matrix_series_norm(values)
    rel = errnorm / max(valnorm, eps(Float64))
    maxabs = maximum(abs, reduce(vcat, vec.(reconstructed .- values)); init=0.0)
    diag = (;
        block_index,
        source_domain = :matsubara,
        npoints = length(νs),
        matrix_dimension = size(first(values), 1),
        relative_residual = rel,
        max_abs_error = maxabs,
        frequencies = copy(νs),
        values = _matrix_series_array(values),
        reconstructed = _matrix_series_array(reconstructed),
        solver,
        residue_constraint = :positive_semidefinite,
        pole_selection = fitmethod == :adaptive ? :bounded_coordinate_refinement : :equal_spacing,
        initial_relative_residual = initial_residual,
        solver_iterations = iterations,
    )
    return poles, residues, diag
end

"Lawson-Hanson active-set nonnegative least squares."
function _nnls(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real};
               maxiter::Int=max(100, 30size(A, 2)),
               tol::Float64=max(size(A)...) * eps(Float64) *
                            max(opnorm(A), 1.0) * max(norm(b), 1.0))
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("NNLS right-hand side length mismatch"))
    x = zeros(Float64, n)
    passive = falses(n)
    w = Float64.(A' * (b - A * x))
    iterations = 0
    while any(i -> !passive[i] && w[i] > tol, 1:n)
        iterations += 1
        iterations <= maxiter ||
            throw(ErrorException("NNLS active-set iteration limit exceeded"))
        candidate = argmax([passive[i] ? -Inf : w[i] for i in 1:n])
        passive[candidate] = true
        while true
            z = zeros(Float64, n)
            inds = findall(passive)
            z[inds] = A[:, inds] \ b
            all(z[i] > tol for i in inds) && (x = z; break)
            ratios = [x[i] / (x[i] - z[i]) for i in inds
                      if z[i] <= tol && x[i] - z[i] > tol]
            α = isempty(ratios) ? 0.0 : minimum(ratios)
            x .+= α .* (z .- x)
            for i in inds
                if x[i] <= tol
                    x[i] = 0.0
                    passive[i] = false
                end
            end
        end
        w = Float64.(A' * (b - A * x))
    end
    x .= max.(x, 0.0)
    return x, iterations
end

function _diagonal_nnls_residues(A, values)
    d = size(first(values), 1)
    residues = [zeros(ComplexF64, d, d) for _ in axes(A, 2)]
    iterations = 0
    for i in 1:d
        rhs = Float64[real(M[i, i]) for M in values]
        x, it = _nnls(A, rhs)
        iterations += it
        for k in eachindex(residues)
            residues[k][i, i] = x[k]
        end
    end
    return residues, iterations
end

function _psd_least_squares(A, values; maxiter::Int=2000,
                            tol::Float64=1e-11)
    d = size(first(values), 1)
    nmode = size(A, 2)
    # Projected unconstrained least-squares initialization is deterministic and
    # exact whenever the supplied shared poles already span the data.
    raw = [zeros(ComplexF64, d, d) for _ in 1:nmode]
    for i in 1:d, j in 1:d
        coeff = A \ ComplexF64[M[i, j] for M in values]
        for k in 1:nmode
            raw[k][i, j] = coeff[k]
        end
    end
    X = [_project_psd(R) for R in raw]
    Z = copy.(X)
    t = 1.0
    L = opnorm(A)^2
    L > 0 || return X, 0
    step = inv(L)
    for iteration in 1:maxiter
        predicted = _matrix_reconstruct_series(A, Z)
        errors = predicted .- values
        gradients = [sum(A[n, k] .* errors[n] for n in axes(A, 1))
                     for k in axes(A, 2)]
        Xnew = [_project_psd(Z[k] .- step .* gradients[k]) for k in 1:nmode]
        delta = _matrix_series_distance(Xnew, X)
        scale = max(_matrix_series_norm(Xnew), 1.0)
        delta <= tol * scale && return Xnew, iteration
        tnew = (1 + sqrt(1 + 4t^2)) / 2
        momentum = (t - 1) / tnew
        Z = [Xnew[k] .+ momentum .* (Xnew[k] .- X[k]) for k in 1:nmode]
        X = Xnew
        t = tnew
    end
    return X, maxiter
end

function _project_psd(M)
    F = eigen(Hermitian((M + M') / 2))
    vals = max.(Float64.(F.values), 0.0)
    R = Matrix{ComplexF64}(F.vectors * (vals .* F.vectors'))
    return (R + R') / 2
end

function _validated_psd_residue(M::Matrix{ComplexF64}, k::Int)
    all(z -> isfinite(real(z)) && isfinite(imag(z)), M) ||
        throw(ArgumentError("residue $k must contain only finite values"))
    scale = max(norm(M), 1.0)
    norm(M - M') <= 100sqrt(eps(Float64)) * scale ||
        throw(ArgumentError("residue $k must be Hermitian"))
    F = eigen(Hermitian((M + M') / 2))
    minimum(F.values) >= -100sqrt(eps(Float64)) * scale ||
        throw(ArgumentError("residue $k must be positive semidefinite"))
    vals = max.(Float64.(F.values), 0.0)
    R = Matrix{ComplexF64}(F.vectors * (vals .* F.vectors'))
    return (R + R') / 2
end

function _has_genuine_offdiag(values)
    for M in values
        diagonal = Matrix(Diagonal(diag(M)))
        norm(M - diagonal) > 100sqrt(eps(Float64)) * max(norm(M), 1.0) &&
            return true
    end
    return false
end

function _matrix_reconstruct_series(A, residues)
    return [sum(A[n, k] .* residues[k] for k in axes(A, 2))
            for n in axes(A, 1)]
end

_matrix_series_norm(values) =
    sqrt(sum((sum(abs2, M; init=0.0) for M in values); init=0.0))
_matrix_series_distance(a, b) =
    sqrt(sum((sum(abs2, a[i] .- b[i]; init=0.0) for i in eachindex(a));
             init=0.0))

function _matrix_series_array(values)
    d = size(first(values), 1)
    out = Array{ComplexF64}(undef, length(values), d, d)
    for n in eachindex(values)
        @views out[n, :, :] .= values[n]
    end
    return out
end

function _adaptive_scalar_poles(νs, values, initial, lo, hi, solver)
    objective = poles -> begin
        A = _boson_matsubara_kernel(νs, poles)
        residues, _ = _nnls(A, values)
        norm(A * residues - values) / max(norm(values), eps(Float64))
    end
    return _bounded_coordinate_refinement(initial, lo, hi, objective)
end

function _adaptive_matrix_poles(νs, values, initial, lo, hi, solver)
    objective = poles -> begin
        A = _boson_matsubara_kernel(νs, poles)
        residues, _ = solver == :nnls ? _diagonal_nnls_residues(A, values) :
                                        _psd_least_squares(A, values; maxiter=500)
        reconstructed = _matrix_reconstruct_series(A, residues)
        _matrix_series_distance(reconstructed, values) /
            max(_matrix_series_norm(values), eps(Float64))
    end
    return _bounded_coordinate_refinement(initial, lo, hi, objective)
end

function _bounded_coordinate_refinement(initial, lo::Float64, hi::Float64,
                                        objective)
    poles = copy(initial)
    initial_loss = objective(poles)
    best_loss = initial_loss
    radius = (hi - lo) / max(length(poles), 1)
    separation = max((hi - lo) * 1e-10, 10eps(max(abs(hi), 1.0)))
    for _ in 1:8
        for k in eachindex(poles)
            lower = k == firstindex(poles) ? max(lo + separation, separation) :
                    poles[k - 1] + separation
            upper = k == lastindex(poles) ? hi - separation :
                    poles[k + 1] - separation
            lower < upper || continue
            candidates = unique(sort(clamp.(
                poles[k] .+ collect(range(-radius, radius; length=17)),
                lower, upper)))
            local_poles = copy(poles)
            local_loss = best_loss
            for candidate in candidates
                trial = copy(poles)
                trial[k] = candidate
                loss = objective(trial)
                if loss < local_loss
                    local_loss = loss
                    local_poles = trial
                end
            end
            poles = local_poles
            best_loss = local_loss
        end
        radius *= 0.35
    end
    return poles, initial_loss
end

function _pole_midpoints(lo::Float64, hi::Float64, nmodes::Int, grid::Symbol)
    edges = _grid_edges(lo, hi, nmodes, grid)
    return [grid == :log ? sqrt(edges[k] * edges[k + 1]) :
            (edges[k] + edges[k + 1]) / 2 for k in 1:nmodes]
end

function _boson_matsubara_kernel(νs, poles)
    A = Matrix{Float64}(undef, length(νs), length(poles))
    for i in eachindex(νs), j in eachindex(poles)
        ω = poles[j]
        A[i, j] = 2 * ω / (νs[i]^2 + ω^2)
    end
    return A
end

function _grid_edges(lo::Float64, hi::Float64, nmodes::Int, grid::Symbol)
    if grid == :linear
        return collect(range(lo, hi; length=nmodes + 1))
    elseif grid == :log
        lo > 0 || throw(ArgumentError("log bath grid requires ωmin > 0"))
        return exp.(range(log(lo), log(hi); length=nmodes + 1))
    else
        throw(ArgumentError("unknown bath grid $grid; expected :linear or :log"))
    end
end

function _midpoint_matrix_modes(J::Function, lo::Float64, hi::Float64,
                                nmodes::Int, grid::Symbol, blockdim::Int,
                                block_index::Int)
    edges = _grid_edges(lo, hi, nmodes, grid)
    poles = Float64[]
    residues = Matrix{ComplexF64}[]
    for k in 1:nmodes
        a, b = edges[k], edges[k + 1]
        ω = grid == :log ? sqrt(a * b) : (a + b) / 2
        density = _matrix_spectral_density(J(ω), blockdim, block_index, ω)
        push!(poles, ω)
        push!(residues, (b - a) .* density)
    end
    weights = [sum(real(R[i, i]) for i in axes(R, 1)) for R in residues]
    diagnostics = (;
        block_index,
        source_domain = :real_axis,
        matrix_dimension = blockdim,
        spectral_weight = sum(weights),
        pole_weights = weights,
        quadrature = :midpoint,
    )
    return poles, residues, diagnostics
end

function _matrix_spectral_density(value, blockdim::Int, block_index::Int,
                                  ω::Float64)
    value isa AbstractMatrix ||
        throw(ArgumentError("matrix spectral density for block $block_index must return a matrix"))
    size(value) == (blockdim, blockdim) ||
        throw(ArgumentError("matrix spectral density for block $block_index must return a $blockdim×$blockdim matrix"))
    M = Matrix{ComplexF64}(value)
    all(z -> isfinite(real(z)) && isfinite(imag(z)), M) ||
        throw(ArgumentError("matrix spectral density at ω=$ω must be finite"))
    scale = max(norm(M), 1.0)
    norm(M - M') <= 100sqrt(eps(Float64)) * scale ||
        throw(ArgumentError("matrix spectral density at ω=$ω must be Hermitian"))
    F = eigen(Hermitian((M + M') / 2))
    minimum(F.values) >= -100sqrt(eps(Float64)) * scale ||
        throw(ArgumentError("matrix spectral density at ω=$ω must be positive semidefinite"))
    vals = max.(Float64.(F.values), 0.0)
    density = Matrix{ComplexF64}(F.vectors * (vals .* F.vectors'))
    return (density + density') / 2
end

function _midpoint_modes(J::Function, lo::Float64, hi::Float64, nmodes::Int, grid::Symbol)
    edges = _grid_edges(lo, hi, nmodes, grid)
    poles = Float64[]
    residues = Float64[]
    for k in 1:nmodes
        a, b = edges[k], edges[k + 1]
        ω = grid == :log ? sqrt(a * b) : (a + b) / 2
        weight = Float64(real(J(ω))) * (b - a)
        weight >= -100eps(Float64) ||
            throw(ArgumentError("boson spectral density must be nonnegative; got J($ω) = $(J(ω))"))
        push!(poles, ω)
        push!(residues, max(weight, 0.0))
    end
    return poles, residues
end

function _moment(poles, residues, power::Int)
    s = 0.0
    for (ω, r) in zip(poles, residues)
        s += r * ω^power
    end
    return s
end

_reldiff(a, b) = abs(a - b) / max(abs(b), eps(Float64))

function _fit_diagnostics(J::Function, poles, residues, lo::Float64, hi::Float64,
                          nmodes::Int, grid::Symbol, block_index::Int)
    p2, r2 = _midpoint_modes(J, lo, hi, max(2nmodes, nmodes + 1), grid)
    m0 = _moment(poles, residues, 0)
    m1 = _moment(poles, residues, 1)
    ref0 = _moment(p2, r2, 0)
    ref1 = _moment(p2, r2, 1)
    return (;
        block_index,
        spectral_weight = m0,
        first_moment = m1,
        reference_nmodes = length(p2),
        rel_weight_change = _reldiff(m0, ref0),
        rel_first_moment_change = _reldiff(m1, ref1),
    )
end

function _block_anchor(block::Vector{Symbol}, i::Int, attach)
    if attach === nothing
        return first(block)
    elseif attach isa Symbol
        i == 1 || throw(ArgumentError("single Symbol attach is valid only for one block"))
        return attach
    elseif attach isa AbstractVector
        length(attach) >= i || throw(ArgumentError("attach vector has no entry for block $i"))
        attach[i] isa Symbol || throw(ArgumentError("attach entries must be Symbols"))
        return attach[i]
    elseif attach isa AbstractDict
        site = haskey(attach, i) ? attach[i] : get(attach, first(block), nothing)
        site isa Symbol || throw(ArgumentError("attach dictionary has no Symbol anchor for block $i"))
        return site
    elseif attach isa Function
        site = attach(block, i)
        site isa Symbol || throw(ArgumentError("attach function must return a Symbol"))
        return site
    else
        throw(ArgumentError("unsupported attach specification"))
    end
end

include("pes_pole_fitting.jl")
include("lorentzian_psd.jl")
include("anderson_real_poles.jl")
include("sparseir_adapter.jl")
include("precompile.jl")

end # module GraftImpurity
