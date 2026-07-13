"""
    AndersonRealPoles(fit::PESPoleFit, partition::Partition;
                      atol=0, rtol=sqrt(eps()))

Finite Hermitian star-bath parameters for a zero-temperature Anderson
impurity. `fit` must be a fermionic real-pole PES fit with PSD residues. The
single supported partition block fixes the row ordering of every residue.
Each PSD residue is factorized as `R_k = sum(v * v')`; a rank-`r` residue
therefore creates `r` bath orbitals at the same real energy.
"""
struct AndersonRealPoles{P<:Partition,D<:NamedTuple} <: BathParametrization
    energies::Vector{Float64}
    couplings::Vector{Vector{ComplexF64}}
    partition::P
    pole_indices::Vector{Int}
    diagnostics::D

    function AndersonRealPoles(energies::AbstractVector{<:Real}, couplings,
                               partition::P,
                               pole_indices::AbstractVector{<:Integer},
                               diagnostics::D) where {P<:Partition,D<:NamedTuple}
        length(energies) == length(couplings) == length(pole_indices) ||
            throw(ArgumentError("AndersonRealPoles needs one coupling and pole index per energy"))
        blocks = partition.blocks
        length(blocks) == 1 ||
            throw(ArgumentError("AndersonRealPoles currently supports exactly one partition block"))
        impurity_sites = only(blocks)
        isempty(impurity_sites) &&
            throw(ArgumentError("AndersonRealPoles needs at least one impurity site"))
        epsilons = Float64.(energies)
        all(isfinite, epsilons) ||
            throw(ArgumentError("AndersonRealPoles energies must be finite and real"))
        vectors = Vector{ComplexF64}[]
        for (j, coupling) in enumerate(couplings)
            length(coupling) == length(impurity_sites) ||
                throw(DimensionMismatch("AndersonRealPoles coupling $j has the wrong impurity dimension"))
            vector = ComplexF64.(coupling)
            all(z -> isfinite(real(z)) && isfinite(imag(z)), vector) ||
                throw(ArgumentError("AndersonRealPoles coupling $j must be finite"))
            push!(vectors, vector)
        end
        indices = Int.(pole_indices)
        all(index -> index > 0, indices) ||
            throw(ArgumentError("AndersonRealPoles pole indices must be positive"))
        return new{P,D}(epsilons, vectors, partition, indices, diagnostics)
    end
end

function AndersonRealPoles(fit::PESPoleFit, partition::Partition;
                           atol::Real=0.0,
                           rtol::Real=sqrt(eps(Float64)))
    fit.statistics == :fermion ||
        throw(ArgumentError("AndersonRealPoles requires a fermionic PESPoleFit"))
    blocks = partition.blocks
    length(blocks) == 1 ||
        throw(ArgumentError("AndersonRealPoles currently supports exactly one partition block"))
    impurity_sites = only(blocks)
    size(first(fit.weights), 1) == length(impurity_sites) ||
        throw(DimensionMismatch("PES residue dimension must match the partition block"))
    orbitals = bath_orbitals(fit; atol, rtol)
    diagnostics = (;
        source = :pes_real_poles,
        source_pole_count = length(fit),
        bath_orbital_count = length(orbitals.energies),
        residue_constraint = :positive_semidefinite,
        statistics = :fermion,
        fit_diagnostics = fit.diagnostics,
    )
    return AndersonRealPoles(orbitals.energies, orbitals.couplings,
                             partition, orbitals.pole_indices, diagnostics)
end

Base.length(bath::AndersonRealPoles) = length(bath.energies)
couplings(bath::AndersonRealPoles) = copy.(bath.couplings)

"""
    mount_bath(topo, bath::AndersonRealPoles;
               prefix=:fbath, attach=:dominant)

Mount one spinless graded fermion site per real-pole bath orbital. The default
`:dominant` placement attaches each bath site to the impurity orbital with the
largest coupling magnitude. A single impurity-site `Symbol`, one site per
mode, or a callback `(coupling, mode_index) -> impurity_site` may be supplied.
"""
function mount_bath(topo::TreeTopology, bath::AndersonRealPoles;
                    prefix::Symbol=:fbath, attach=:dominant)
    impurity_sites = only(bath.partition.blocks)
    for site in impurity_sites
        _anderson_require_topology_site(topo, site, "impurity")
    end
    top = topo
    sites = Symbol[]
    anchors = Symbol[]
    for (k, coupling) in enumerate(bath.couplings)
        anchor = _anderson_anchor(attach, coupling, k, impurity_sites)
        _anderson_require_topology_site(top, anchor, "attachment")
        local_prefix = Symbol(prefix, :_, k, :_)
        site = Symbol(local_prefix, 1)
        try
            nodeindex(top, site)
        catch err
            err isa KeyError || rethrow()
        else
            throw(ArgumentError("Anderson bath site $site already exists in the topology"))
        end
        top = mount_chain(top, anchor, 1; prefix=local_prefix)
        push!(sites, site)
        push!(anchors, anchor)
    end
    return (; topology=top, sites, anchors)
end

"""
    AndersonBath(fit::PESPoleFit, partition::Partition;
                 topology, phys=nothing, ops=fermion_ops_z2(),
                 prefix=:fbath, attach=:dominant, atol=0,
                 rtol=sqrt(eps()))

Lower a fermionic PSD real-pole fit to a finite graded Anderson star. Each
partition entry is one spin-orbital site with local space `ops.P`. The result
contains the mounted topology, physical-space dictionary, bath and
hybridization `OpSum`, and the explicit real-pole parameters.
"""
struct AndersonBath{B<:AndersonRealPoles,P<:AbstractDict,
                    O<:NamedTuple,D<:NamedTuple}
    real_poles::B
    topology::TreeTopology
    phys::P
    impurity_sites::Vector{Symbol}
    bath_sites::Vector{Symbol}
    anchors::Vector{Symbol}
    H::OpSum
    ops::O
    diagnostics::D
end

function AndersonBath(fit::PESPoleFit, partition::Partition;
                      topology::TreeTopology, phys=nothing,
                      ops::NamedTuple=fermion_ops_z2(),
                      prefix::Symbol=:fbath, attach=:dominant,
                      atol::Real=0.0,
                      rtol::Real=sqrt(eps(Float64)))
    _anderson_validate_ops(ops)
    bath = AndersonRealPoles(fit, partition; atol, rtol)
    impurity_sites = only(partition.blocks)
    base_phys = _anderson_physical_spaces(
        phys, impurity_sites, topology, ops.P)
    mounted = mount_bath(topology, bath; prefix, attach)
    allunique(vcat(collect(keys(base_phys)), mounted.sites)) ||
        throw(ArgumentError("Anderson bath-site labels collide with existing physical sites"))
    merged_phys = copy(base_phys)
    for site in mounted.sites
        merged_phys[site] = ops.P
    end
    H = _anderson_bath_opsum(bath, impurity_sites, mounted.sites, ops)
    diagnostics = (;
        representation = :finite_real_pole_star,
        statistics = :fermion,
        temperature = 0.0,
        impurity_orbital_count = length(impurity_sites),
        bath_orbital_count = length(bath),
        attachment = attach,
        residue_constraint = :positive_semidefinite,
    )
    return AndersonBath(bath, mounted.topology, merged_phys,
                        copy(impurity_sites), mounted.sites, mounted.anchors,
                        H, ops, diagnostics)
end

"""
    solve(problem::AndersonBath, H_loc::OpSum=OpSum();
          psi0, observables=(;), times=nothing, evolver=nothing,
          taus=nothing, beta_eff=nothing, thermal_evolver=nothing,
          thermal_nsteps=40, thermal_prop_nsteps=nothing,
          trunc=TruncationScheme(), nsweeps=10, tol=1e-10,
          krylovdim=20, verbose=true)

Solve a finite zero-temperature real-pole Anderson star with two-site DMRG.
`psi0` is required and is updated in place, enabling DMFT warm starts. Named
`OpSum` observables are evaluated after convergence. `times` plus `evolver`
request zero-temperature real-time particle/hole series from the ground
state. `taus` requests imaginary-time series and therefore additionally
requires a finite `beta_eff` plus `thermal_evolver`; this path uses thermal
purification and never changes the real-time temperature. No Fourier transform
or self-energy is implicitly manufactured.
"""
function solve(problem::AndersonBath, H_loc::OpSum=OpSum();
               psi0::TTNS, observables::NamedTuple=(;),
               times=nothing, evolver=nothing,
               taus=nothing, beta_eff=nothing, thermal_evolver=nothing,
               thermal_nsteps::Int=40,
               thermal_prop_nsteps::Union{Nothing,Int}=nothing,
               trunc::TruncationScheme=TruncationScheme(),
               nsweeps::Int=10, tol::Float64=1e-10,
               krylovdim::Int=20, verbose::Bool=true)
    topology(psi0) == problem.topology ||
        throw(ArgumentError("solve: psi0 topology does not match the mounted Anderson bath"))
    (times === nothing) == (evolver === nothing) ||
        throw(ArgumentError("solve: times and evolver must be supplied together"))
    _anderson_validate_imaginary_time(
        taus, beta_eff, thermal_evolver, thermal_nsteps,
        thermal_prop_nsteps)
    total_H = H_loc + problem.H
    Httno = ttno_from_opsum(total_H, problem.topology, problem.phys;
                            hermitian=true)
    state, energies = dmrg2!(psi0, Httno; trunc, nsweeps, tol,
                             krylovdim, verbose)
    normalize!(state)
    energy = real(expect(state, Httno))
    measured = _anderson_observables(
        observables, state, problem.topology, problem.phys)
    real_time = times === nothing ? nothing :
        _anderson_real_time_correlators(
            problem, state, energy, Httno, times, evolver)
    imaginary_time = taus === nothing ? nothing :
        _anderson_imaginary_time_correlators(
            problem, total_H, taus, Float64(beta_eff), thermal_evolver,
            thermal_nsteps, thermal_prop_nsteps)
    return (;
        state,
        energy,
        energies,
        observables = measured,
        real_time,
        imaginary_time,
        H = total_H,
        Httno,
        problem,
    )
end

function _anderson_require_topology_site(topo::TreeTopology, site::Symbol,
                                         role::AbstractString)
    try
        nodeindex(topo, site)
    catch err
        err isa KeyError || rethrow()
        throw(ArgumentError("Anderson $role site $site is not present in the topology"))
    end
    return nothing
end

function _anderson_anchor(attach, coupling, mode_index::Int, impurity_sites)
    anchor = if attach === :dominant
        impurity_sites[argmax(abs.(coupling))]
    elseif attach isa Symbol
        attach
    elseif attach isa AbstractVector
        length(attach) == 0 &&
            throw(ArgumentError("Anderson attach vector may not be empty"))
        length(attach) >= mode_index ||
            throw(DimensionMismatch("Anderson attach vector needs one site per bath mode"))
        attach[mode_index]
    elseif attach isa Function
        attach(coupling, mode_index)
    else
        throw(ArgumentError("unsupported Anderson attachment specification"))
    end
    anchor isa Symbol ||
        throw(ArgumentError("Anderson attachment must resolve to a Symbol"))
    anchor in impurity_sites ||
        throw(ArgumentError("Anderson bath modes must attach to a declared impurity site"))
    return anchor
end

function _anderson_validate_ops(ops::NamedTuple)
    for name in (:C, :Cd, :N, :P)
        hasproperty(ops, name) ||
            throw(ArgumentError("Anderson fermion operators need field $name"))
    end
    return nothing
end

function _anderson_physical_spaces(phys, impurity_sites, topo, space)
    spaces = Dict{Symbol,typeof(space)}()
    if phys === nothing
        for site in impurity_sites
            _anderson_require_topology_site(topo, site, "impurity")
            spaces[site] = space
        end
        return spaces
    end
    for (site, value) in pairs(phys)
        value == space ||
            throw(ArgumentError("AndersonBath currently supports only ops.P fermion physical spaces"))
        spaces[site] = value
    end
    for site in impurity_sites
        _anderson_require_topology_site(topo, site, "impurity")
        haskey(spaces, site) ||
            throw(ArgumentError("Anderson impurity site $site is missing from phys"))
    end
    return spaces
end

function _anderson_bath_opsum(bath, impurity_sites, bath_sites, ops)
    H = OpSum()
    for k in eachindex(bath.energies)
        bath_site = bath_sites[k]
        energy = bath.energies[k]
        iszero(energy) ||
            (H += Term(energy, SiteOp(bath_site, :N, ops.N)))
        for (impurity_site, coupling) in
                zip(impurity_sites, bath.couplings[k])
            iszero(coupling) && continue
            H += Term(coupling,
                      SiteOp(impurity_site, :Cd, ops.Cd),
                      SiteOp(bath_site, :C, ops.C))
            H += Term(conj(coupling),
                      SiteOp(impurity_site, :C, ops.C),
                      SiteOp(bath_site, :Cd, ops.Cd))
        end
    end
    return H
end

function _anderson_observables(observables::NamedTuple, state, topo, phys)
    pairs_out = Pair{Symbol,Any}[]
    for (name, observable) in pairs(observables)
        observable isa OpSum ||
            throw(ArgumentError("Anderson observable $name must be an OpSum"))
        operator = ttno_from_opsum(observable, topo, phys; hermitian=false)
        push!(pairs_out, name => expect(state, operator))
    end
    return (; pairs_out...)
end

function _anderson_real_time_correlators(problem, state, energy, Httno,
                                         times, evolver)
    impurity_sites = problem.impurity_sites
    dimension = length(impurity_sites)
    particle = Matrix{Any}(undef, dimension, dimension)
    hole = Matrix{Any}(undef, dimension, dimension)
    for a in 1:dimension, b in 1:dimension
        particle[a, b] = correlator_series(
            state, energy,
            impurity_sites[a] => problem.ops.C,
            impurity_sites[b] => problem.ops.Cd,
            times; H=Httno, evolver,
            metadata=(; channel=:particle, a, b))
        hole[a, b] = correlator_series(
            state, energy,
            impurity_sites[a] => problem.ops.Cd,
            impurity_sites[b] => problem.ops.C,
            times; H=Httno, evolver,
            metadata=(; channel=:hole, a, b))
    end
    return (;
        times=collect(times),
        particle,
        hole,
        temperature=0.0,
        convention=:raw_correlator,
    )
end

function _anderson_validate_imaginary_time(taus, beta_eff, thermal_evolver,
                                           thermal_nsteps,
                                           thermal_prop_nsteps)
    if taus === nothing
        beta_eff === nothing ||
            throw(ArgumentError("solve: beta_eff is only valid with imaginary-time taus"))
        thermal_evolver === nothing ||
            throw(ArgumentError("solve: thermal_evolver is only valid with imaginary-time taus"))
        thermal_prop_nsteps === nothing ||
            throw(ArgumentError("solve: thermal_prop_nsteps is only valid with imaginary-time taus"))
        return nothing
    end
    beta_eff isa Real && isfinite(beta_eff) && beta_eff > 0 ||
        throw(ArgumentError("solve: imaginary-time taus require finite positive beta_eff"))
    thermal_evolver === nothing &&
        throw(ArgumentError("solve: imaginary-time taus require thermal_evolver"))
    thermal_nsteps > 0 ||
        throw(ArgumentError("solve: thermal_nsteps must be positive"))
    thermal_prop_nsteps === nothing || thermal_prop_nsteps > 0 ||
        throw(ArgumentError("solve: thermal_prop_nsteps must be positive"))
    tau_values = Float64.(taus)
    all(tau -> 0 <= tau <= beta_eff, tau_values) ||
        throw(ArgumentError("solve: imaginary times must lie in [0, beta_eff]"))
    return nothing
end

function _anderson_imaginary_time_correlators(
        problem, total_H, taus, beta_eff::Float64, thermal_evolver,
        thermal_nsteps::Int, thermal_prop_nsteps)
    tau_values = Float64.(taus)
    thermal_problem = purification_problem(
        total_H, problem.topology, problem.phys; hermitian=true)
    save_betas = sort(unique(vcat(beta_eff .- tau_values, [beta_eff])))
    trajectory = thermalize(
        Purified(), thermal_problem, beta_eff;
        evolver=thermal_evolver, nsteps=thermal_nsteps,
        save_betas)
    impurity_sites = problem.impurity_sites
    dimension = length(impurity_sites)
    particle = Matrix{Any}(undef, dimension, dimension)
    hole = Matrix{Any}(undef, dimension, dimension)
    for a in 1:dimension, b in 1:dimension
        particle[a, b] = thermal_correlator(
            Purified(), thermal_problem,
            impurity_sites[a] => problem.ops.C,
            impurity_sites[b] => problem.ops.Cd,
            beta_eff, tau_values; evolver=thermal_evolver,
            trajectory, prop_nsteps=thermal_prop_nsteps,
            metadata=(; channel=:particle, a, b, beta_eff))
        hole[a, b] = thermal_correlator(
            Purified(), thermal_problem,
            impurity_sites[a] => problem.ops.Cd,
            impurity_sites[b] => problem.ops.C,
            beta_eff, tau_values; evolver=thermal_evolver,
            trajectory, prop_nsteps=thermal_prop_nsteps,
            metadata=(; channel=:hole, a, b, beta_eff))
    end
    return (;
        taus=tau_values,
        beta_eff,
        particle,
        hole,
        convention=:raw_correlator,
        trajectory,
        problem=thermal_problem,
    )
end
