# The scalar Cayley-tree bath mapping (root mode, binary branching z = 3,
# energy-split multibranch) follows Zhan, Chen, Fan, and Xiang,
# Phys. Rev. B 113, 195144 (2026), doi:10.1103/ycty-d5f9. The supplied local
# text gives the root construction and recursive diagonalize/partition/root
# procedure. That paper treats a single-band SIAM with scalar couplings only;
# ownership groups, virtual forest hubs, and all block-Cayley work are original
# GraftImpurity constructions and are audited here by exact one-particle data.

mutable struct _ScalarCayleyState
    group::Symbol
    labels::Vector{Symbol}
    vectors::Vector{Vector{ComplexF64}}
    edges::Vector{Pair{Symbol,Symbol}}
    forest_roots::Vector{Symbol}
    coupled_roots::Vector{Symbol}
    zero_hopping_components::Int
end

function _scalar_node!(state::_ScalarCayleyState, vector::Vector{ComplexF64};
                       forest_root::Bool=false, coupled_root::Bool=false)
    label = Symbol(:cayley_, state.group, :_, length(state.labels) + 1)
    push!(state.labels, label)
    push!(state.vectors, vector)
    forest_root && push!(state.forest_roots, label)
    coupled_root && push!(state.coupled_roots, label)
    return label
end

function _scalar_orthogonal_complement(basis::Matrix{ComplexF64},
                                       root::Vector{ComplexF64})
    expected = size(basis, 2) - 1
    expected <= 0 && return zeros(ComplexF64, size(basis, 1), 0)
    normalized_root = root / norm(root)
    coordinates = adjoint(basis) * normalized_root
    local_unitary = Matrix(qr(hcat(
        coordinates, Matrix{ComplexF64}(I, length(coordinates), length(coordinates)),
    )).Q)
    candidate = basis * local_unitary[:, 1]
    phase = adjoint(normalized_root) * candidate
    iszero(phase) && throw(ArgumentError(
        "Cayley basis completion lost its declared root direction",
    ))
    local_unitary[:, 1] .*= conj(phase) / abs(phase)
    complement = basis * local_unitary[:, 2:end]
    size(complement, 2) == expected || throw(ArgumentError(
        "Cayley basis completion lost a declared bath direction",
    ))
    return Matrix{ComplexF64}(complement)
end

function _scalar_balanced_subsets(amplitudes::Vector{ComplexF64}, branching::Int,
                                  zero_tolerance::Float64)
    active = findall(value -> abs(value) > zero_tolerance, amplitudes)
    isempty(active) && return Vector{Vector{Int}}()
    subset_count = min(branching, length(active))
    subsets = [Int[] for _ in 1:subset_count]
    loads = zeros(Float64, subset_count)
    ordered = sort(active; by=index -> (-abs(amplitudes[index]), index))
    for index in ordered
        destination = findmin(loads)[2]
        push!(subsets[destination], index)
        loads[destination] += abs2(amplitudes[index])
    end
    for index in findall(value -> abs(value) <= zero_tolerance, amplitudes)
        destination = findmin(loads)[2]
        push!(subsets[destination], index)
    end
    return subsets
end

function _scalar_spectators!(state::_ScalarCayleyState,
                             eigenbasis::Matrix{ComplexF64})
    for column in axes(eigenbasis, 2)
        _scalar_node!(state, copy(@view eigenbasis[:, column]); forest_root=true)
        state.zero_hopping_components += 1
    end
    return nothing
end

function _scalar_descendants!(state::_ScalarCayleyState, parent::Symbol,
                              parent_vector::Vector{ComplexF64},
                              subspace::Matrix{ComplexF64},
                              bath_hamiltonian::Matrix{ComplexF64},
                              kernel::CayleyTreeKernel)
    size(subspace, 2) == 0 && return nothing
    decomposition = eigen(Hermitian(adjoint(subspace) * bath_hamiltonian * subspace))
    eigenbasis = subspace * Matrix{ComplexF64}(decomposition.vectors)
    amplitudes = Vector{ComplexF64}(
        vec(adjoint(parent_vector) * bath_hamiltonian * eigenbasis),
    )
    # This threshold separates floating-point zero from a physical small edge.
    # `tree_tolerance` remains a report/audit criterion and never prunes data.
    roundoff_tolerance = 128 * eps(Float64) *
                         max(norm(bath_hamiltonian), 1.0) * size(subspace, 2)
    subsets = _scalar_balanced_subsets(amplitudes, kernel.branching,
                                       roundoff_tolerance)
    isempty(subsets) && return _scalar_spectators!(state, eigenbasis)

    for subset in subsets
        strength = norm(amplitudes[subset])
        if strength <= roundoff_tolerance
            _scalar_spectators!(state, Matrix{ComplexF64}(eigenbasis[:, subset]))
            continue
        end
        child_vector = eigenbasis[:, subset] * conj.(amplitudes[subset]) / strength
        child = _scalar_node!(state, child_vector)
        push!(state.edges, parent => child)
        child_subspace = _scalar_orthogonal_complement(
            eigenbasis[:, subset], child_vector,
        )
        _scalar_descendants!(state, child, child_vector, child_subspace,
                             bath_hamiltonian, kernel)
    end
    return nothing
end

_scalar_root_subsets(::BalancedCayleyPartitioner, energies::Vector{Float64}) =
    [collect(eachindex(energies))]

function _scalar_root_subsets(partitioner::EnergySplitCayleyPartitioner,
                              energies::Vector{Float64})
    lower = findall(energy -> energy < partitioner.cutoff, energies)
    upper = findall(energy -> energy >= partitioner.cutoff, energies)
    return [subset for subset in (lower, upper) if !isempty(subset)]
end

function _scalar_group_mapping(kernel::CayleyTreeKernel{ScalarCayley},
                               group::CayleyOwnershipGroup,
                               bath::DiscreteBath,
                               coupling::Matrix{ComplexF64})
    length(group.flavors) == 1 || throw(ArgumentError(
        "ScalarCayley group $(group.name) must declare exactly one flavor; " *
        "use BlockCayley for matrix-coupling groups",
    ))
    modes = collect(group.modes)
    energies = bath_orbitals(bath).energies[modes]
    bath_hamiltonian = Matrix{ComplexF64}(Diagonal(ComplexF64.(energies)))
    row = flavor_index(bath_layout(bath), only(group.flavors))
    values = copy(@view coupling[row, modes])
    state = _ScalarCayleyState(
        group.name, Symbol[], Vector{Vector{ComplexF64}}(), Pair{Symbol,Symbol}[],
        Symbol[], Symbol[], 0,
    )
    identity_basis = Matrix{ComplexF64}(I, length(modes), length(modes))
    for subset in _scalar_root_subsets(kernel.partitioner, energies)
        root_strength = norm(values[subset])
        support = identity_basis[:, subset]
        if iszero(root_strength)
            _scalar_spectators!(state, support)
            continue
        end
        root_vector = zeros(ComplexF64, length(modes))
        root_vector[subset] .= conj.(values[subset]) / root_strength
        root = _scalar_node!(state, root_vector;
                             forest_root=true, coupled_root=true)
        root_subspace = _scalar_orthogonal_complement(support, root_vector)
        _scalar_descendants!(state, root, root_vector, root_subspace,
                             bath_hamiltonian, kernel)
    end
    length(state.vectors) == length(modes) || throw(ArgumentError(
        "scalar Cayley mapping did not retain every canonical bath mode",
    ))
    return (; modes, state, transform=hcat(state.vectors...))
end

function _scalar_tree_sparsity_error(bath_hamiltonian::Matrix{ComplexF64},
                                     edges::Vector{Pair{Symbol,Symbol}},
                                     label_index::Dict{Symbol,Int})
    allowed = falses(size(bath_hamiltonian))
    for edge in edges
        parent = label_index[edge.first]
        child = label_index[edge.second]
        allowed[parent, child] = true
        allowed[child, parent] = true
    end
    error2 = 0.0
    for column in axes(bath_hamiltonian, 2), row in axes(bath_hamiltonian, 1)
        row == column && continue
        allowed[row, column] && continue
        error2 += abs2(bath_hamiltonian[row, column])
    end
    return sqrt(error2)
end

function _scalar_mapping_report(kernel::CayleyTreeKernel{ScalarCayley},
                                bath::DiscreteBath,
                                canonical_coupling::Matrix{ComplexF64},
                                transformed_coupling::Matrix{ComplexF64},
                                transform::Matrix{ComplexF64},
                                transformed_hamiltonian::Matrix{ComplexF64},
                                edges::Vector{Pair{Symbol,Symbol}},
                                label_index::Dict{Symbol,Int},
                                root_indices::Vector{Int},
                                group_reports::Vector{CayleyGroupReport},
                                forest_roots::Vector{Symbol},
                                zero_hopping_components::Int,
                                virtual_hub::Bool,
                                elapsed_seconds::Float64)
    dimension = length(bath)
    unitary_error = norm(adjoint(transform) * transform -
                         Matrix{ComplexF64}(I, dimension, dimension))
    canonical_hamiltonian = Matrix{ComplexF64}(
        Diagonal(ComplexF64.(bath_orbitals(bath).energies)),
    )
    spectrum_error = norm(
        sort(real.(eigvals(Hermitian(transformed_hamiltonian)))) -
        sort(real.(eigvals(Hermitian(canonical_hamiltonian)))),
    )
    hybridization_error = maximum(
        norm(_cayley_hybridization(canonical_coupling, canonical_hamiltonian, point) -
             _cayley_hybridization(transformed_coupling, transformed_hamiltonian, point))
        for point in kernel.validation_points;
        init=0.0,
    )
    tree_sparsity_error = _scalar_tree_sparsity_error(
        transformed_hamiltonian, edges, label_index,
    )
    root_mask = falses(dimension)
    root_mask[root_indices] .= true
    root_coupling_residual = norm(transformed_coupling[:, .!root_mask])
    approximate = tree_sparsity_error >
                  kernel.tree_tolerance * max(norm(transformed_hamiltonian), 1.0) ||
                  root_coupling_residual >
                  kernel.tree_tolerance * max(norm(canonical_coupling), 1.0)
    return BathMappingReport(
        unitarity_error=unitary_error,
        spectrum_error=spectrum_error,
        hybridization_error=hybridization_error,
        tree_sparsity_error=tree_sparsity_error,
        tree_tolerance=kernel.tree_tolerance,
        rank_tolerance=kernel.rank_tolerance,
        root_coupling_residual=root_coupling_residual,
        tree_connected=length(forest_roots) == 1,
        virtual_hub=virtual_hub,
        zero_hopping_components=zero_hopping_components,
        groups=group_reports,
        validation_points=kernel.validation_points,
        timing_seconds=elapsed_seconds,
        approximate=approximate,
        experimental=true,
    )
end

"""
    map_bath(kernel::CayleyTreeKernel{ScalarCayley}, bath::DiscreteBath)
        -> CayleyMappingResult with `mapped::ScalarCayleyBath`

Perform an explicit ownership-group-local scalar bath-only unitary mapping. It
retains every canonical bath mode and the complete transformed coupling matrix;
dark components become explicit virtual-hub forest roots rather than deleted
or reported as nonzero Cayley hoppings.
"""
function map_bath(kernel::CayleyTreeKernel{ScalarCayley}, bath::DiscreteBath)
    kernel.rank_tolerance === nothing || throw(ArgumentError(
        "rank_tolerance applies only to the BlockCayley route",
    ))
    started = time_ns()
    canonical_coupling = _cayley_coupling_matrix(bath)
    _validate_cayley_groups(kernel, bath, canonical_coupling)
    mappings = [_scalar_group_mapping(kernel, group, bath, canonical_coupling)
                for group in kernel.groups]
    dimension = length(bath)
    transform = zeros(ComplexF64, dimension, dimension)
    sites = Symbol[]
    physical_edges = Pair{Symbol,Symbol}[]
    forest_roots = Symbol[]
    coupled_roots = Tuple{Symbol,Symbol}[]
    group_reports = CayleyGroupReport[]
    zero_hopping_components = 0
    column = 0
    for (group, mapping) in zip(kernel.groups, mappings)
        local_dimension = length(mapping.modes)
        columns = (column + 1):(column + local_dimension)
        transform[mapping.modes, columns] .= mapping.transform
        append!(sites, mapping.state.labels)
        append!(physical_edges, mapping.state.edges)
        append!(forest_roots, mapping.state.forest_roots)
        append!(coupled_roots, ((group.name, label)
                                for label in mapping.state.coupled_roots))
        push!(group_reports, CayleyGroupReport(
            group.name, group.modes, group.flavors,
            fill(1, length(mapping.state.coupled_roots)); scalar=true,
            full_root_rank=length(mapping.state.coupled_roots),
            retained_root_rank=length(mapping.state.coupled_roots),
            zero_hopping_components=mapping.state.zero_hopping_components,
        ))
        zero_hopping_components += mapping.state.zero_hopping_components
        column += local_dimension
    end
    column == dimension || throw(ArgumentError(
        "scalar Cayley mapping failed to assign every transformed column",
    ))
    label_index = Dict(label => index for (index, label) in enumerate(sites))
    virtual_hub = length(forest_roots) > 1
    topology_root = virtual_hub ? :cayley_hub : only(forest_roots)
    topology_edges = copy(physical_edges)
    if virtual_hub
        prepend!(topology_edges,
                 Pair{Symbol,Symbol}[:cayley_hub => root for root in forest_roots])
    end
    topology = TreeTopology(topology_root, topology_edges)
    canonical_hamiltonian = Matrix{ComplexF64}(
        Diagonal(ComplexF64.(bath_orbitals(bath).energies)),
    )
    transformed_hamiltonian = adjoint(transform) * canonical_hamiltonian * transform
    transformed_coupling = canonical_coupling * transform
    edges = ScalarCayleyEdge[
        ScalarCayleyEdge(edge.first, edge.second,
                          transformed_hamiltonian[label_index[edge.first],
                                                 label_index[edge.second]])
        for edge in physical_edges
    ]
    groups_by_name = Dict(group.name => group for group in kernel.groups)
    roots = ScalarCayleyRoot[]
    for (group_name, site) in coupled_roots
        group = groups_by_name[group_name]
        flavor = only(group.flavors)
        row = flavor_index(bath_layout(bath), flavor)
        push!(roots, ScalarCayleyRoot(
            group_name, flavor, site, transformed_coupling[row, label_index[site]],
        ))
    end
    root_indices = Int[label_index[site] for (_, site) in coupled_roots]
    elapsed_seconds = (time_ns() - started) / 1e9
    report = _scalar_mapping_report(
        kernel, bath, canonical_coupling, transformed_coupling, transform,
        transformed_hamiltonian, physical_edges, label_index, root_indices,
        group_reports, forest_roots, zero_hopping_components, virtual_hub,
        elapsed_seconds,
    )
    mapped = ScalarCayleyBath(
        bath, topology, sites, Float64.(real.(diag(transformed_hamiltonian))),
        edges, roots, transformed_hamiltonian, transformed_coupling,
    )
    return CayleyMappingResult(bath, mapped, transform, report)
end
