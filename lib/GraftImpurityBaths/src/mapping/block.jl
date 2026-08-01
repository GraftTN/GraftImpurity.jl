# Block-Cayley is a GraftImpurity construction for declared full-matrix
# hybridization groups. It follows the scalar route's group-local/forest
# invariants, but does not claim a scalar Cayley graph after block flattening.

mutable struct _BlockCayleyState
    group::Symbol
    labels::Vector{Symbol}
    bases::Vector{Matrix{ComplexF64}}
    edges::Vector{Pair{Symbol,Symbol}}
    forest_roots::Vector{Symbol}
    coupled_roots::Vector{Symbol}
    full_root_rank::Int
    retained_root_rank::Int
    root_rank_thresholds::Vector{Float64}
    root_singular_values::Vector{Vector{Float64}}
    rank_reduction_error2::Float64
    zero_hopping_components::Int
    rank_reduced::Bool
end

function _block_node!(state::_BlockCayleyState, basis::Matrix{ComplexF64};
                      forest_root::Bool=false, coupled_root::Bool=false)
    size(basis, 2) > 0 || throw(ArgumentError(
        "block Cayley nodes must retain at least one bath direction",
    ))
    label = Symbol(:block_cayley_, state.group, :_, length(state.labels) + 1)
    push!(state.labels, label)
    push!(state.bases, basis)
    forest_root && push!(state.forest_roots, label)
    coupled_root && push!(state.coupled_roots, label)
    return label
end

function _block_machine_rank_tolerance(singular_values::AbstractVector{<:Real},
                                       rows::Int, columns::Int)
    scale = isempty(singular_values) ? 0.0 : maximum(singular_values)
    return 128 * eps(Float64) * max(rows, columns) * scale
end

function _block_svd_ranks(singular_values::AbstractVector{<:Real}, rows::Int,
                          columns::Int, kernel::CayleyTreeKernel{BlockCayley})
    machine_tolerance = _block_machine_rank_tolerance(singular_values, rows, columns)
    explicit_tolerance = kernel.rank_tolerance === nothing ? 0.0 :
                         kernel.rank_tolerance *
                         (isempty(singular_values) ? 0.0 : maximum(singular_values))
    threshold = max(machine_tolerance, explicit_tolerance)
    full_rank = count(value -> value > machine_tolerance, singular_values)
    retained_rank = count(value -> value > threshold, singular_values)
    return (; full_rank, retained_rank, threshold)
end

function _block_record_rank_reduction!(state::_BlockCayleyState,
                                       singular_values::Vector{Float64},
                                       ranks)
    ranks.retained_rank < ranks.full_rank || return nothing
    state.rank_reduced = true
    state.rank_reduction_error2 += sum(abs2,
                                       @view singular_values[
                                           (ranks.retained_rank + 1):ranks.full_rank,
                                       ])
    return nothing
end

function _block_orthogonal_complement(basis::Matrix{ComplexF64},
                                      node_basis::Matrix{ComplexF64})
    expected = size(basis, 2) - size(node_basis, 2)
    expected <= 0 && return zeros(ComplexF64, size(basis, 1), 0)
    coordinates = adjoint(basis) * node_basis
    local_unitary = Matrix(qr(hcat(
        coordinates,
        Matrix{ComplexF64}(I, size(basis, 2), size(basis, 2)),
    )).Q)
    complement = basis * local_unitary[:, (size(node_basis, 2) + 1):end]
    size(complement, 2) == expected || throw(ArgumentError(
        "block Cayley basis completion lost a declared bath direction",
    ))
    return Matrix{ComplexF64}(complement)
end

function _block_balanced_subsets(weights::Vector{Float64}, branching::Int,
                                 roundoff_tolerance::Float64)
    active = findall(weight -> weight > roundoff_tolerance, weights)
    isempty(active) && return Vector{Vector{Int}}()
    subset_count = min(branching, length(active))
    subsets = [Int[] for _ in 1:subset_count]
    loads = zeros(Float64, subset_count)
    ordered = sort(active; by=index -> (-weights[index], index))
    for index in ordered
        destination = findmin(loads)[2]
        push!(subsets[destination], index)
        loads[destination] += weights[index]
    end
    for index in findall(weight -> weight <= roundoff_tolerance, weights)
        destination = findmin(loads)[2]
        push!(subsets[destination], index)
    end
    return subsets
end

function _block_spectator!(state::_BlockCayleyState,
                           basis::Matrix{ComplexF64}; zero_hopping::Bool)
    _block_node!(state, basis; forest_root=true)
    zero_hopping && (state.zero_hopping_components += 1)
    return nothing
end

function _block_descendants!(state::_BlockCayleyState, parent::Symbol,
                             parent_basis::Matrix{ComplexF64},
                             subspace::Matrix{ComplexF64},
                             bath_hamiltonian::Matrix{ComplexF64},
                             kernel::CayleyTreeKernel{BlockCayley})
    size(subspace, 2) == 0 && return nothing
    decomposition = eigen(Hermitian(adjoint(subspace) * bath_hamiltonian * subspace))
    eigenbasis = subspace * Matrix{ComplexF64}(decomposition.vectors)
    coupling = adjoint(parent_basis) * bath_hamiltonian * eigenbasis
    weights = Float64.(vec(real.(sum(abs2, coupling; dims=1))))
    roundoff_tolerance = 128 * eps(Float64) *
                         max(norm(bath_hamiltonian), 1.0) * size(subspace, 2)
    subsets = _block_balanced_subsets(weights, kernel.branching, roundoff_tolerance)
    isempty(subsets) && return _block_spectator!(
        state, eigenbasis; zero_hopping=true,
    )

    for subset in subsets
        child_space = Matrix{ComplexF64}(eigenbasis[:, subset])
        parent_coupling = adjoint(parent_basis) * bath_hamiltonian * child_space
        decomposition = svd(adjoint(parent_coupling); full=true)
        singular_values = Float64.(decomposition.S)
        ranks = _block_svd_ranks(
            singular_values, size(parent_coupling, 2),
            size(parent_coupling, 1), kernel,
        )
        _block_record_rank_reduction!(state, singular_values, ranks)
        if ranks.retained_rank == 0
            _block_spectator!(state, child_space;
                              zero_hopping=ranks.full_rank == 0)
            continue
        end
        unitary = Matrix{ComplexF64}(decomposition.U)
        child_basis = child_space * unitary[:, 1:ranks.retained_rank]
        child = _block_node!(state, Matrix{ComplexF64}(child_basis))
        push!(state.edges, parent => child)
        child_subspace = _block_orthogonal_complement(child_space, child_basis)
        _block_descendants!(state, child, Matrix{ComplexF64}(child_basis),
                             child_subspace, bath_hamiltonian, kernel)
    end
    return nothing
end

_block_root_subsets(::BalancedCayleyPartitioner, energies::Vector{Float64}) =
    [collect(eachindex(energies))]

function _block_root_subsets(partitioner::EnergySplitCayleyPartitioner,
                             energies::Vector{Float64})
    lower = findall(energy -> energy < partitioner.cutoff, energies)
    upper = findall(energy -> energy >= partitioner.cutoff, energies)
    return [subset for subset in (lower, upper) if !isempty(subset)]
end

function _block_group_mapping(kernel::CayleyTreeKernel{BlockCayley},
                              group::CayleyOwnershipGroup,
                              bath::DiscreteBath,
                              coupling::Matrix{ComplexF64})
    length(group.flavors) >= 2 || throw(ArgumentError(
        "BlockCayley groups need at least two declared flavors; use ScalarCayley " *
        "for a one-flavor ownership group",
    ))
    modes = collect(group.modes)
    energies = bath_orbitals(bath).energies[modes]
    bath_hamiltonian = Matrix{ComplexF64}(Diagonal(ComplexF64.(energies)))
    rows = Int[flavor_index(bath_layout(bath), flavor) for flavor in group.flavors]
    values = coupling[rows, modes]
    state = _BlockCayleyState(
        group.name, Symbol[], Matrix{ComplexF64}[], Pair{Symbol,Symbol}[],
        Symbol[], Symbol[], 0, 0, Float64[], Vector{Vector{Float64}}(),
        0.0, 0, false,
    )
    identity_basis = Matrix{ComplexF64}(I, length(modes), length(modes))
    for subset in _block_root_subsets(kernel.partitioner, energies)
        root_space = Matrix{ComplexF64}(identity_basis[:, subset])
        decomposition = svd(values[:, subset]; full=true)
        singular_values = Float64.(decomposition.S)
        ranks = _block_svd_ranks(
            singular_values, size(values, 1), length(subset), kernel,
        )
        state.full_root_rank += ranks.full_rank
        state.retained_root_rank += ranks.retained_rank
        push!(state.root_rank_thresholds, ranks.threshold)
        push!(state.root_singular_values, singular_values)
        _block_record_rank_reduction!(state, singular_values, ranks)
        if ranks.retained_rank == 0
            _block_spectator!(state, root_space;
                              zero_hopping=ranks.full_rank == 0)
            continue
        end
        unitary = Matrix{ComplexF64}(adjoint(decomposition.Vt))
        root_basis = root_space * unitary[:, 1:ranks.retained_rank]
        root = _block_node!(state, Matrix{ComplexF64}(root_basis);
                            forest_root=true, coupled_root=true)
        root_subspace = _block_orthogonal_complement(root_space, root_basis)
        _block_descendants!(state, root, Matrix{ComplexF64}(root_basis),
                             root_subspace, bath_hamiltonian, kernel)
    end
    local_transform = hcat(state.bases...)
    size(local_transform, 2) == length(modes) || throw(ArgumentError(
        "block Cayley mapping did not retain every canonical bath mode",
    ))
    return (; modes, rows, state, transform=local_transform)
end

function _block_tree_sparsity_error(bath_hamiltonian::Matrix{ComplexF64},
                                    site_ranges::Dict{Symbol,UnitRange{Int}},
                                    edges::Vector{Pair{Symbol,Symbol}})
    allowed = falses(size(bath_hamiltonian))
    for range in values(site_ranges)
        allowed[range, range] .= true
    end
    for edge in edges
        parent = site_ranges[edge.first]
        child = site_ranges[edge.second]
        allowed[parent, child] .= true
        allowed[child, parent] .= true
    end
    error2 = 0.0
    for column in axes(bath_hamiltonian, 2), row in axes(bath_hamiltonian, 1)
        allowed[row, column] && continue
        error2 += abs2(bath_hamiltonian[row, column])
    end
    return sqrt(error2)
end

function _block_mapping_report(kernel::CayleyTreeKernel{BlockCayley},
                               bath::DiscreteBath,
                               canonical_coupling::Matrix{ComplexF64},
                               transformed_coupling::Matrix{ComplexF64},
                               transform::Matrix{ComplexF64},
                               transformed_hamiltonian::Matrix{ComplexF64},
                               site_ranges::Dict{Symbol,UnitRange{Int}},
                               edges::Vector{Pair{Symbol,Symbol}},
                               root_indices::Vector{Int},
                               group_reports::Vector{CayleyGroupReport},
                               forest_roots::Vector{Symbol},
                               zero_hopping_components::Int,
                               rank_reduced::Bool,
                               rank_reduction_residual::Float64,
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
    tree_sparsity_error = _block_tree_sparsity_error(
        transformed_hamiltonian, site_ranges, edges,
    )
    root_mask = falses(dimension)
    root_mask[root_indices] .= true
    root_coupling_residual = norm(transformed_coupling[:, .!root_mask])
    roundoff_residual = 128 * eps(Float64) *
                        max(norm(canonical_coupling), 1.0) *
                        max(size(canonical_coupling)...)
    approximate = rank_reduced ||
                  tree_sparsity_error >
                  kernel.tree_tolerance * max(norm(transformed_hamiltonian), 1.0) ||
                  root_coupling_residual > roundoff_residual
    return BathMappingReport(
        unitarity_error=unitary_error,
        spectrum_error=spectrum_error,
        hybridization_error=hybridization_error,
        tree_sparsity_error=tree_sparsity_error,
        tree_tolerance=kernel.tree_tolerance,
        rank_tolerance=kernel.rank_tolerance,
        root_coupling_residual=root_coupling_residual,
        rank_reduction_residual=rank_reduction_residual,
        tree_connected=length(forest_roots) == 1,
        virtual_hub=length(forest_roots) > 1,
        zero_hopping_components=zero_hopping_components,
        groups=group_reports,
        validation_points=kernel.validation_points,
        timing_seconds=elapsed_seconds,
        approximate=approximate,
        experimental=true,
    )
end

"""
    map_bath(kernel::CayleyTreeKernel{BlockCayley}, bath::DiscreteBath)
        -> CayleyMappingResult with `mapped::BlockCayleyBath`

Perform a caller-declared group-local block-Cayley transformation. Each root
uses the bath-space right singular vectors of its full matrix coupling. An
explicit rank cutoff can reduce the topology's root/edge blocks, but the result
retains complete transformed matrices and reports the residual as approximate.
"""
function map_bath(kernel::CayleyTreeKernel{BlockCayley}, bath::DiscreteBath)
    started = time_ns()
    canonical_coupling = _cayley_coupling_matrix(bath)
    _validate_cayley_groups(kernel, bath, canonical_coupling)
    mappings = [_block_group_mapping(kernel, group, bath, canonical_coupling)
                for group in kernel.groups]
    dimension = length(bath)
    transform = zeros(ComplexF64, dimension, dimension)
    sites = Symbol[]
    site_dimensions = Int[]
    site_ranges = Dict{Symbol,UnitRange{Int}}()
    physical_edges = Pair{Symbol,Symbol}[]
    forest_roots = Symbol[]
    coupled_roots = Tuple{Symbol,Symbol}[]
    group_columns = Dict{Symbol,UnitRange{Int}}()
    zero_hopping_components = 0
    rank_reduced = false
    rank_reduction_error2 = 0.0
    column = 0
    for (group, mapping) in zip(kernel.groups, mappings)
        local_dimension = length(mapping.modes)
        columns = (column + 1):(column + local_dimension)
        transform[mapping.modes, columns] .= mapping.transform
        group_columns[group.name] = columns
        local_column = first(columns)
        for (label, basis) in zip(mapping.state.labels, mapping.state.bases)
            width = size(basis, 2)
            site_ranges[label] = local_column:(local_column + width - 1)
            push!(sites, label)
            push!(site_dimensions, width)
            local_column += width
        end
        append!(physical_edges, mapping.state.edges)
        append!(forest_roots, mapping.state.forest_roots)
        append!(coupled_roots, ((group.name, label)
                                for label in mapping.state.coupled_roots))
        zero_hopping_components += mapping.state.zero_hopping_components
        rank_reduced |= mapping.state.rank_reduced
        rank_reduction_error2 += mapping.state.rank_reduction_error2
        column += local_dimension
    end
    column == dimension || throw(ArgumentError(
        "block Cayley mapping failed to assign every transformed column",
    ))
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
    edges = BlockCayleyEdge[
        BlockCayleyEdge(edge.first, edge.second,
                        Matrix{ComplexF64}(
                            transformed_hamiltonian[site_ranges[edge.first],
                                                   site_ranges[edge.second]],
                        ))
        for edge in physical_edges
    ]
    groups_by_name = Dict(group.name => group for group in kernel.groups)
    roots = BlockCayleyRoot[]
    root_indices = Int[]
    for (group_name, site) in coupled_roots
        group = groups_by_name[group_name]
        rows = Int[flavor_index(bath_layout(bath), flavor) for flavor in group.flavors]
        columns = site_ranges[site]
        append!(root_indices, columns)
        push!(roots, BlockCayleyRoot(
            group_name, group.flavors, site,
            Matrix{ComplexF64}(transformed_coupling[rows, columns]),
        ))
    end
    group_reports = CayleyGroupReport[]
    for (group, mapping) in zip(kernel.groups, mappings)
        rows = mapping.rows
        columns = group_columns[group.name]
        root_columns = Int[index for site in mapping.state.coupled_roots
                           for index in site_ranges[site]]
        root_mask = falses(length(columns))
        root_mask[root_columns .- first(columns) .+ 1] .= true
        residual = norm(transformed_coupling[rows, columns[.!root_mask]])
        dimensions = Int[length(site_ranges[site])
                         for site in mapping.state.coupled_roots]
        push!(group_reports, CayleyGroupReport(
            group.name, group.modes, group.flavors, dimensions;
            scalar=false,
            full_root_rank=mapping.state.full_root_rank,
            retained_root_rank=mapping.state.retained_root_rank,
            root_coupling_residual=residual,
            rank_reduction_residual=sqrt(mapping.state.rank_reduction_error2),
            root_rank_thresholds=mapping.state.root_rank_thresholds,
            root_singular_values=mapping.state.root_singular_values,
            rank_reduced=mapping.state.rank_reduced,
            zero_hopping_components=mapping.state.zero_hopping_components,
        ))
    end
    elapsed_seconds = (time_ns() - started) / 1e9
    report = _block_mapping_report(
        kernel, bath, canonical_coupling, transformed_coupling, transform,
        transformed_hamiltonian, site_ranges, physical_edges, root_indices,
        group_reports, forest_roots, zero_hopping_components, rank_reduced,
        sqrt(rank_reduction_error2),
        elapsed_seconds,
    )
    onsite = Matrix{ComplexF64}[
        Matrix{ComplexF64}(transformed_hamiltonian[site_ranges[site],
                                                   site_ranges[site]])
        for site in sites
    ]
    mapped = BlockCayleyBath(
        bath, topology, sites, site_dimensions, onsite, edges, roots,
        transformed_hamiltonian, transformed_coupling,
    )
    return CayleyMappingResult(bath, mapped, transform, report)
end
