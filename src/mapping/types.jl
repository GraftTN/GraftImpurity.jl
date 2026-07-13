abstract type AbstractCayleyRoute end

"""Typed route marker for scalar, independently owned Cayley trees."""
struct ScalarCayley <: AbstractCayleyRoute end

"""Typed route marker for full matrix-coupling block-Cayley trees."""
struct BlockCayley <: AbstractCayleyRoute end

abstract type AbstractCayleyPartitioner end

"""Deterministic coupling-weight-balanced recursive Cayley partitioning."""
struct BalancedCayleyPartitioner <: AbstractCayleyPartitioner end

"""
    EnergySplitCayleyPartitioner(cutoff=0.0)

Split a declared group's canonical bath modes into energies below and at/above
`cutoff` before constructing independent Cayley roots. Recursive descendants
use the deterministic balanced rule.
"""
struct EnergySplitCayleyPartitioner <: AbstractCayleyPartitioner
    cutoff::Float64

    function EnergySplitCayleyPartitioner(cutoff::Real=0.0)
        value = Float64(cutoff)
        isfinite(value) || throw(ArgumentError(
            "EnergySplitCayleyPartitioner cutoff must be finite",
        ))
        new(value)
    end
end

"""
    CayleyOwnershipGroup(name, modes, flavors)

An immutable caller-declared bath-only mixing domain. `modes` are canonical
`BathOrbitals` indices and `flavors` are the only impurity flavors whose
nonzero coupling support may occur in those columns. M5 validates the exact
cover and support against a `DiscreteBath`; it never infers this relation from
coupling magnitudes.
"""
struct CayleyOwnershipGroup
    name::Symbol
    modes::Tuple{Vararg{Int}}
    flavors::Tuple{Vararg{Symbol}}

    function CayleyOwnershipGroup(name::Symbol, modes::Tuple{Vararg{Int}},
                                  flavors::Tuple{Vararg{Symbol}},
                                  ::Val{:validated})
        new(name, modes, flavors)
    end
end

function CayleyOwnershipGroup(name::Symbol,
                              modes::AbstractVector{<:Integer},
                              flavors::AbstractVector{Symbol})
    isempty(String(name)) && throw(ArgumentError(
        "Cayley ownership group name must be nonempty",
    ))
    canonical_modes = Tuple(Int.(modes))
    isempty(canonical_modes) && throw(ArgumentError(
        "Cayley ownership groups need at least one canonical bath mode",
    ))
    all(mode -> mode > 0, canonical_modes) || throw(ArgumentError(
        "Cayley ownership group mode indices must be positive",
    ))
    allunique(canonical_modes) || throw(ArgumentError(
        "Cayley ownership group mode indices must be unique",
    ))
    canonical_flavors = Tuple(Symbol.(flavors))
    isempty(canonical_flavors) && throw(ArgumentError(
        "Cayley ownership groups need at least one declared flavor",
    ))
    allunique(canonical_flavors) || throw(ArgumentError(
        "Cayley ownership group flavors must be unique",
    ))
    return CayleyOwnershipGroup(name, canonical_modes, canonical_flavors,
                                Val(:validated))
end

"""
    CayleyTreeKernel(route, groups; branching=2, partitioner=BalancedCayleyPartitioner(),
                     rank_tolerance=nothing, tree_tolerance=1e-10,
                     validation_points=(0.5im, 1.0 + 0.5im))

Explicit experimental bath-Hamiltonian mapping kernel. The scalar and block
route markers select concrete result types by dispatch. Groups are mandatory:
the kernel never introduces an undeclared global bath rotation.
`tree_tolerance` is an audit threshold for the retained full transformed
Hamiltonian; it never authorizes physical hopping pruning.
"""
struct CayleyTreeKernel{R<:AbstractCayleyRoute,G<:Tuple,
                        P<:AbstractCayleyPartitioner} <: AbstractBathMappingKernel
    route::R
    groups::G
    branching::Int
    partitioner::P
    rank_tolerance::Union{Nothing,Float64}
    tree_tolerance::Float64
    validation_points::Tuple{Vararg{ComplexF64}}

    function CayleyTreeKernel(route::R, groups::G, branching::Int,
                              partitioner::P,
                              rank_tolerance::Union{Nothing,Float64},
                              tree_tolerance::Float64,
                              validation_points::Tuple{Vararg{ComplexF64}},
                              ::Val{:validated}) where {
                                  R<:AbstractCayleyRoute,G<:Tuple,
                                  P<:AbstractCayleyPartitioner}
        new{R,G,P}(route, groups, branching, partitioner, rank_tolerance,
                   tree_tolerance, validation_points)
    end
end

function CayleyTreeKernel(route::R,
                          groups::Union{Tuple,AbstractVector};
                          branching::Integer=2,
                          partitioner::P=BalancedCayleyPartitioner(),
                          rank_tolerance=nothing,
                          tree_tolerance::Real=1e-10,
                          validation_points=(0.5im, 1.0 + 0.5im)) where {
                              R<:AbstractCayleyRoute,P<:AbstractCayleyPartitioner}
    canonical_groups = Tuple(groups)
    isempty(canonical_groups) && throw(ArgumentError(
        "CayleyTreeKernel requires at least one explicit ownership group",
    ))
    all(group -> group isa CayleyOwnershipGroup, canonical_groups) ||
        throw(ArgumentError(
            "CayleyTreeKernel groups must be CayleyOwnershipGroup values",
        ))
    names = Tuple(group.name for group in canonical_groups)
    allunique(names) || throw(ArgumentError(
        "CayleyTreeKernel ownership group names must be unique",
    ))
    resolved_branching = Int(branching)
    resolved_branching >= 2 || throw(ArgumentError(
        "CayleyTreeKernel branching must be at least two",
    ))
    resolved_rank = rank_tolerance === nothing ? nothing : Float64(rank_tolerance)
    resolved_rank === nothing ||
        (isfinite(resolved_rank) && resolved_rank >= 0) || throw(ArgumentError(
            "CayleyTreeKernel rank_tolerance must be finite and nonnegative",
        ))
    resolved_tree = Float64(tree_tolerance)
    isfinite(resolved_tree) && resolved_tree > 0 || throw(ArgumentError(
        "CayleyTreeKernel tree_tolerance must be finite and positive",
    ))
    points = Tuple(ComplexF64.(validation_points))
    isempty(points) && throw(ArgumentError(
        "CayleyTreeKernel needs at least one hybridization validation point",
    ))
    all(point -> isfinite(real(point)) && isfinite(imag(point)) &&
                 !iszero(imag(point)), points) || throw(ArgumentError(
        "CayleyTreeKernel validation points must be finite and off the real axis",
    ))
    return CayleyTreeKernel(route, canonical_groups, resolved_branching,
                            partitioner, resolved_rank, resolved_tree, points,
                            Val(:validated))
end

"""One scalar mapped bath-hopping edge in parent-to-child topology order."""
struct ScalarCayleyEdge
    parent::Symbol
    child::Symbol
    hopping::ComplexF64
end

"""One scalar impurity-to-root coupling retained after mapping."""
struct ScalarCayleyRoot
    group::Symbol
    flavor::Symbol
    site::Symbol
    coupling::ComplexF64
end

"""One block mapped bath-hopping edge in parent-to-child topology order."""
struct BlockCayleyEdge
    parent::Symbol
    child::Symbol
    hopping::Matrix{ComplexF64}
end

"""One dense impurity-to-block-root coupling retained after mapping."""
struct BlockCayleyRoot
    group::Symbol
    flavors::Tuple{Vararg{Symbol}}
    site::Symbol
    coupling::Matrix{ComplexF64}
end

abstract type AbstractCayleyBath <: AbstractBathParametrization end

"""
Scalar mapped bath. `topology` includes a virtual `:cayley_hub` only when a
forest has several impurity-coupled or dark roots; `edges` contains physical
nonzero bath hoppings only and `bath_hamiltonian` remains the full transformed
one-particle matrix for audit.
"""
struct ScalarCayleyBath{C<:DiscreteBath} <: AbstractCayleyBath
    canonical::C
    topology::TreeTopology
    sites::Vector{Symbol}
    onsite::Vector{Float64}
    edges::Vector{ScalarCayleyEdge}
    roots::Vector{ScalarCayleyRoot}
    bath_hamiltonian::Matrix{ComplexF64}
    coupling_matrix::Matrix{ComplexF64}
end

"""
Truthful block-node Cayley representation for full matrix hybridization. Each
topology site can carry more than one one-particle bath mode; it is not a
scalar-site topology disguised as independent channels.
"""
struct BlockCayleyBath{C<:DiscreteBath} <: AbstractCayleyBath
    canonical::C
    topology::TreeTopology
    sites::Vector{Symbol}
    site_dimensions::Vector{Int}
    onsite::Vector{Matrix{ComplexF64}}
    edges::Vector{BlockCayleyEdge}
    roots::Vector{BlockCayleyRoot}
    bath_hamiltonian::Matrix{ComplexF64}
    coupling_matrix::Matrix{ComplexF64}

    function BlockCayleyBath(canonical::C, topology::TreeTopology,
                             sites::Vector{Symbol}, site_dimensions::Vector{Int},
                             onsite::Vector{Matrix{ComplexF64}},
                             edges::Vector{BlockCayleyEdge},
                             roots::Vector{BlockCayleyRoot},
                             bath_hamiltonian::Matrix{ComplexF64},
                             coupling_matrix::Matrix{ComplexF64},
                             ::Val{:validated}) where {C<:DiscreteBath}
        new{C}(canonical, topology, sites, site_dimensions, onsite, edges,
               roots, bath_hamiltonian, coupling_matrix)
    end
end

function _block_cayley_site_ranges(sites::Vector{Symbol}, dimensions::Vector{Int})
    ranges = Dict{Symbol,UnitRange{Int}}()
    column = 1
    for (site, dimension) in zip(sites, dimensions)
        ranges[site] = column:(column + dimension - 1)
        column += dimension
    end
    return ranges
end

function BlockCayleyBath(canonical::C, topology::TreeTopology,
                         sites::AbstractVector{Symbol},
                         site_dimensions::AbstractVector{<:Integer},
                         onsite::AbstractVector{<:AbstractMatrix{<:Number}},
                         edges::AbstractVector{<:BlockCayleyEdge},
                         roots::AbstractVector{<:BlockCayleyRoot},
                         bath_hamiltonian::AbstractMatrix{<:Number},
                         coupling_matrix::AbstractMatrix{<:Number}) where {
                             C<:DiscreteBath}
    canonical_sites = Symbol.(sites)
    dimensions = Int.(site_dimensions)
    allunique(canonical_sites) || throw(ArgumentError(
        "BlockCayleyBath site labels must be unique",
    ))
    :cayley_hub in canonical_sites && throw(ArgumentError(
        "BlockCayleyBath virtual hub is topology-only, never a block site",
    ))
    all(dimension -> dimension > 0, dimensions) || throw(ArgumentError(
        "BlockCayleyBath site dimensions must be positive",
    ))
    length(canonical_sites) == length(dimensions) == length(onsite) ||
        throw(DimensionMismatch(
            "BlockCayleyBath needs one dimension and onsite block per site",
        ))
    sum(dimensions) == length(canonical) || throw(DimensionMismatch(
        "BlockCayleyBath site dimensions must retain every canonical bath mode",
    ))
    transformed_hamiltonian = Matrix{ComplexF64}(bath_hamiltonian)
    transformed_coupling = Matrix{ComplexF64}(coupling_matrix)
    size(transformed_hamiltonian) == (length(canonical), length(canonical)) ||
        throw(DimensionMismatch(
            "BlockCayleyBath Hamiltonian must cover every canonical bath mode",
        ))
    size(transformed_coupling) == (length(flavors(bath_layout(canonical))),
                                   length(canonical)) || throw(DimensionMismatch(
        "BlockCayleyBath coupling matrix must use global flavor rows",
    ))
    all(value -> isfinite(real(value)) && isfinite(imag(value)),
        transformed_hamiltonian) || throw(ArgumentError(
        "BlockCayleyBath Hamiltonian entries must be finite",
    ))
    all(value -> isfinite(real(value)) && isfinite(imag(value)),
        transformed_coupling) || throw(ArgumentError(
        "BlockCayleyBath coupling entries must be finite",
    ))
    hermitian_tolerance = 128 * eps(Float64) * max(norm(transformed_hamiltonian), 1.0)
    norm(transformed_hamiltonian - adjoint(transformed_hamiltonian)) <=
        hermitian_tolerance || throw(ArgumentError(
            "BlockCayleyBath Hamiltonian must be Hermitian within roundoff",
        ))
    topology_sites = Symbol[site for site in topology.ids if site != :cayley_hub]
    Set(topology_sites) == Set(canonical_sites) || throw(ArgumentError(
        "BlockCayleyBath topology physical sites must match block-site labels",
    ))

    ranges = _block_cayley_site_ranges(canonical_sites, dimensions)
    canonical_onsite = Matrix{ComplexF64}[Matrix{ComplexF64}(block) for block in onsite]
    for (site, block) in zip(canonical_sites, canonical_onsite)
        range = ranges[site]
        size(block) == (length(range), length(range)) || throw(DimensionMismatch(
            "BlockCayleyBath onsite block for $site has the wrong dimension",
        ))
        norm(block - transformed_hamiltonian[range, range]) <= hermitian_tolerance ||
            throw(ArgumentError(
                "BlockCayleyBath onsite block for $site disagrees with its full Hamiltonian",
            ))
    end

    canonical_edges = BlockCayleyEdge[edge for edge in edges]
    physical_topology_edges = Pair{Symbol,Symbol}[
        topology.ids[topology.parent[child]] => topology.ids[child]
        for child in eachindex(topology.ids)
        if topology.parent[child] != 0 &&
           topology.ids[topology.parent[child]] != :cayley_hub
    ]
    Pair{Symbol,Symbol}[edge.parent => edge.child for edge in canonical_edges] ==
        physical_topology_edges || throw(ArgumentError(
            "BlockCayleyBath edges must match nonvirtual topology edges in order",
        ))
    for edge in canonical_edges
        haskey(ranges, edge.parent) && haskey(ranges, edge.child) ||
            throw(ArgumentError("BlockCayleyBath edge references an unknown block site"))
        parent_range = ranges[edge.parent]
        child_range = ranges[edge.child]
        size(edge.hopping) == (length(parent_range), length(child_range)) ||
            throw(DimensionMismatch("BlockCayleyBath edge has the wrong block shape"))
        norm(edge.hopping - transformed_hamiltonian[parent_range, child_range]) <=
            hermitian_tolerance || throw(ArgumentError(
                "BlockCayleyBath edge disagrees with its full Hamiltonian",
            ))
        !iszero(norm(edge.hopping)) || throw(ArgumentError(
            "BlockCayleyBath physical edges must retain a nonzero hopping block",
        ))
    end

    canonical_roots = BlockCayleyRoot[root for root in roots]
    allunique(root.site for root in canonical_roots) || throw(ArgumentError(
        "BlockCayleyBath root sites must be unique",
    ))
    layout = bath_layout(canonical)
    for root in canonical_roots
        haskey(ranges, root.site) || throw(ArgumentError(
            "BlockCayleyBath root references an unknown block site",
        ))
        topology_index = topology.index[root.site]
        topology.parent[topology_index] == 0 ||
            topology.ids[topology.parent[topology_index]] == :cayley_hub ||
            throw(ArgumentError(
                "BlockCayleyBath roots must be topology roots or virtual-hub children",
            ))
        all(flavor -> flavor in flavors(layout), root.flavors) || throw(ArgumentError(
            "BlockCayleyBath root references an unknown flavor",
        ))
        Tuple(flavor for flavor in flavors(layout) if flavor in root.flavors) ==
            root.flavors || throw(ArgumentError(
                "BlockCayleyBath root flavors must retain global layout order",
            ))
        rows = Int[flavor_index(layout, flavor) for flavor in root.flavors]
        range = ranges[root.site]
        size(root.coupling) == (length(rows), length(range)) || throw(DimensionMismatch(
            "BlockCayleyBath root coupling has the wrong block shape",
        ))
        norm(root.coupling - transformed_coupling[rows, range]) <=
            128 * eps(Float64) * max(norm(transformed_coupling), 1.0) ||
            throw(ArgumentError(
                "BlockCayleyBath root coupling disagrees with its full matrix",
            ))
        !iszero(norm(root.coupling)) || throw(ArgumentError(
            "BlockCayleyBath roots must retain a nonzero impurity coupling",
        ))
    end
    return BlockCayleyBath(
        canonical, topology, canonical_sites, dimensions, canonical_onsite,
        canonical_edges, canonical_roots, transformed_hamiltonian,
        transformed_coupling, Val(:validated),
    )
end

"""Per caller-declared ownership group mapping evidence."""
struct CayleyGroupReport
    name::Symbol
    modes::Tuple{Vararg{Int}}
    flavors::Tuple{Vararg{Symbol}}
    root_dimensions::Vector{Int}
    scalar::Bool
    full_root_rank::Int
    retained_root_rank::Int
    root_coupling_residual::Float64
    rank_reduction_residual::Float64
    root_rank_thresholds::Vector{Float64}
    root_singular_values::Vector{Vector{Float64}}
    rank_reduced::Bool
    zero_hopping_components::Int

    function CayleyGroupReport(name::Symbol, modes::Tuple{Vararg{Int}},
                               flavors::Tuple{Vararg{Symbol}},
                               root_dimensions::Vector{Int}, scalar::Bool,
                               full_root_rank::Int, retained_root_rank::Int,
                               root_coupling_residual::Float64,
                               rank_reduction_residual::Float64,
                               root_rank_thresholds::Vector{Float64},
                               root_singular_values::Vector{Vector{Float64}},
                               rank_reduced::Bool,
                               zero_hopping_components::Int,
                               ::Val{:validated})
        new(name, modes, flavors, root_dimensions, scalar, full_root_rank,
            retained_root_rank, root_coupling_residual, rank_reduction_residual,
            root_rank_thresholds, root_singular_values, rank_reduced,
            zero_hopping_components)
    end
end

function CayleyGroupReport(name::Symbol, modes::Tuple{Vararg{Int}},
                           flavors::Tuple{Vararg{Symbol}},
                           root_dimensions::AbstractVector{<:Integer};
                           scalar::Bool,
                           full_root_rank::Integer=sum(root_dimensions),
                           retained_root_rank::Integer=sum(root_dimensions),
                           root_coupling_residual::Real=0.0,
                           rank_reduction_residual::Real=0.0,
                           root_rank_thresholds=Float64[],
                           root_singular_values=Vector{Vector{Float64}}(),
                           rank_reduced::Bool=false,
                           zero_hopping_components::Integer=0)
    dimensions = Int.(root_dimensions)
    all(dimension -> dimension > 0, dimensions) || throw(ArgumentError(
        "Cayley group root dimensions must be positive",
    ))
    scalar && all(==(1), dimensions) || !scalar || throw(ArgumentError(
        "scalar Cayley groups require one-dimensional roots",
    ))
    resolved_full_rank = Int(full_root_rank)
    resolved_retained_rank = Int(retained_root_rank)
    resolved_full_rank >= resolved_retained_rank >= sum(dimensions) ||
        throw(ArgumentError(
            "Cayley group ranks must satisfy full >= retained >= root dimensions",
        ))
    resolved_residual = Float64(root_coupling_residual)
    isfinite(resolved_residual) && resolved_residual >= 0 || throw(ArgumentError(
        "Cayley group root coupling residual must be finite and nonnegative",
    ))
    resolved_rank_reduction = Float64(rank_reduction_residual)
    isfinite(resolved_rank_reduction) && resolved_rank_reduction >= 0 ||
        throw(ArgumentError(
            "Cayley group rank reduction residual must be finite and nonnegative",
        ))
    thresholds = Float64.(root_rank_thresholds)
    singular_values = Vector{Float64}[Float64.(values)
                                      for values in root_singular_values]
    length(thresholds) == length(singular_values) || throw(ArgumentError(
        "Cayley group root thresholds and singular-value sets must align",
    ))
    all(value -> isfinite(value) && value >= 0, thresholds) || throw(ArgumentError(
        "Cayley group root thresholds must be finite and nonnegative",
    ))
    all(values -> all(value -> isfinite(value) && value >= 0, values),
        singular_values) || throw(ArgumentError(
        "Cayley group root singular values must be finite and nonnegative",
    ))
    rank_reduced == (resolved_rank_reduction > 0) ||
        throw(ArgumentError(
            "Cayley group rank-reduced flag must match its reduction residual",
        ))
    zero_hopping_components >= 0 || throw(ArgumentError(
        "Cayley group zero_hopping_components must be nonnegative",
    ))
    return CayleyGroupReport(name, modes, flavors, dimensions, scalar,
                             resolved_full_rank, resolved_retained_rank,
                             resolved_residual, resolved_rank_reduction,
                             thresholds, singular_values, rank_reduced,
                             Int(zero_hopping_components),
                             Val(:validated))
end

"""Numerical and topology evidence for an experimental Cayley mapping."""
struct BathMappingReport{G<:Tuple}
    unitarity_error::Float64
    spectrum_error::Float64
    hybridization_error::Float64
    tree_sparsity_error::Float64
    tree_tolerance::Float64
    rank_tolerance::Union{Nothing,Float64}
    root_coupling_residual::Float64
    rank_reduction_residual::Float64
    tree_connected::Bool
    virtual_hub::Bool
    zero_hopping_components::Int
    groups::G
    validation_points::Tuple{Vararg{ComplexF64}}
    timing_seconds::Float64
    approximate::Bool
    experimental::Bool
end

function BathMappingReport(; unitarity_error::Real, spectrum_error::Real,
                           hybridization_error::Real,
                           tree_sparsity_error::Real,
                           tree_tolerance::Real,
                           rank_tolerance=nothing,
                           root_coupling_residual::Real,
                           rank_reduction_residual::Real=0.0,
                           tree_connected::Bool,
                           virtual_hub::Bool,
                           zero_hopping_components::Integer,
                           groups=(), validation_points=(),
                           timing_seconds::Real, approximate::Bool=false,
                           experimental::Bool=true)
    errors = Float64[unitarity_error, spectrum_error, hybridization_error,
                     tree_sparsity_error, root_coupling_residual,
                     rank_reduction_residual, timing_seconds]
    all(value -> isfinite(value) && value >= 0, errors) || throw(ArgumentError(
        "BathMappingReport errors and timing must be finite and nonnegative",
    ))
    resolved_tree_tolerance = Float64(tree_tolerance)
    isfinite(resolved_tree_tolerance) && resolved_tree_tolerance > 0 ||
        throw(ArgumentError("BathMappingReport tree_tolerance must be finite and positive"))
    resolved_rank_tolerance = rank_tolerance === nothing ? nothing :
                              Float64(rank_tolerance)
    resolved_rank_tolerance === nothing ||
        (isfinite(resolved_rank_tolerance) && resolved_rank_tolerance >= 0) ||
        throw(ArgumentError("BathMappingReport rank_tolerance must be finite and nonnegative"))
    zero_hopping_components >= 0 || throw(ArgumentError(
        "BathMappingReport zero_hopping_components must be nonnegative",
    ))
    canonical_groups = Tuple(groups)
    all(group -> group isa CayleyGroupReport, canonical_groups) ||
        throw(ArgumentError("BathMappingReport groups must be CayleyGroupReport values"))
    points = Tuple(ComplexF64.(validation_points))
    all(point -> isfinite(real(point)) && isfinite(imag(point)), points) ||
        throw(ArgumentError("BathMappingReport validation points must be finite"))
    approximate || iszero(errors[6]) || throw(ArgumentError(
        "BathMappingReport rank reduction must be marked approximate",
    ))
    return BathMappingReport(errors[1], errors[2], errors[3], errors[4],
                             resolved_tree_tolerance, resolved_rank_tolerance, errors[5],
                             errors[6],
                             tree_connected, virtual_hub,
                             Int(zero_hopping_components), canonical_groups,
                             points, errors[7], approximate, experimental)
end

"""
Typed mapping result retaining both canonical and mapped bath descriptions.
`transform` is the complete bath-only unitary `U` in `c = U * a`, so the
retained mapped matrices satisfy `H' = U' * H * U` and `W' = W * U`.
"""
struct CayleyMappingResult{C<:DiscreteBath,B<:AbstractCayleyBath,
                           R<:BathMappingReport}
    canonical::C
    mapped::B
    transform::Matrix{ComplexF64}
    report::R
end

bath_layout(bath::AbstractCayleyBath) = bath_layout(bath.canonical)
bath_partition(bath::AbstractCayleyBath) = bath_partition(bath.canonical)
bath_orbitals(bath::AbstractCayleyBath) = bath_orbitals(bath.canonical)
bath_statistics(bath::AbstractCayleyBath) = bath_statistics(bath.canonical)
Base.length(bath::AbstractCayleyBath) = length(bath.canonical)

function _cayley_coupling_matrix(bath::DiscreteBath)
    layout = bath_layout(bath)
    partition = bath_partition(bath)
    orbitals = bath_orbitals(bath)
    matrix = zeros(ComplexF64, length(flavors(layout)), length(orbitals))
    for mode in eachindex(orbitals.energies)
        block = block_names(partition)[orbitals.block_indices[mode]]
        for (component, flavor) in enumerate(block_flavors(partition, block))
            matrix[flavor_index(layout, flavor), mode] = orbitals.couplings[mode][component]
        end
    end
    return matrix
end

function _validate_cayley_groups(kernel::CayleyTreeKernel, bath::DiscreteBath,
                                 coupling::Matrix{ComplexF64})
    layout = bath_layout(bath)
    orbitals = bath_orbitals(bath)
    all_modes = Int[]
    for group in kernel.groups
        all(flavor -> flavor in flavors(layout), group.flavors) || throw(ArgumentError(
            "Cayley ownership group $(group.name) references an unknown flavor",
        ))
        all(mode -> mode <= length(orbitals), group.modes) || throw(ArgumentError(
            "Cayley ownership group $(group.name) references a missing bath mode",
        ))
        group_blocks = unique(orbitals.block_indices[collect(group.modes)])
        length(group_blocks) == 1 || throw(ArgumentError(
            "Cayley ownership group $(group.name) must remain within one named Partition block",
        ))
        declared_block = block_names(bath_partition(bath))[only(group_blocks)]
        declared_flavors = block_flavors(bath_partition(bath), declared_block)
        Tuple(flavor for flavor in declared_flavors if flavor in group.flavors) ==
            group.flavors || throw(ArgumentError(
                "Cayley ownership group $(group.name) flavors must be an ordered " *
                "subsequence of block $declared_block",
            ))
        append!(all_modes, group.modes)
        for mode in group.modes
            orbitals.associated_flavors[mode] in group.flavors || throw(ArgumentError(
                "Cayley ownership group $(group.name) excludes canonical owner " *
                "$(orbitals.associated_flavors[mode]) for mode $mode",
            ))
            for (row, flavor) in enumerate(flavors(layout))
                iszero(coupling[row, mode]) || flavor in group.flavors || throw(ArgumentError(
                    "Cayley ownership group $(group.name) excludes nonzero support " *
                    "$flavor for mode $mode",
                ))
            end
        end
    end
    sort!(all_modes)
    all_modes == collect(1:length(orbitals)) || throw(ArgumentError(
        "Cayley ownership groups must partition every canonical bath mode exactly once",
    ))
    return nothing
end

function _cayley_hybridization(coupling::Matrix{ComplexF64},
                               bath_hamiltonian::Matrix{ComplexF64}, z::ComplexF64)
    dimension = size(bath_hamiltonian, 1)
    resolvent = z * Matrix{ComplexF64}(I, dimension, dimension) - bath_hamiltonian
    return coupling * (resolvent \ adjoint(coupling))
end
