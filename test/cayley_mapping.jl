using Test
using LinearAlgebra: Diagonal, Hermitian, I, eigen, eigvals, norm
using Graft
using GraftImpurity

function _cayley_delta(coupling::AbstractMatrix{<:Number},
                       bath_hamiltonian::AbstractMatrix{<:Number}, z::Complex)
    dimension = size(bath_hamiltonian, 1)
    resolvent = z * Matrix{ComplexF64}(I, dimension, dimension) - bath_hamiltonian
    return coupling * (resolvent \ adjoint(coupling))
end

function _scalar_cayley_fixture(; energies=[-1.2, -0.4, 0.35, 1.1],
                                couplings=ComplexF64[0.4 + 0.2im, -0.1im,
                                                     0.3 - 0.15im, 0.2])
    layout = FlavorLayout(
        [:imp], Dict(:imp => :impurity), Dict(:impurity => [:imp]);
        basis=:scalar_cayley,
    )
    partition = Partition(:imp => [:imp])
    count = length(energies)
    orbitals = BathOrbitals(
        energies, [[value] for value in couplings], collect(1:count),
        fill(1, count), fill(:imp, count); layout, partition,
    )
    return DiscreteBath(layout, partition, orbitals; statistics=:fermion)
end

function _scalar_kernel(mode_count::Int; partitioner=BalancedCayleyPartitioner())
    group = CayleyOwnershipGroup(:imp_tree, collect(1:mode_count), [:imp])
    return CayleyTreeKernel(ScalarCayley(), [group]; branching=2, partitioner)
end

function _site_index(mapped::ScalarCayleyBath, site::Symbol)
    index = findfirst(==(site), mapped.sites)
    index === nothing && throw(KeyError(site))
    return index
end

function _block_cayley_fixture(; energies=[-1.1, -0.25, 0.4, 1.0],
                               couplings=[
                                   ComplexF64[0.35 + 0.1im, 0.2 - 0.05im],
                                   ComplexF64[0.15im, 0.3 + 0.2im],
                                   ComplexF64[0.2 - 0.1im, -0.12im],
                                   ComplexF64[0.1 + 0.18im, 0.25],
                               ])
    layout = FlavorLayout(
        [:up, :down], Dict(:up => :up_site, :down => :down_site),
        Dict(:up_site => [:up], :down_site => [:down]); basis=:block_cayley,
    )
    partition = Partition(:spin => [:up, :down])
    count = length(energies)
    orbitals = BathOrbitals(
        energies, couplings, collect(1:count), fill(1, count),
        [isodd(index) ? :up : :down for index in 1:count]; layout, partition,
    )
    return DiscreteBath(layout, partition, orbitals; statistics=:fermion)
end

function _block_kernel(mode_count::Int;
                       partitioner=BalancedCayleyPartitioner(),
                       rank_tolerance=nothing)
    group = CayleyOwnershipGroup(:spin_tree, collect(1:mode_count), [:up, :down])
    return CayleyTreeKernel(
        BlockCayley(), [group]; branching=2, partitioner, rank_tolerance,
    )
end

function _block_site_range(mapped::BlockCayleyBath, site::Symbol)
    index = findfirst(==(site), mapped.sites)
    index === nothing && throw(KeyError(site))
    start = sum(mapped.site_dimensions[1:(index - 1)]; init=0) + 1
    return start:(start + mapped.site_dimensions[index] - 1)
end

function _block_site_dimension(mapped::BlockCayleyBath, site::Symbol)
    index = findfirst(==(site), mapped.sites)
    index === nothing && throw(KeyError(site))
    return mapped.site_dimensions[index]
end

function _fock_annihilator(mode_count::Int, mode::Int)
    dimension = 1 << mode_count
    operator = zeros(ComplexF64, dimension, dimension)
    bit = 1 << (mode - 1)
    for state in 0:(dimension - 1)
        state & bit == 0 && continue
        destination = state & ~bit
        sign = isodd(count_ones(state & (bit - 1))) ? -1.0 : 1.0
        operator[destination + 1, state + 1] = sign
    end
    return operator
end

function _fock_onebody_hamiltonian(onebody::AbstractMatrix{<:Number})
    mode_count = size(onebody, 1)
    size(onebody, 2) == mode_count || throw(DimensionMismatch("one-body matrix must be square"))
    annihilators = [_fock_annihilator(mode_count, mode) for mode in 1:mode_count]
    hamiltonian = zeros(ComplexF64, 1 << mode_count, 1 << mode_count)
    for column in 1:mode_count, row in 1:mode_count
        hamiltonian .+= onebody[row, column] *
                       adjoint(annihilators[row]) * annihilators[column]
    end
    return hamiltonian, annihilators
end

@testset "M5 scalar Cayley bath mapping" begin
    bath = _scalar_cayley_fixture()
    canonical_energies = copy(bath.orbitals.energies)
    canonical_couplings = deepcopy(bath.orbitals.couplings)
    result = map_bath(_scalar_kernel(length(bath)), bath)

    @test result isa CayleyMappingResult
    @test result.mapped isa ScalarCayleyBath
    @test result.canonical === bath
    @test bath.orbitals.energies == canonical_energies
    @test bath.orbitals.couplings == canonical_couplings
    @test result.report.experimental
    @test !result.report.virtual_hub
    @test !result.report.approximate
    @test result.report.tree_tolerance == 1e-10
    @test result.report.tree_sparsity_error < 1e-10
    @test result.report.root_coupling_residual < 1e-10

    root = only(result.mapped.roots)
    root_index = _site_index(result.mapped, root.site)
    @test Graft.Trees.nodeid(result.mapped.topology,
                             result.mapped.topology.root) == root.site
    vector = ComplexF64[value[1] for value in bath.orbitals.couplings]
    @test result.transform[:, root_index] ≈ conj.(vector) / norm(vector) atol=1e-12
    @test root.coupling ≈ norm(vector) atol=1e-12
    @test abs(imag(root.coupling)) < 1e-12
    @test norm(result.mapped.coupling_matrix[:, setdiff(1:length(bath), [root_index])]) <
          1e-10
    @test adjoint(result.transform) * result.transform ≈
          Matrix{ComplexF64}(I, length(bath), length(bath)) atol=1e-10
    @test sort(real.(eigvals(Hermitian(result.mapped.bath_hamiltonian)))) ≈
          sort(bath.orbitals.energies) atol=1e-10
    for z in (0.7im, -0.25 + 0.8im)
        @test _cayley_delta(
            GraftImpurity._cayley_coupling_matrix(bath),
            Matrix{ComplexF64}(Diagonal(ComplexF64.(bath.orbitals.energies))), z,
        ) ≈ _cayley_delta(result.mapped.coupling_matrix,
                           result.mapped.bath_hamiltonian, z) atol=1e-10
    end
    @test all(abs(edge.hopping) > 1e-12 for edge in result.mapped.edges)
    @test all(Graft.Trees.nchildren(result.mapped.topology,
                                    Graft.Trees.nodeindex(result.mapped.topology, edge.parent)) <= 2
              for edge in result.mapped.edges)

    split = map_bath(
        _scalar_kernel(length(bath);
                       partitioner=EnergySplitCayleyPartitioner(0.0)), bath,
    )
    @test length(split.mapped.roots) == 2
    @test !split.report.tree_connected
    @test split.report.virtual_hub
    @test split.report.hybridization_error < 1e-10

    degenerate = _scalar_cayley_fixture(
        energies=[0.5, 0.5], couplings=ComplexF64[0.3 + 0.2im, -0.1im],
    )
    dark = map_bath(_scalar_kernel(length(degenerate)), degenerate)
    @test length(dark.mapped.sites) == length(degenerate)
    @test dark.report.zero_hopping_components == 1
    @test !dark.report.tree_connected
    @test dark.report.virtual_hub
    @test isempty(dark.mapped.edges)

    # A physical edge below the user audit tolerance remains in the topology;
    # only machine-scale roundoff is classified as a dark component.
    nearby = _scalar_cayley_fixture(
        energies=[0.0, 1e-12], couplings=ComplexF64[0.3, 0.2],
    )
    nearby_result = map_bath(_scalar_kernel(length(nearby)), nearby)
    @test length(nearby_result.mapped.edges) == 1
    @test abs(only(nearby_result.mapped.edges).hopping) <
          nearby_result.report.tree_tolerance
    @test nearby_result.report.tree_sparsity_error < 1e-20
    @test !nearby_result.report.approximate

    diagonal_layout = FlavorLayout(
        [:up, :down], Dict(:up => :up_site, :down => :down_site),
        Dict(:up_site => [:up], :down_site => [:down]); basis=:diagonal_cayley,
    )
    diagonal_partition = Partition(:spin => [:up, :down])
    diagonal_orbitals = BathOrbitals(
        [-1.0, -0.7, 0.4, 0.9],
        [ComplexF64[0.3, 0.0], ComplexF64[0.0, 0.2im],
         ComplexF64[0.1, 0.0], ComplexF64[0.0, 0.4]],
        [1, 2, 3, 4], fill(1, 4), [:up, :down, :up, :down];
        layout=diagonal_layout, partition=diagonal_partition,
    )
    diagonal_bath = DiscreteBath(diagonal_layout, diagonal_partition,
                                 diagonal_orbitals; statistics=:fermion)
    diagonal_kernel = CayleyTreeKernel(
        ScalarCayley(), [
            CayleyOwnershipGroup(:up_tree, [1, 3], [:up]),
            CayleyOwnershipGroup(:down_tree, [2, 4], [:down]),
        ]; branching=2,
    )
    diagonal_result = map_bath(diagonal_kernel, diagonal_bath)
    @test diagonal_result.mapped isa ScalarCayleyBath
    @test norm(diagonal_result.transform[[1, 3], 3:4]) < 1e-12
    @test norm(diagonal_result.transform[[2, 4], 1:2]) < 1e-12
    @test diagonal_result.report.hybridization_error < 1e-10

    matrix_partition = Partition(:spin => [:up, :down])
    matrix_orbitals = BathOrbitals(
        [-0.4], [ComplexF64[0.2, 0.1im]], [1], [1], [:up];
        layout=diagonal_layout, partition=matrix_partition,
    )
    matrix_bath = DiscreteBath(diagonal_layout, matrix_partition, matrix_orbitals;
                                statistics=:fermion)
    invalid_scalar = CayleyTreeKernel(
        ScalarCayley(), [CayleyOwnershipGroup(:up_only, [1], [:up])],
    )
    @test_throws ArgumentError map_bath(invalid_scalar, matrix_bath)
    @test_throws ArgumentError map_bath(
        CayleyTreeKernel(
            ScalarCayley(), [CayleyOwnershipGroup(:imp_tree, [1, 2, 3, 4], [:imp])];
            rank_tolerance=1e-2,
        ),
        bath,
    )
end

@testset "M5 block Cayley bath mapping" begin
    bath = _block_cayley_fixture()
    canonical_energies = copy(bath.orbitals.energies)
    canonical_couplings = deepcopy(bath.orbitals.couplings)
    result = map_bath(_block_kernel(length(bath)), bath)

    @test result.mapped isa BlockCayleyBath
    @test result.canonical === bath
    @test bath.orbitals.energies == canonical_energies
    @test bath.orbitals.couplings == canonical_couplings
    @test sum(result.mapped.site_dimensions) == length(bath)
    @test !result.report.approximate
    @test result.report.rank_tolerance === nothing
    @test result.report.tree_sparsity_error < 1e-10
    @test result.report.root_coupling_residual < 1e-10
    report_group = only(result.report.groups)
    @test !report_group.scalar
    @test report_group.full_root_rank == 2
    @test report_group.retained_root_rank == 2
    @test report_group.root_coupling_residual < 1e-10
    @test report_group.rank_reduction_residual == 0.0
    @test !report_group.rank_reduced
    @test length(report_group.root_rank_thresholds) == 1
    @test length(report_group.root_singular_values) == 1

    root = only(result.mapped.roots)
    root_range = _block_site_range(result.mapped, root.site)
    @test length(root_range) == 2
    @test size(root.coupling) == (2, 2)
    @test root.coupling ≈ result.mapped.coupling_matrix[:, root_range] atol=1e-12
    @test adjoint(result.transform) * result.transform ≈
          Matrix{ComplexF64}(I, length(bath), length(bath)) atol=1e-10
    @test sort(real.(eigvals(Hermitian(result.mapped.bath_hamiltonian)))) ≈
          sort(bath.orbitals.energies) atol=1e-10
    for z in (0.7im, -0.25 + 0.8im)
        @test _cayley_delta(
            GraftImpurity._cayley_coupling_matrix(bath),
            Matrix{ComplexF64}(Diagonal(ComplexF64.(bath.orbitals.energies))), z,
        ) ≈ _cayley_delta(result.mapped.coupling_matrix,
                           result.mapped.bath_hamiltonian, z) atol=1e-10
    end
    @test all(size(edge.hopping) ==
              (_block_site_dimension(result.mapped, edge.parent),
               _block_site_dimension(result.mapped, edge.child))
              for edge in result.mapped.edges)
    @test !isempty(result.mapped.edges)
    invalid_onsite = copy(result.mapped.onsite)
    invalid_onsite[1] = zeros(ComplexF64, size(invalid_onsite[1]))
    @test_throws ArgumentError BlockCayleyBath(
        bath, result.mapped.topology, result.mapped.sites,
        result.mapped.site_dimensions, invalid_onsite, result.mapped.edges,
        result.mapped.roots, result.mapped.bath_hamiltonian,
        result.mapped.coupling_matrix,
    )
    chosen_edge = first(result.mapped.edges)
    @test chosen_edge.parent == root.site
    descendant_site = chosen_edge.child
    descendant_range = _block_site_range(result.mapped, descendant_site)
    reoriented_topology_edges = Pair{Symbol,Symbol}[
        descendant_site => root.site,
    ]
    append!(reoriented_topology_edges,
            Pair{Symbol,Symbol}[edge.parent => edge.child
                                 for edge in result.mapped.edges[2:end]])
    descendant_topology = Graft.TreeTopology(
        descendant_site, reoriented_topology_edges,
    )
    descendant_edges = BlockCayleyEdge[
        BlockCayleyEdge(
            descendant_site, root.site,
            Matrix{ComplexF64}(
                result.mapped.bath_hamiltonian[descendant_range, root_range],
            ),
        ),
    ]
    append!(descendant_edges, result.mapped.edges[2:end])
    @test_throws ArgumentError BlockCayleyBath(
        bath, descendant_topology, result.mapped.sites,
        result.mapped.site_dimensions, result.mapped.onsite, descendant_edges,
        result.mapped.roots, result.mapped.bath_hamiltonian,
        result.mapped.coupling_matrix,
    )
    @test_throws MethodError mount_bath(result.mapped.topology, result.mapped)

    impurity_onebody = ComplexF64[-0.55 0.04im; -0.04im 0.15]
    canonical_onebody = vcat(
        hcat(impurity_onebody, GraftImpurity._cayley_coupling_matrix(bath)),
        hcat(adjoint(GraftImpurity._cayley_coupling_matrix(bath)),
             Matrix{ComplexF64}(Diagonal(ComplexF64.(bath.orbitals.energies)))),
    )
    mapped_onebody = vcat(
        hcat(impurity_onebody, result.mapped.coupling_matrix),
        hcat(adjoint(result.mapped.coupling_matrix), result.mapped.bath_hamiltonian),
    )
    canonical_ed, annihilators = _fock_onebody_hamiltonian(canonical_onebody)
    mapped_ed, _ = _fock_onebody_hamiltonian(mapped_onebody)
    @test sort(real.(eigvals(Hermitian(canonical_ed)))) ≈
          sort(real.(eigvals(Hermitian(mapped_ed)))) atol=1e-10
    impurity_number = adjoint(annihilators[1]) * annihilators[1] +
                      adjoint(annihilators[2]) * annihilators[2]
    canonical_ground = eigen(Hermitian(canonical_ed)).vectors[:, 1]
    mapped_ground = eigen(Hermitian(mapped_ed)).vectors[:, 1]
    @test real(adjoint(canonical_ground) * impurity_number * canonical_ground) ≈
          real(adjoint(mapped_ground) * impurity_number * mapped_ground) atol=1e-10

    split = map_bath(
        _block_kernel(length(bath);
                      partitioner=EnergySplitCayleyPartitioner(0.0)), bath,
    )
    @test length(split.mapped.roots) == 2
    @test split.report.virtual_hub
    @test split.report.root_coupling_residual < 1e-10

    rank_one = _block_cayley_fixture(
        energies=[-0.9, 0.2, 0.8],
        couplings=[
            ComplexF64[0.4 + 0.1im, -0.2im],
            ComplexF64[0.8 + 0.2im, -0.4im],
            ComplexF64[-0.2 - 0.05im, 0.1im],
        ],
    )
    rank_one_result = map_bath(_block_kernel(length(rank_one)), rank_one)
    rank_one_group = only(rank_one_result.report.groups)
    @test rank_one_group.full_root_rank == 1
    @test rank_one_group.retained_root_rank == 1
    @test length(only(rank_one_result.mapped.roots).coupling[1, :]) == 1
    @test rank_one_result.report.root_coupling_residual < 1e-10
    @test !rank_one_result.report.approximate

    reduced = _block_cayley_fixture(
        energies=[-0.4, 0.7],
        couplings=[ComplexF64[1.0, 0.0], ComplexF64[0.0, 1e-3]],
    )
    default_reduced_result = map_bath(_block_kernel(length(reduced)), reduced)
    default_reduced_group = only(default_reduced_result.report.groups)
    @test !default_reduced_result.report.approximate
    @test default_reduced_group.retained_root_rank == 2
    @test default_reduced_result.report.root_coupling_residual < 1e-10
    reduced_result = map_bath(
        _block_kernel(length(reduced); rank_tolerance=1e-2), reduced,
    )
    reduced_group = only(reduced_result.report.groups)
    @test reduced_result.report.approximate
    @test reduced_result.report.rank_tolerance == 1e-2
    @test reduced_group.full_root_rank == 2
    @test reduced_group.retained_root_rank == 1
    @test reduced_group.rank_reduced
    @test length(only(reduced_result.mapped.roots).coupling[1, :]) == 1
    @test reduced_group.root_coupling_residual > 1e-4
    @test reduced_group.rank_reduction_residual > 1e-4
    @test reduced_result.report.rank_reduction_residual > 1e-4
    @test sum(reduced_result.mapped.site_dimensions) == length(reduced)
    for z in (0.5im, 0.3 + 0.9im)
        @test _cayley_delta(
            GraftImpurity._cayley_coupling_matrix(reduced),
            Matrix{ComplexF64}(Diagonal(ComplexF64.(reduced.orbitals.energies))), z,
        ) ≈ _cayley_delta(reduced_result.mapped.coupling_matrix,
                           reduced_result.mapped.bath_hamiltonian, z) atol=1e-10
    end

    matrix_group = CayleyOwnershipGroup(:scalar_reject, collect(1:length(bath)),
                                         [:up, :down])
    @test_throws ArgumentError map_bath(
        CayleyTreeKernel(ScalarCayley(), [matrix_group]), bath,
    )

    four_layout = FlavorLayout(
        [:a, :b, :c, :d],
        Dict(:a => :a_site, :b => :b_site, :c => :c_site, :d => :d_site),
        Dict(:a_site => [:a], :b_site => [:b], :c_site => [:c], :d_site => [:d]);
        basis=:two_block_cayley,
    )
    four_partition = Partition(:left => [:a, :b], :right => [:c, :d])
    four_orbitals = BathOrbitals(
        [-0.8, -0.3, 0.3, 0.9],
        [ComplexF64[0.3, 0.1im], ComplexF64[0.2, -0.1im],
         ComplexF64[0.1im, 0.25], ComplexF64[-0.05im, 0.35]],
        [1, 2, 3, 4], [1, 2, 1, 2], [:a, :c, :b, :d];
        layout=four_layout, partition=four_partition,
    )
    four_bath = DiscreteBath(four_layout, four_partition, four_orbitals;
                             statistics=:fermion)
    two_group_kernel = CayleyTreeKernel(
        BlockCayley(), [
            CayleyOwnershipGroup(:left_tree, [1, 3], [:a, :b]),
            CayleyOwnershipGroup(:right_tree, [2, 4], [:c, :d]),
        ],
    )
    two_group_result = map_bath(two_group_kernel, four_bath)
    @test norm(two_group_result.transform[[1, 3], 3:4]) < 1e-12
    @test norm(two_group_result.transform[[2, 4], 1:2]) < 1e-12
    @test two_group_result.report.virtual_hub
end
