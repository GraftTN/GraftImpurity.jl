using Test
using LinearAlgebra: Diagonal, Hermitian, I, eigvals, norm
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
end
