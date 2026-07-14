using Test
using Graft
using GraftImpurity

function _foundation_layout()
    return FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis=:spin_orbital,
    )
end

@testset "breaking impurity foundations" begin
    layout = _foundation_layout()
    partition = Partition(:spin => [:up, :down])
    singleton_partition = Partition(:up => [:up], :down => [:down])

    @test flavors(layout) == (:up, :down)
    @test flavor_index(layout, :down) == 2
    @test physical_site(layout, :up) == :impurity
    @test site_modes(layout, :impurity) == (:up, :down)
    @test layout_sites(layout) == (:impurity,)
    @test basis_identity(layout) == :spin_orbital
    @test validate_partition(partition, layout) === partition
    @test partition.blocks.spin == (:up, :down)
    @test block_names(partition) == (:spin,)
    @test block_flavors(partition, :spin) == (:up, :down)
    @test block_index(partition, :spin) == 1
    @test partition_flavors(partition) == (:up, :down)
    @test hash(partition) == hash(Partition(:spin => [:up, :down]))

    @test_throws ArgumentError FlavorLayout(
        [:up, :up],
        Dict(:up => :impurity),
        Dict(:impurity => [:up]);
        basis=:bad,
    )
    @test_throws ArgumentError Partition(:up => [:up], :down => [:up])
    @test_throws MethodError Partition([[:up, :down]])
    @test_throws ArgumentError validate_partition(
        Partition(:spin => [:down, :up]), layout)

    scalar = BlockRealPoles(
        layout,
        singleton_partition,
        [-1.5, 0.7],
        [0.25, -0.15],
        [1, 2];
        statistics=:fermion,
    )
    @test length(scalar) == 2
    @test scalar.poles == [-1.5, 0.7]
    @test scalar.block_indices == [1, 2]
    @test scalar.residues == [0.25, -0.15]

    nonmountable_matrix = BlockRealPoles(
        layout,
        partition,
        [0.4],
        [ComplexF64[1 2 + im; 0 -0.5]],
        [1];
        statistics=:fermion,
    )
    @test nonmountable_matrix.residues[1][1, 2] == 2 + im
    @test nonmountable_matrix.residues[1][2, 1] == 0
    @test nonmountable_matrix.residues[1][2, 2] == -0.5
    @test PoleExpansion(nonmountable_matrix;
                        kernel=:boundary, trace=(; attempt=1)).kernel == :boundary
    @test_throws DimensionMismatch BlockRealPoles(
        layout,
        partition,
        [0.4],
        [reshape(ComplexF64[1], 1, 1)],
        [1];
        statistics=:fermion,
    )

    orbitals = BathOrbitals(
        [-1.5, 0.7],
        [ComplexF64[0.5], ComplexF64[0.2 - 0.1im]],
        [1, 2],
        [1, 2],
        [:up, :down];
        layout,
        partition=singleton_partition,
    )
    @test fieldnames(BathOrbitals) == (
        :energies,
        :couplings,
        :pole_indices,
        :block_indices,
        :associated_flavors,
    )
    @test length(orbitals) == 2
    @test orbitals.associated_flavors == [:up, :down]
    @test_throws ArgumentError BathOrbitals(
        [0.0],
        [ComplexF64[0.3]],
        [1],
        [1],
        [:down];
        layout,
        partition=singleton_partition,
    )

    bath = DiscreteBath(layout, singleton_partition, orbitals;
                        statistics=:fermion)
    @test bath_layout(bath) === layout
    @test bath_partition(bath) === singleton_partition
    @test bath_orbitals(bath) === orbitals
    @test bath_statistics(bath) === :fermion

    topology = TreeTopology(
        :impurity,
        [:impurity => :bath_up, :impurity => :bath_down],
    )
    ops = fermion_ops_z2()
    physical_spaces = Dict(
        :impurity => ops.P,
        :bath_up => ops.P,
        :bath_down => ops.P,
    )
    mounted = AndersonBath(
        bath,
        topology,
        physical_spaces,
        [:bath_up, :bath_down],
        [:impurity, :impurity],
        OpSum();
        diagnostics=(; stage=:foundation),
    )
    @test mounted.sites == (:bath_up, :bath_down)
    @test mounted.anchors == (:impurity, :impurity)
    @test hasproperty(mounted.phys, :bath_up)
    @test_throws DimensionMismatch AndersonBath(
        bath,
        topology,
        physical_spaces,
        [:bath_up],
        [:impurity],
        OpSum(),
    )

    t3ns = T3NS(layout)
    ftps = FTPS(layout; flavor_order=[:down, :up])
    @test t3ns.layout === layout
    @test ftps.flavor_order == (:down, :up)
    @test_throws ArgumentError T3NS(layout; flavor_order=[:up, :up])

    @test length(methods(mount_bath)) == 1
    @test length(methods(map_bath)) == 2
    @test length(methods(impurity_topology)) == 2
    @test length(methods(lower_interaction)) == 1
    @test length(methods(audit_partition)) == 0
    @test length(methods(reconstruct_hybridization)) >= 3
    @test length(methods(audit_bathfit)) == 1
    @test length(methods(audit_symmetry)) == 1
    @test length(methods(realize_quasi_lindblad)) == 0
    @test length(methods(realize_coupled_lindblad)) == 0
    @test length(methods(set_weiss!)) == 2
    @test length(methods(set_hybridization!)) == 2
    @test length(methods(solve!)) == 1

    @test !isdefined(GraftImpurity, :BathParametrization)
    @test !isdefined(GraftImpurity, :RealPoles)
    @test !isdefined(GraftImpurity, :MatrixRealPoles)
    @test !isdefined(GraftImpurity, :ThermofieldRealPoles)
    @test isdefined(GraftImpurity, :ComplexPoles)
    @test ComplexPoles <: AbstractBCFParametrization
    @test !isdefined(GraftImpurity, :fit_bath)
    @test !isdefined(GraftImpurity, :matsubara_reconstruct)
    @test !isdefined(GraftImpurity, :couplings)
    @test isdefined(GraftImpurity, :BosonBath)
    @test !isdefined(GraftImpurity, :AndersonRealPoles)
    @test isdefined(GraftImpurity, :AndersonBath)
    @test !isdefined(GraftImpurity, :MountedBath)
    @test ScalarCayley <: AbstractCayleyRoute
    @test BlockCayley <: AbstractCayleyRoute
    cayley_group = CayleyOwnershipGroup(:up, [1], [:up])
    cayley_kernel = CayleyTreeKernel(ScalarCayley(), [cayley_group])
    @test cayley_kernel.groups == (cayley_group,)
    @test_throws ArgumentError CayleyOwnershipGroup(:bad, [0], [:up])
    @test_throws ArgumentError CayleyTreeKernel(ScalarCayley(), CayleyOwnershipGroup[])
    @test !isdefined(GraftImpurity, :solve)
end
