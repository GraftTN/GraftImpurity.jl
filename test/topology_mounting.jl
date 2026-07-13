using Test
using Graft
using GraftImpurity
using Graft.Backend: blocks, dim, FermionParity
using LinearAlgebra: I, diag

function _m5_flavor_layout()
    return FlavorLayout(
        [:a, :b, :c],
        Dict(:a => :imp_a, :b => :imp_b, :c => :imp_c),
        Dict(:imp_a => [:a], :imp_b => [:b], :imp_c => [:c]);
        basis=:m5_flavor,
    )
end

function _m5_bath(layout::FlavorLayout; owners=[:b, :a, :c, :a])
    partition = Partition(:all => [:a, :b, :c])
    orbitals = BathOrbitals(
        [-1.0, -0.2, 0.3, 0.9],
        [
            ComplexF64[0.2, 0.1im, 0.0],
            ComplexF64[0.15, 0.0, -0.05im],
            ComplexF64[0.0, 0.35, 0.1],
            ComplexF64[-0.1im, 0.2, 0.25],
        ],
        [1, 2, 3, 4],
        [1, 1, 1, 1],
        owners;
        layout,
        partition,
    )
    return partition, DiscreteBath(layout, partition, orbitals; statistics=:fermion)
end

function _physical_dict(mounted::AndersonBath)
    return Dict(site => getproperty(mounted.phys, site)
                for site in propertynames(mounted.phys))
end

function _is_descendant(topology::TreeTopology, node::Symbol, ancestor::Symbol)
    index = Graft.Trees.nodeindex(topology, node)
    ancestor_index = Graft.Trees.nodeindex(topology, ancestor)
    while index != 0
        index == ancestor_index && return true
        index = topology.parent[index]
    end
    return false
end

function _charged_local_matrix(operator)
    local_blocks = Dict(blocks(operator))
    even = Matrix(local_blocks[FermionParity(0)])
    odd = Matrix(local_blocks[FermionParity(1)])
    zeros_even_odd = zeros(ComplexF64, size(even, 1), size(odd, 2))
    zeros_odd_even = zeros(ComplexF64, size(odd, 1), size(even, 2))
    return [zeros_even_odd even; odd zeros_odd_even]
end

function _neutral_local_matrix(operator)
    local_blocks = Dict(blocks(operator))
    even = Matrix(local_blocks[FermionParity(0)])
    odd = Matrix(local_blocks[FermionParity(1)])
    return [even zeros(ComplexF64, size(even, 1), size(odd, 2));
            zeros(ComplexF64, size(odd, 1), size(even, 2)) odd]
end

function _opsum_signature(H::OpSum)
    return [(term.coeff, Tuple((operator.site, operator.name)
                               for operator in term.ops)) for term in H]
end

@testset "M5 ownership-preserving topology and mounting" begin
    layout = _m5_flavor_layout()
    partition, bath = _m5_bath(layout)
    order = [:c, :a, :b]

    t3ns = impurity_topology(T3NS(layout; flavor_order=order), partition, bath)
    expected_bath_sites = (:bath_b_1, :bath_a_2, :bath_c_3, :bath_a_4)
    physical = Symbol[:imp_a, :imp_b, :imp_c, expected_bath_sites...]
    @test Graft.is_t3ns(t3ns; physical)
    @test count(node -> !(Graft.Trees.nodeid(t3ns, node) in physical),
                1:Graft.Trees.nnodes(t3ns)) == 1
    @test Graft.Trees.nodeid(t3ns, t3ns.root) == :imp_c
    for (site, owner) in zip(expected_bath_sites,
                              bath.orbitals.associated_flavors)
        @test _is_descendant(t3ns, site, physical_site(layout, owner))
    end

    ftps = impurity_topology(FTPS(layout; flavor_order=order), partition, bath)
    spine = [:imp_c, :imp_a, :imp_b]
    @test [Graft.Trees.nodeid(ftps, index) for index in 1:3] == spine
    @test !Graft.is_t3ns(ftps; physical)
    for (site, owner) in zip(expected_bath_sites,
                              bath.orbitals.associated_flavors)
        @test _is_descendant(ftps, site, physical_site(layout, owner))
    end

    changed_partition, changed_bath = _m5_bath(layout; owners=[:a, :a, :c, :a])
    changed = impurity_topology(T3NS(layout; flavor_order=order),
                                 changed_partition, changed_bath)
    @test hash(changed) != hash(t3ns)
    @test_throws ArgumentError impurity_topology(T3NS(layout),
                                                 Partition(:a => [:a], :b => [:b], :c => [:c]),
                                                 bath)
    _, zero_tooth_bath = _m5_bath(layout; owners=[:a, :a, :a, :a])
    zero_tooth_t3ns = impurity_topology(T3NS(layout; flavor_order=order),
                                         partition, zero_tooth_bath)
    @test Graft.is_t3ns(zero_tooth_t3ns;
                        physical=Symbol[:imp_a, :imp_b, :imp_c,
                                        :bath_a_1, :bath_a_2, :bath_a_3, :bath_a_4])
    @test count(node -> Graft.Trees.nodeid(zero_tooth_t3ns, node) == :t3ns_junction_1,
                1:Graft.Trees.nnodes(zero_tooth_t3ns)) == 0

    minimal_layout = FlavorLayout(
        [:r, :s, :t, :u],
        Dict(:r => :imp_r, :s => :imp_s, :t => :imp_t, :u => :imp_u),
        Dict(:imp_r => [:r], :imp_s => [:s], :imp_t => [:t], :imp_u => [:u]);
        basis=:minimal_t3ns,
    )
    minimal_partition = Partition(:r => [:r], :s => [:s], :t => [:t], :u => [:u])
    minimal_orbitals = BathOrbitals(
        [0.2], [ComplexF64[0.3]], [1], [1], [:r];
        layout=minimal_layout, partition=minimal_partition,
    )
    minimal_bath = DiscreteBath(minimal_layout, minimal_partition, minimal_orbitals;
                                statistics=:fermion)
    minimal_t3ns = impurity_topology(T3NS(minimal_layout), minimal_partition,
                                     minimal_bath)
    minimal_physical = Symbol[:imp_r, :imp_s, :imp_t, :imp_u, :bath_r_1]
    @test Graft.is_t3ns(minimal_t3ns; physical=minimal_physical)
    @test count(node -> !(Graft.Trees.nodeid(minimal_t3ns, node) in minimal_physical),
                1:Graft.Trees.nnodes(minimal_t3ns)) == 0

    junction_collision_layout = FlavorLayout(
        [:j, :d, :e, :f],
        Dict(:j => :t3ns_junction_1, :d => :imp_d, :e => :imp_e, :f => :imp_f),
        Dict(:t3ns_junction_1 => [:j], :imp_d => [:d], :imp_e => [:e], :imp_f => [:f]);
        basis=:junction_collision,
    )
    junction_collision_partition = Partition(:j => [:j], :d => [:d], :e => [:e], :f => [:f])
    junction_collision_orbitals = BathOrbitals(
        [0.2, 0.3, 0.4], [ComplexF64[0.3], ComplexF64[0.2], ComplexF64[0.1]],
        [1, 2, 3], [1, 2, 3], [:j, :d, :e];
        layout=junction_collision_layout, partition=junction_collision_partition,
    )
    junction_collision_bath = DiscreteBath(
        junction_collision_layout, junction_collision_partition,
        junction_collision_orbitals; statistics=:fermion,
    )
    @test_throws ArgumentError impurity_topology(
        T3NS(junction_collision_layout), junction_collision_partition,
        junction_collision_bath,
    )

    mounted = mount_bath(t3ns, bath)
    @test mounted isa AndersonBath
    @test mounted.topology == t3ns
    @test mounted.sites == expected_bath_sites
    @test mounted.anchors == (:imp_b, :imp_a, :imp_c, :imp_a)
    @test mounted.diagnostics.topology_source === :prebuilt
    @test mounted.diagnostics.retained_couplings == 9
    @test length(mounted.H) == 4 + 2 * 9
    @test count(term -> any(operator -> operator.site == :bath_b_1, term.ops),
                mounted.H.terms) == 5
    @test all(_is_descendant(mounted.topology, site, anchor)
              for (site, anchor) in zip(mounted.sites, mounted.anchors))
    changed_mounted = mount_bath(changed, changed_bath)
    @test changed_mounted.diagnostics.ownership_hash != mounted.diagnostics.ownership_hash
    wrong_owner_topology = TreeTopology(
        :imp_a,
        [:imp_a => :imp_b, :imp_b => :imp_c,
         :imp_a => :bath_b_1, :imp_a => :bath_a_2,
         :imp_c => :bath_c_3, :imp_a => :bath_a_4],
    )
    @test_throws ArgumentError AndersonBath(
        bath, wrong_owner_topology, _physical_dict(mounted),
        collect(mounted.sites), collect(mounted.anchors), mounted.H,
    )
    incorrect_anchors = collect(mounted.anchors)
    incorrect_anchors[1] = :imp_a
    @test_throws ArgumentError AndersonBath(
        bath, t3ns, _physical_dict(mounted), collect(mounted.sites),
        incorrect_anchors, mounted.H,
    )

    physical_spaces = _physical_dict(mounted)
    @test ttno_from_opsum(mounted.H, mounted.topology, physical_spaces;
                          hermitian=true) isa TTNO

    ftps_mounted = mount_bath(ftps, bath)
    @test _opsum_signature(ftps_mounted.H) == _opsum_signature(mounted.H)
    @test ftps_mounted.sites == mounted.sites
    @test ftps_mounted.anchors == mounted.anchors

    custom = TreeTopology(:imp_a, [:imp_a => :imp_b, :imp_b => :imp_c])
    custom_mounted = mount_bath(custom, bath)
    @test custom_mounted.diagnostics.topology_source === :extended
    @test custom_mounted.sites == expected_bath_sites
    @test length(custom_mounted.topology.ids) == length(custom.ids) + length(bath)

    shared_layout = FlavorLayout(
        [:up, :down],
        Dict(:up => :cluster, :down => :cluster),
        Dict(:cluster => [:up, :down]);
        basis=:shared_m5,
    )
    shared_partition = Partition(:spin => [:up, :down])
    shared_orbitals = BathOrbitals(
        [0.4], [ComplexF64[0.3, -0.2im]], [1], [1], [:up];
        layout=shared_layout, partition=shared_partition,
    )
    shared_bath = DiscreteBath(shared_layout, shared_partition, shared_orbitals;
                               statistics=:fermion)
    shared_down_orbitals = BathOrbitals(
        [0.4], [ComplexF64[0.3, -0.2im]], [1], [1], [:down];
        layout=shared_layout, partition=shared_partition,
    )
    shared_down_bath = DiscreteBath(shared_layout, shared_partition,
                                    shared_down_orbitals; statistics=:fermion)
    local_ops = FermionSiteOperators(shared_layout, :cluster)
    @test local_ops.modes == (:up, :down)
    @test dim(local_ops.P) == 4
    C_up = _charged_local_matrix(local_annihilator(local_ops, :up))
    C_down = _charged_local_matrix(local_annihilator(local_ops, :down))
    Cd_up = _charged_local_matrix(local_creator(local_ops, :up))
    Cd_down = _charged_local_matrix(local_creator(local_ops, :down))
    N_up = _neutral_local_matrix(local_number(local_ops, :up))
    N_down = _neutral_local_matrix(local_number(local_ops, :down))
    @test C_up[1, 3] == 1
    @test C_up[4, 2] == 1
    @test C_down[1, 4] == 1
    @test C_down[3, 2] == -1
    @test diag(N_up) == ComplexF64[0, 1, 1, 0]
    @test diag(N_down) == ComplexF64[0, 1, 0, 1]
    @test C_up * C_down + C_down * C_up ≈ zeros(ComplexF64, 4, 4)
    @test C_up * Cd_down + Cd_down * C_up ≈ zeros(ComplexF64, 4, 4)
    @test C_up * Cd_up + Cd_up * C_up ≈ Matrix{ComplexF64}(I, 4, 4)
    @test Cd_up * C_up ≈ N_up
    @test Cd_down * C_down ≈ N_down
    @test_throws ArgumentError impurity_topology(T3NS(shared_layout),
                                                 shared_partition, shared_bath)
    shared_mounted = mount_bath(TreeTopology(:cluster, Pair{Symbol,Symbol}[]),
                                shared_bath)
    @test shared_mounted isa AndersonBath
    @test length(shared_mounted.H) == 5
    @test shared_mounted.diagnostics.retained_couplings == 2
    @test_throws ArgumentError mount_bath(
        TreeTopology(:cluster, Pair{Symbol,Symbol}[]), shared_bath;
        site_labels=[:cluster],
    )
    shared_prebuilt = TreeTopology(:cluster, [:cluster => :shared_bath_site])
    shared_up_mounted = mount_bath(shared_prebuilt, shared_bath;
                                   site_labels=[:shared_bath_site])
    shared_down_mounted = mount_bath(shared_prebuilt, shared_down_bath;
                                     site_labels=[:shared_bath_site])
    @test shared_up_mounted.topology == shared_down_mounted.topology
    @test shared_up_mounted.diagnostics.ownership_hash !=
          shared_down_mounted.diagnostics.ownership_hash
    shared_physical = _physical_dict(shared_mounted)
    @test ttno_from_opsum(shared_mounted.H, shared_mounted.topology,
                          shared_physical; hermitian=true) isa TTNO

    bosonic = DiscreteBath(shared_layout, shared_partition, shared_orbitals;
                           statistics=:boson)
    F = fermion_ops_z2()
    manual_boson = BosonBath(
        bosonic,
        TreeTopology(:cluster, [:cluster => :manual_boson]),
        Dict(:cluster => local_ops.P, :manual_boson => F.P),
        [:manual_boson], [:cluster], OpSum(); diagnostics=(; source=:explicit),
    )
    @test manual_boson isa BosonBath
    @test_throws ArgumentError mount_bath(TreeTopology(:cluster, Pair{Symbol,Symbol}[]),
                                           bosonic)
end
