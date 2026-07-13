using Test
using Graft
using GraftImpurity
using Graft.TestUtils: dense_hamiltonian, to_dense, product_ttns
using Graft.Backend: FermionParity, ⊠, U1Irrep
using LinearAlgebra: diag, norm, Hermitian, eigvals

function _m6_shared_layout()
    return FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis=:m6_shared,
    )
end

function _m6_local_physical(operators::ImpurityOperators)
    return Dict(:impurity => site_operators(operators, :impurity).P)
end

function _m6_kanamori_layout()
    return FlavorLayout(
        [:a_up, :a_down, :b_up, :b_down],
        Dict(flavor => :impurity for flavor in [:a_up, :a_down, :b_up, :b_down]),
        Dict(:impurity => [:a_up, :a_down, :b_up, :b_down]);
        basis=:m6_kanamori,
    )
end

@testset "M6 interaction lowering" begin
    layout = _m6_shared_layout()
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    topology = TreeTopology(:impurity, Pair{Symbol,Symbol}[])
    physical = _m6_local_physical(operators)

    density = DensityDensityInteraction(ComplexF64[0 2; 2 0], layout)
    density_opsum = lower_interaction(density, operators, nothing)
    density_dense = dense_hamiltonian(density_opsum, topology, physical)
    @test diag(density_dense) == ComplexF64[0, 0, 0, 2]
    @test ttno_from_opsum(density_opsum, topology, physical; hermitian=true) isa TTNO

    diagonal_density = DensityDensityInteraction(
        ComplexF64[0.7 0; 0 1.1], layout,
    )
    diagonal_dense = dense_hamiltonian(
        lower_interaction(diagonal_density, operators, nothing), topology, physical,
    )
    @test diag(diagonal_dense) == ComplexF64[0, 0.7, 1.1, 1.8]
    diagonal_components = split_density_density(diagonal_density)
    @test diagonal_components isa DensityDensityDecomposition
    @test diagonal_components.one_body.matrix == ComplexF64[0.7 0; 0 1.1]
    @test all(iszero, diag(diagonal_components.interaction.U))

    bare = zeros(ComplexF64, 2, 2, 2, 2)
    bare[1, 2, 1, 2] = 2
    bare[2, 1, 2, 1] = 2
    full_bare = FullCoulombInteraction(bare, BareCoulombTensor(), layout)
    bare_dense = dense_hamiltonian(
        lower_interaction(full_bare, operators, nothing), topology, physical,
    )
    @test bare_dense ≈ density_dense atol=1e-12

    vertex = zeros(ComplexF64, 2, 2, 2, 2)
    vertex[1, 2, 1, 2] = 2
    vertex[2, 1, 1, 2] = -2
    vertex[1, 2, 2, 1] = -2
    vertex[2, 1, 2, 1] = 2
    full_vertex = FullCoulombInteraction(vertex, AntisymmetrizedVertex(), layout)
    vertex_dense = dense_hamiltonian(
        lower_interaction(full_vertex, operators, nothing), topology, physical,
    )
    @test vertex_dense ≈ density_dense atol=1e-12

    partition = Partition(:all => [:up, :down])
    orbitals = BathOrbitals(
        [0.3], [ComplexF64[0.2, 0.1im]], [1], [1], [:up];
        layout, partition,
    )
    bath = DiscreteBath(layout, partition, orbitals; statistics=:fermion)
    mounted = mount_bath(
        TreeTopology(:impurity, [:impurity => :bath_up_1]), bath;
        site_labels=[:bath_up_1],
    )
    h_loc = ImpurityOneBody(ComplexF64[0.1 0.02im; -0.02im -0.2], layout)
    soc = ImpurityOneBody(ComplexF64[0 0.03im; -0.03im 0], layout; label=:soc)
    assembled = lower_hamiltonian(
        mounted, density, operators;
        h_loc, soc, compression_atol=1e-12,
    )
    @test assembled.audit.hermiticity === :certified
    @test only(assembled.audit.abelian).status === :preserved
    @test assembled.compression.mode === :exact_rank
    mounted_physical = Dict(site => getproperty(mounted.phys, site)
                            for site in propertynames(mounted.phys))
    uncompressed_assembled = ttno_from_opsum(
        assembled.opsum, mounted.topology, mounted_physical; hermitian=true,
    )
    @test to_dense(assembled.operator) ≈ to_dense(uncompressed_assembled) atol=1e-10
    impurity_basis = ((FermionParity(0) ⊠ U1Irrep(0), 1, 0),
                      (FermionParity(1) ⊠ U1Irrep(1), 1, 1),
                      (FermionParity(1) ⊠ U1Irrep(1), 2, 1),
                      (FermionParity(0) ⊠ U1Irrep(2), 1, 2))
    bath_basis = ((FermionParity(0) ⊠ U1Irrep(0), 0),
                  (FermionParity(1) ⊠ U1Irrep(1), 1))
    hermiticity_states = Tuple{Int,Any}[]
    for (impurity_sector, impurity_index, impurity_number) in impurity_basis,
        (bath_sector, bath_number) in bath_basis
        state = product_ttns(
            ComplexF64, mounted.topology, mounted_physical,
            Dict(:impurity => impurity_sector => impurity_index,
                 :bath_up_1 => bath_sector),
        )
        push!(hermiticity_states, (impurity_number + bath_number, state))
    end
    hermiticity_images = [apply(assembled.operator, state)
                          for (_, state) in hermiticity_states]
    for left in eachindex(hermiticity_states), right in eachindex(hermiticity_states)
        hermiticity_states[left][1] == hermiticity_states[right][1] || continue
        forward = inner(hermiticity_states[left][2], hermiticity_images[right])
        reverse = inner(hermiticity_states[right][2], hermiticity_images[left])
        @test forward ≈ conj(reverse) atol=1e-10
    end

    @test_throws ArgumentError lower_hamiltonian(
        mounted, density, operators;
        h_loc, soc,
        symmetry=SymmetrySpec(layout; bath_owners=(:bath_up_1 => :down,)),
        compression_atol=1e-12,
    )
    @test_throws ArgumentError lower_hamiltonian(
        mounted, density, operators;
        h_loc,
        symmetry=SymmetrySpec(
            layout; abelian=(ChargeU1(layout), FlavorU1(:spin_z, [1.0, -1.0], layout)),
        ),
        compression_atol=1e-12,
    )
    tampered = mount_bath(
        TreeTopology(:impurity, [:impurity => :tampered_bath]), bath;
        site_labels=[:tampered_bath], sector=ParticleNumberSector(),
    )
    nonzero_term = findfirst(term -> !iszero(term.coeff), tampered.H.terms)
    tampered_term = tampered.H.terms[nonzero_term]
    tampered.H.terms[nonzero_term] = Term(
        tampered_term.coeff + 0.01, tampered_term.ops,
    )
    @test_throws ArgumentError lower_hamiltonian(
        tampered, density, operators; compression_atol=1e-12,
    )
    forged = AndersonBath(
        bath, mounted.topology, mounted_physical, [:bath_up_1], [:impurity], mounted.H,
    )
    @test forged.certificate === nothing
    @test_throws ArgumentError lower_hamiltonian(
        forged, density, operators; compression_atol=1e-12,
    )
    ownership_orbitals = BathOrbitals(
        [0.3], [ComplexF64[0.2, 0.1im]], [1], [1], [:up]; layout, partition,
    )
    ownership_bath = DiscreteBath(layout, partition, ownership_orbitals;
                                   statistics=:fermion)
    ownership_mounted = mount_bath(
        TreeTopology(:impurity, [:impurity => :ownership_bath]), ownership_bath;
        site_labels=[:ownership_bath], sector=ParticleNumberSector(),
    )
    bath_orbitals(ownership_bath).associated_flavors[1] = :down
    @test_throws ArgumentError lower_hamiltonian(
        ownership_mounted, density, operators; compression_atol=1e-12,
    )

    function certified_mount()
        certificate_orbitals = BathOrbitals(
            [0.3], [ComplexF64[0.2, 0.1im]], [1], [1], [:up]; layout, partition,
        )
        certificate_bath = DiscreteBath(
            layout, partition, certificate_orbitals; statistics=:fermion,
        )
        certificate_mounted = mount_bath(
            TreeTopology(:impurity, [:impurity => :certificate_bath]), certificate_bath;
            site_labels=[:certificate_bath], sector=ParticleNumberSector(),
        )
        return certificate_bath, certificate_mounted
    end
    for mutate! in (
        bath -> (bath_orbitals(bath).energies[1] += 0.1),
        bath -> (bath_orbitals(bath).couplings[1][1] += 0.1im),
        bath -> (bath_orbitals(bath).pole_indices[1] += 1),
        bath -> (bath_orbitals(bath).block_indices[1] += 1),
    )
        certificate_bath, certificate_mounted = certified_mount()
        mutate!(certificate_bath)
        @test_throws ArgumentError GraftImpurity._require_mounted_hamiltonian_integrity(
            certificate_mounted,
        )
    end
    _, topology_tampered = certified_mount()
    topology_tampered.topology.index[:certificate_bath] = 1
    @test_throws ArgumentError lower_hamiltonian(
        topology_tampered, density, operators; compression_atol=1e-12,
    )

    kanamori_layout = _m6_kanamori_layout()
    kanamori_operators = ImpurityOperators(
        kanamori_layout; sector=ParticleNumberSector(),
    )
    kanamori_topology = TreeTopology(:impurity, Pair{Symbol,Symbol}[])
    kanamori_physical = _m6_local_physical(kanamori_operators)
    flavor_map = KanamoriFlavorMap(
        kanamori_layout, [(:a_up, :a_down), (:b_up, :b_down)],
    )
    @test_throws UndefKeywordError KanamoriInteraction(
        4.0, 2.0, 0.5, kanamori_layout; spin_flip=false, pair_hopping=false,
    )
    kanamori_dense = Dict{Tuple{Bool,Bool},Matrix{ComplexF64}}()
    kanamori_values = Dict{Tuple{Bool,Bool},KanamoriInteraction}()
    for spin_flip in (false, true), pair_hopping in (false, true)
        interaction = KanamoriInteraction(
            4.0, 2.0, 0.5, kanamori_layout;
            flavor_map, spin_flip, pair_hopping,
        )
        lowered = lower_interaction(interaction, kanamori_operators, nothing)
        dense = dense_hamiltonian(lowered, kanamori_topology, kanamori_physical)
        kanamori_dense[(spin_flip, pair_hopping)] = dense
        kanamori_values[(spin_flip, pair_hopping)] = interaction
        @test dense ≈ dense' atol=1e-12
    end
    density_only = kanamori_dense[(false, false)]
    spin_only = kanamori_dense[(true, false)]
    pair_only = kanamori_dense[(false, true)]
    both = kanamori_dense[(true, true)]
    @test norm(spin_only - density_only) > 1e-10
    @test norm(pair_only - density_only) > 1e-10
    @test both ≈ spin_only + pair_only - density_only atol=1e-12
    @test hash(kanamori_values[(false, false)]) != hash(kanamori_values[(true, false)])
    @test length(lower_interaction(kanamori_values[(false, false)],
                                   kanamori_operators, nothing)) == 6
    @test length(lower_interaction(kanamori_values[(true, false)],
                                   kanamori_operators, nothing)) == 8
    @test length(lower_interaction(kanamori_values[(false, true)],
                                   kanamori_operators, nothing)) == 8
    @test length(lower_interaction(kanamori_values[(true, true)],
                                   kanamori_operators, nothing)) == 10

    rotated_layout = FlavorLayout(
        [:plus, :minus],
        Dict(:plus => :impurity, :minus => :impurity),
        Dict(:impurity => [:plus, :minus]);
        basis=:m6_rotated,
    )
    rotation = ComplexF64[1 1; 1 -1] / sqrt(2)
    @test_throws ArgumentError rotate_interaction(
        diagonal_density, rotation, rotated_layout,
    )
    rotated_interaction = rotate_interaction(density, rotation, rotated_layout)
    @test rotated_interaction isa FullCoulombInteraction
    @test rotated_interaction.convention isa AntisymmetrizedVertex
    rotated_onebody = rotate_one_body(h_loc, rotation, rotated_layout)
    rotated_operators = ImpurityOperators(
        rotated_layout; sector=ParticleNumberSector(),
    )
    rotated_physical = Dict(:impurity => site_operators(rotated_operators, :impurity).P)
    original_dense = dense_hamiltonian(
        lower_one_body(h_loc, operators, nothing) + density_opsum,
        topology, physical,
    )
    rotated_dense = dense_hamiltonian(
        lower_one_body(rotated_onebody, rotated_operators, nothing) +
        lower_interaction(rotated_interaction, rotated_operators, nothing),
        topology, rotated_physical,
    )
    @test eigvals(Hermitian(original_dense)) ≈ eigvals(Hermitian(rotated_dense)) atol=1e-10
    rotated_diagonal_onebody = rotate_one_body(
        diagonal_components.one_body, rotation, rotated_layout,
    )
    rotated_diagonal_interaction = rotate_interaction(
        diagonal_components.interaction, rotation, rotated_layout,
    )
    original_diagonal_dense = dense_hamiltonian(
        lower_one_body(diagonal_components.one_body, operators, nothing) +
        lower_interaction(diagonal_components.interaction, operators, nothing),
        topology, physical,
    )
    rotated_diagonal_dense = dense_hamiltonian(
        lower_one_body(rotated_diagonal_onebody, rotated_operators, nothing) +
        lower_interaction(rotated_diagonal_interaction, rotated_operators, nothing),
        topology, rotated_physical,
    )
    @test eigvals(Hermitian(original_diagonal_dense)) ≈
        eigvals(Hermitian(rotated_diagonal_dense)) atol=1e-10

    complex_rotation = ComplexF64[1 im; im 1] / sqrt(2)
    complex_onebody = ImpurityOneBody(
        ComplexF64[0.3 0.12 - 0.19im; 0.12 + 0.19im -0.4], layout;
        label=:complex_rotation,
    )
    complex_rotated_layout = FlavorLayout(
        [:complex_plus, :complex_minus],
        Dict(:complex_plus => :impurity, :complex_minus => :impurity),
        Dict(:impurity => [:complex_plus, :complex_minus]);
        basis=:m6_complex_rotated,
    )
    complex_rotated_operators = ImpurityOperators(
        complex_rotated_layout; sector=ParticleNumberSector(),
    )
    complex_rotated_physical = Dict(
        :impurity => site_operators(complex_rotated_operators, :impurity).P,
    )
    complex_rotated_onebody = rotate_one_body(
        complex_onebody, complex_rotation, complex_rotated_layout,
    )
    complex_rotated_interaction = rotate_interaction(
        density, complex_rotation, complex_rotated_layout,
    )
    complex_old_dense = dense_hamiltonian(
        lower_one_body(complex_onebody, operators, nothing) + density_opsum,
        topology, physical,
    )
    complex_new_dense = dense_hamiltonian(
        lower_one_body(complex_rotated_onebody, complex_rotated_operators, nothing) +
        lower_interaction(complex_rotated_interaction, complex_rotated_operators, nothing),
        topology, complex_rotated_physical,
    )
    fock_rotation = zeros(ComplexF64, 4, 4)
    fock_rotation[1, 1] = 1
    fock_rotation[2:3, 2:3] = complex_rotation
    fock_rotation[4, 4] = complex_rotation[1, 1] * complex_rotation[2, 2] -
                           complex_rotation[1, 2] * complex_rotation[2, 1]
    @test complex_new_dense ≈ fock_rotation' * complex_old_dense * fock_rotation atol=1e-10

    rotated_partition = Partition(:all => [:plus, :minus])
    rotated_bath = rotate_bath(
        bath, rotation, rotated_layout, rotated_partition;
        mode_blocks=[:all], associated_flavors=[:plus],
    )
    @test bath_orbitals(rotated_bath).energies == bath_orbitals(bath).energies
    @test bath_orbitals(rotated_bath).associated_flavors == [:plus]
    old_coupling = reshape(bath_orbitals(bath).couplings[1], :, 1)
    new_coupling = reshape(bath_orbitals(rotated_bath).couplings[1], :, 1)
    z = 0.7 + 0.4im
    old_delta = old_coupling * old_coupling' / (z - bath_orbitals(bath).energies[1])
    new_delta = new_coupling * new_coupling' /
        (z - bath_orbitals(rotated_bath).energies[1])
    @test new_delta ≈ rotation' * old_delta * rotation atol=1e-12
    split_rotated_partition = Partition(:plus => [:plus], :minus => [:minus])
    @test_throws ArgumentError rotate_bath(
        bath, rotation, rotated_layout, split_rotated_partition;
        mode_blocks=[:plus], associated_flavors=[:plus],
    )
    rotated_mounted = mount_bath(
        TreeTopology(:impurity, [:impurity => :bath_plus_1]), rotated_bath;
        site_labels=[:bath_plus_1], sector=ParticleNumberSector(),
    )
    rotated_complete = lower_hamiltonian(
        rotated_mounted, rotated_interaction, rotated_operators;
        h_loc=rotated_onebody,
        soc=rotate_one_body(soc, rotation, rotated_layout),
        compression_atol=1e-12,
    )
    @test eigvals(Hermitian(to_dense(assembled.operator))) ≈
        eigvals(Hermitian(to_dense(rotated_complete.operator))) atol=1e-10

    cross_layout = FlavorLayout(
        [:a, :b, :c, :d],
        Dict(:a => :site_a, :b => :site_b, :c => :site_c, :d => :site_d),
        Dict(:site_a => [:a], :site_b => [:b], :site_c => [:c], :site_d => [:d]);
        basis=:m6_cross_site,
    )
    cross_operators = ImpurityOperators(cross_layout; sector=ParticleNumberSector())
    cross_topology = TreeTopology(
        :site_a, [:site_a => :site_b, :site_b => :site_c, :site_c => :site_d],
    )
    cross_physical = Dict(site => site_operators(cross_operators, site).P
                          for site in layout_sites(cross_layout))
    cross_tensor = zeros(ComplexF64, 4, 4, 4, 4)
    coupling = 0.31 + 0.17im
    cross_tensor[1, 2, 3, 4] = 2 * coupling
    cross_tensor[3, 4, 1, 2] = 2 * conj(coupling)
    cross_interaction = FullCoulombInteraction(
        cross_tensor, BareCoulombTensor(), cross_layout,
    )
    cross_opsum = lower_interaction(cross_interaction, cross_operators, nothing)
    cross_ttno = ttno_from_opsum(cross_opsum, cross_topology, cross_physical;
                                 hermitian=true)
    vacuum = FermionParity(0) ⊠ U1Irrep(0)
    occupied = FermionParity(1) ⊠ U1Irrep(1)
    input_cd = product_ttns(
        ComplexF64, cross_topology, cross_physical,
        Dict(:site_a => vacuum, :site_b => vacuum,
             :site_c => occupied, :site_d => occupied),
    )
    output_ab = product_ttns(
        ComplexF64, cross_topology, cross_physical,
        Dict(:site_a => occupied, :site_b => occupied,
             :site_c => vacuum, :site_d => vacuum),
    )
    @test inner(output_ab, apply(cross_ttno, input_cd)) ≈ coupling atol=1e-12
    input_ab = product_ttns(
        ComplexF64, cross_topology, cross_physical,
        Dict(:site_a => occupied, :site_b => occupied,
             :site_c => vacuum, :site_d => vacuum),
    )
    output_cd = product_ttns(
        ComplexF64, cross_topology, cross_physical,
        Dict(:site_a => vacuum, :site_b => vacuum,
             :site_c => occupied, :site_d => occupied),
    )
    @test inner(output_cd, apply(cross_ttno, input_ab)) ≈ conj(coupling) atol=1e-12

    cross_vertex = zeros(ComplexF64, 4, 4, 4, 4)
    for (a, b, c, d, value) in ((1, 2, 3, 4, coupling),
                                (3, 4, 1, 2, conj(coupling)))
        cross_vertex[a, b, c, d] = value
        cross_vertex[b, a, c, d] = -value
        cross_vertex[a, b, d, c] = -value
        cross_vertex[b, a, d, c] = value
    end
    cross_vertex_ttno = ttno_from_opsum(
        lower_interaction(
            FullCoulombInteraction(
                cross_vertex, AntisymmetrizedVertex(), cross_layout,
            ), cross_operators, nothing,
        ),
        cross_topology, cross_physical; hermitian=true,
    )
    @test inner(output_ab, apply(cross_vertex_ttno, input_cd)) ≈ coupling atol=1e-12
    @test inner(output_cd, apply(cross_vertex_ttno, input_ab)) ≈ conj(coupling) atol=1e-12

    pair_layout = FlavorLayout(
        [:a, :b, :c, :d],
        Dict(:a => :left, :b => :left, :c => :right, :d => :right),
        Dict(:left => [:a, :b], :right => [:c, :d]);
        basis=:m6_pair_transfer,
    )
    pair_operators = ImpurityOperators(pair_layout; sector=ParticleNumberSector())
    pair_topology = TreeTopology(:left, [:left => :right])
    pair_physical = Dict(
        site => site_operators(pair_operators, site).P for site in layout_sites(pair_layout)
    )
    pair_tensor = zeros(ComplexF64, 4, 4, 4, 4)
    pair_coupling = 0.21 - 0.09im
    pair_tensor[1, 2, 3, 4] = 2 * pair_coupling
    pair_tensor[3, 4, 1, 2] = 2 * conj(pair_coupling)
    pair_opsum = lower_interaction(
        FullCoulombInteraction(pair_tensor, BareCoulombTensor(), pair_layout),
        pair_operators, nothing,
    )
    pair_left = only(filter(operator -> operator.site === :left,
                            first(pair_opsum).ops))
    pair_right = only(filter(operator -> operator.site === :right,
                             first(pair_opsum).ops))
    @test pair_left.charge == FermionParity(0) ⊠ U1Irrep(2)
    @test pair_right.charge == FermionParity(0) ⊠ U1Irrep(-2)
    pair_ttno = ttno_from_opsum(pair_opsum, pair_topology, pair_physical;
                                hermitian=true)
    pair_vacuum = FermionParity(0) ⊠ U1Irrep(0)
    pair_occupied = FermionParity(0) ⊠ U1Irrep(2)
    pair_input = product_ttns(
        ComplexF64, pair_topology, pair_physical,
        Dict(:left => pair_vacuum, :right => pair_occupied),
    )
    pair_output = product_ttns(
        ComplexF64, pair_topology, pair_physical,
        Dict(:left => pair_occupied, :right => pair_vacuum),
    )
    @test inner(pair_output, apply(pair_ttno, pair_input)) ≈ pair_coupling atol=1e-12
end
