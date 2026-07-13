using Test
using Graft
using GraftImpurity

@testset "M6 one-body symmetry audit" begin
    layout = FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis=:m6_symmetry,
    )
    operators = ImpurityOperators(layout; sector=ParticleNumberSector())
    h_loc = ImpurityOneBody(ComplexF64[0.2 0; 0 -0.1], layout)
    generic_soc = ImpurityOneBody(
        ComplexF64[0 0.3im; -0.3im 0], layout; label=:generic_soc,
    )
    spin_z = FlavorU1(:spin_z, [1.0, -1.0], layout)
    n_up = FlavorU1(:n_up, [1.0, 0.0], layout)
    generic_spec = SymmetrySpec(
        layout;
        abelian=(ChargeU1(layout), spin_z, n_up),
        nonabelian=(SU2Reduce(
            layout; name=:spin_su2, axial_generator=spin_z,
        ),),
    )
    generic_audit = audit_symmetry(
        one_body_opsum(h_loc, generic_soc, operators, generic_spec), generic_spec,
    )
    @test generic_audit.hermiticity === :unverified
    @test generic_audit.abelian[1].name === :charge
    @test generic_audit.abelian[1].status === :preserved
    @test generic_audit.abelian[2].status === :broken
    @test generic_audit.abelian[3].status === :broken
    @test only(generic_audit.nonabelian).status === :broken
    @test only(generic_audit.nonabelian).lowering_status === :unsupported

    cancelling_h = ImpurityOneBody(
        ComplexF64[0 0.25; 0.25 0], layout; label=:cancelling_h,
    )
    cancelling_soc = ImpurityOneBody(
        ComplexF64[0 -0.25; -0.25 0], layout; label=:cancelling_soc,
    )
    cancellation_spec = SymmetrySpec(layout; abelian=(spin_z,))
    @test only(audit_symmetry(
        one_body_opsum(cancelling_h, nothing, operators, cancellation_spec),
        cancellation_spec,
    ).abelian).status === :broken
    @test only(audit_symmetry(
        one_body_opsum(cancelling_h, cancelling_soc, operators, cancellation_spec),
        cancellation_spec,
    ).abelian).status === :preserved

    collinear_soc = ImpurityOneBody(
        ComplexF64[0.4 0; 0 -0.4], layout; label=:ising_soc,
    )
    axial_spec = SymmetrySpec(layout; abelian=(ChargeU1(layout), spin_z))
    axial_audit = audit_symmetry(
        one_body_opsum(h_loc, collinear_soc, operators, axial_spec), axial_spec,
    )
    @test all(item -> item.status === :preserved, axial_audit.abelian)
    @test axial_audit.abelian[2].lowering_status === :audit_only

    su2_candidate = SU2Reduce(
        layout; name=:total_j, axial_generator=spin_z,
    )
    su2_candidate_spec = SymmetrySpec(layout; nonabelian=(su2_candidate,))
    candidate_audit = audit_symmetry(
        one_body_opsum(h_loc, nothing, operators, su2_candidate_spec),
        su2_candidate_spec,
    )
    @test only(candidate_audit.nonabelian).status === :candidate
    @test only(candidate_audit.nonabelian).lowering_status === :unsupported
    metadata_free = SymmetrySpec(layout; nonabelian=(SU2Reduce(layout),))
    @test only(audit_symmetry(
        one_body_opsum(h_loc, nothing, operators, metadata_free), metadata_free,
    ).nonabelian).status === :unavailable

    partition = Partition(:all => [:up, :down])
    axial_orbitals = BathOrbitals(
        [0.4], [ComplexF64[0.2, 0]], [1], [1], [:up]; layout, partition,
    )
    axial_bath = DiscreteBath(layout, partition, axial_orbitals; statistics=:fermion)
    axial_mounted = mount_bath(
        TreeTopology(:impurity, [:impurity => :axial_bath]), axial_bath;
        site_labels=[:axial_bath], sector=ParticleNumberSector(),
    )
    axial_full_spec = SymmetrySpec(
        layout;
        abelian=(ChargeU1(layout), spin_z),
        bath_owners=(:axial_bath => :up,),
    )
    axial_full_audit = audit_symmetry(axial_mounted.H, axial_full_spec)
    @test all(item -> item.status === :preserved, axial_full_audit.abelian)
    zero_density = DensityDensityInteraction(zeros(ComplexF64, 2, 2), layout)
    @test_throws ArgumentError lower_hamiltonian(
        axial_mounted, zero_density, operators;
        symmetry=axial_full_spec, compression_atol=1e-12,
    )
    @test_throws ArgumentError lower_hamiltonian(
        axial_mounted, zero_density, operators;
        symmetry=SymmetrySpec(
            layout;
            nonabelian=(SU2Reduce(
                layout; name=:total_j, axial_generator=spin_z,
            ),),
        ),
        compression_atol=1e-12,
    )

    anisotropic_orbitals = BathOrbitals(
        [0.4], [ComplexF64[0, 0.2]], [1], [1], [:up]; layout, partition,
    )
    anisotropic_bath = DiscreteBath(
        layout, partition, anisotropic_orbitals; statistics=:fermion,
    )
    anisotropic_mounted = mount_bath(
        TreeTopology(:impurity, [:impurity => :anisotropic_bath]), anisotropic_bath;
        site_labels=[:anisotropic_bath], sector=ParticleNumberSector(),
    )
    anisotropic_spec = SymmetrySpec(
        layout;
        abelian=(ChargeU1(layout), spin_z),
        bath_owners=(:anisotropic_bath => :up,),
    )
    anisotropic_audit = audit_symmetry(anisotropic_mounted.H, anisotropic_spec)
    @test anisotropic_audit.abelian[1].status === :preserved
    @test anisotropic_audit.abelian[2].status === :broken

    delimiter_layout = FlavorLayout(
        [:orb__up, :orb__down],
        Dict(:orb__up => :impurity, :orb__down => :impurity),
        Dict(:impurity => [:orb__up, :orb__down]);
        basis=:m6_delimiter,
    )
    delimiter_operators = ImpurityOperators(
        delimiter_layout; sector=ParticleNumberSector(),
    )
    delimiter_axis = FlavorU1(:delimiter_axis, [1.0, -1.0], delimiter_layout)
    delimiter_spec = SymmetrySpec(
        delimiter_layout; abelian=(ChargeU1(delimiter_layout), delimiter_axis),
    )
    delimiter_h = ImpurityOneBody(
        ComplexF64[0 0.2; 0.2 0], delimiter_layout,
    )
    delimiter_audit = audit_symmetry(
        one_body_opsum(delimiter_h, nothing, delimiter_operators, delimiter_spec),
        delimiter_spec,
    )
    @test delimiter_audit.abelian[1].status === :preserved
    @test delimiter_audit.abelian[2].status === :broken

    tiny_breaking = ImpurityOneBody(
        ComplexF64[0 1e-16; 1e-16 0], layout; label=:tiny_breaking,
    )
    tiny_opsum = one_body_opsum(tiny_breaking, nothing, operators, axial_spec)
    @test only(audit_symmetry(tiny_opsum, axial_spec).abelian[2:2]).status === :broken
    @test only(audit_symmetry(tiny_opsum, axial_spec; tolerance=1e-15).abelian[2:2]).status === :preserved
    @test_throws ArgumentError SymmetrySpec(
        layout; abelian=(ChargeU1(layout), FlavorU1(:charge, [1.0, 1.0], layout)),
    )
end
