using Test
using GraftImpurity
using GreenFunc

function _bathfit_report_layout()
    return FlavorLayout(
        [:charge, :up, :down],
        Dict(:charge => :impurity, :up => :impurity, :down => :impurity),
        Dict(:impurity => [:charge, :up, :down]);
        basis=:bathfit_report,
    )
end

function _bathfit_report_scalar_samples(
    energies::Vector{Float64}, residues::Vector{ComplexF64},
    points::Vector{ComplexF64},
)
    return ComplexF64[
        sum(residue / (point - energy)
            for (energy, residue) in zip(energies, residues))
        for point in points
    ]
end

function _bathfit_report_matrix_samples(
    energies::Vector{Float64}, residues::Vector{Matrix{ComplexF64}},
    points::Vector{ComplexF64},
)
    dimension = size(first(residues), 1)
    samples = Matrix{ComplexF64}[]
    for point in points
        sample = zeros(ComplexF64, dimension, dimension)
        for (energy, residue) in zip(energies, residues)
            sample .+= residue ./ (point - energy)
        end
        push!(samples, sample)
    end
    return samples
end

function _bathfit_report_matrix_data(samples::Vector{Matrix{ComplexF64}})
    dimension = size(first(samples), 1)
    data = Array{ComplexF64}(undef, dimension, dimension, length(samples))
    for (index, sample) in enumerate(samples)
        data[:, :, index] .= sample
    end
    return data
end

function _bathfit_report_fixture()
    layout = _bathfit_report_layout()
    partition = Partition(:charge => [:charge], :spin => [:up, :down])
    mesh = ImFreq(8.0, true; grid=[-2, -1, 0, 1])
    frequencies = Float64[mesh[index] for index in eachindex(mesh)]
    points = ComplexF64[im * frequency for frequency in frequencies]

    charge_energies = [-0.25, 0.25]
    charge_residues = ComplexF64[0.75, 0.125]
    spin_energies = [-0.75, 0.5]
    first_coupling = ComplexF64[0.8 + 0.2im, -0.25 + 0.4im]
    second_coupling = ComplexF64[0.2 - 0.35im, 0.6 + 0.1im]
    spin_residues = Matrix{ComplexF64}[
        first_coupling * first_coupling',
        second_coupling * second_coupling',
    ]
    charge_samples = _bathfit_report_scalar_samples(
        charge_energies, charge_residues, points,
    )
    spin_samples = _bathfit_report_matrix_samples(
        spin_energies, spin_residues, points,
    )
    charge_gf = Gf(
        mesh; data=charge_samples, statistics=true, component=:matsubara,
        metadata=(; block=:charge, fixture=:bathfit_report),
    )
    spin_gf = Gf(
        mesh; target_shape=(2, 2), data=_bathfit_report_matrix_data(spin_samples),
        statistics=true, component=:matsubara,
        target_labels=((:up, :down), (:up, :down)),
        metadata=(; block=:spin, fixture=:bathfit_report),
    )
    source = BlockGf(:charge => charge_gf, :spin => spin_gf)
    input = BathFitInput(layout, source; metadata=(; fixture=:bathfit_report))
    plan = DiscretizationPlan(
        :charge => BlockDiscretizationPlan(
            [SpectralInterval(-1.0, 1.0, 2)]; discarded_weight=0.125,
            weight_measure=:fixture_area,
        ),
        :spin => BlockDiscretizationPlan(
            [SpectralInterval(-1.0, 1.0, 2)]; discarded_weight=0.25,
            weight_measure=:fixture_area,
        );
        shared_grid=true,
    )
    raw = BlockRealPoles(
        layout,
        partition,
        vcat(charge_energies, spin_energies),
        Any[charge_residues..., spin_residues...],
        [1, 1, 2, 2];
        statistics=:fermion,
    )
    expansion = PoleExpansion(
        raw; kernel=:synthetic,
        trace=(; plan, source=:bathfit_report_fixture),
    )
    return (; layout, partition, mesh, frequencies, source, input, plan, expansion,
            charge_energies, charge_residues, spin_energies, spin_residues)
end

@testset "M4 concrete BathFitReport reconstruction and audit" begin
    fixture = _bathfit_report_fixture()
    result = realize_bath(
        fixture.input, fixture.expansion, fixture.partition;
        orbital_order=(; charge=[:charge], spin=[:up, :down]),
    )

    @test result isa DiscretizationResult
    report = result.report
    @test report isa BathFitReport
    @test report.source === fixture.input
    @test report.reconstruction isa BathFitInput
    @test report.source.source_template isa BlockGf
    @test report.reconstruction.source_template isa BlockGf
    @test report.source.source_template[:spin].data == fixture.source[:spin].data
    @test report.reconstruction.source_template[:spin].data ≈
          fixture.source[:spin].data atol=1e-12
    @test report.plan === fixture.plan
    @test report.kernel === :synthetic
    @test report.mountable
    @test Tuple(keys(report.blocks)) == (:charge, :spin)
    @test report.blocks.charge isa BathFitBlockReport
    @test report.blocks.spin isa BathFitBlockReport
    @test report.blocks.charge.pole_count == 2
    @test report.blocks.spin.pole_count == 2
    @test report.blocks.charge.mode_count == 2
    @test report.blocks.spin.mode_count == 2
    @test report.blocks.spin.max_spacing ≈ 1.25
    @test report.blocks.spin.revival_time ≈ 2pi / 1.25
    @test report.blocks.spin.minimum_residue_eigenvalue >= -1e-12
    @test report.blocks.spin.psd_cone_distance <= 1e-12
    @test report.blocks.spin.relative_psd_cone_distance <= 1e-12
    @test report.blocks.charge.discarded_weight == 0.125
    @test report.blocks.spin.discarded_weight == 0.25
    @test report.blocks.spin.weight_measure === :fixture_area
    for block in values(report.blocks)
        @test block.residual isa BathFitResidual
        @test block.residual.absolute <= 1e-12
        @test block.residual.maximum <= 1e-12
        @test block.residual.l2 <= 1e-12
        @test block.residual.relative_l2 <= 1e-12
    end
    @test all(diagnostic -> diagnostic isa PoleBinDiagnostic, report.diagnostics)
    @test length(report.diagnostics) == 4
    @test all(warning -> warning isa BathFitWarning, report.warnings)
    @test report.timing isa BathFitTiming
    for seconds in (
        report.timing.fit_seconds,
        report.timing.realization_seconds,
        report.timing.reconstruction_seconds,
    )
        @test seconds === nothing || (isfinite(seconds) && seconds >= 0)
    end

    reconstructed_input = reconstruct_hybridization(result.bath, fixture.input)
    @test reconstructed_input isa BathFitInput
    @test reconstructed_input.layout === fixture.layout
    @test reconstructed_input.domain === :matsubara
    @test reconstructed_input.statistics === :fermion
    @test reconstructed_input.frequencies == fixture.frequencies
    @test reconstructed_input.target_labels == fixture.input.target_labels
    for block in (:charge, :spin)
        for (actual, expected) in zip(
            getproperty(reconstructed_input.blocks, block),
            getproperty(fixture.input.blocks, block),
        )
            @test actual ≈ expected atol=1e-12
        end
    end

    reconstructed_blocks = reconstruct_hybridization(result.bath, fixture.source)
    @test reconstructed_blocks isa BlockGf
    @test Tuple(keys(reconstructed_blocks)) == (:charge, :spin)
    @test reconstructed_blocks[:charge].data ≈ fixture.source[:charge].data atol=1e-12
    @test reconstructed_blocks[:spin].data ≈ fixture.source[:spin].data atol=1e-12
    @test reconstructed_blocks[:spin].target_shape == (2, 2)
    @test reconstructed_blocks[:spin].target_labels == ((:up, :down), (:up, :down))
    @test reconstructed_blocks[:spin].statistics
    @test reconstructed_blocks[:spin].component === :matsubara
    @test reconstructed_blocks[:spin].temperature == fixture.source[:spin].temperature
    @test reconstructed_blocks[:spin].metadata == fixture.source[:spin].metadata

    reconstructed_spin = reconstruct_hybridization(
        result.bath, fixture.source[:spin]; block=:spin,
    )
    @test reconstructed_spin isa Gf
    @test reconstructed_spin.data ≈ fixture.source[:spin].data atol=1e-12
    @test_throws ArgumentError reconstruct_hybridization(
        result.bath, fixture.source[:spin]; block=:spin, broadening=0.2,
    )

    boson_mesh = ImFreq(8.0, false; grid=[0, 1, 2])
    boson_template = Gf(
        boson_mesh; target_shape=(2, 2),
        data=zeros(ComplexF64, 2, 2, length(boson_mesh)),
        statistics=false, component=:matsubara,
        target_labels=((:up, :down), (:up, :down)),
        metadata=(; fixture=:bathfit_report_boson),
    )
    boson_bath = DiscreteBath(
        fixture.layout, fixture.partition, result.bath.orbitals; statistics=:boson,
    )
    reconstructed_boson = reconstruct_hybridization(
        boson_bath, boson_template; block=:spin,
    )
    for index in eachindex(boson_mesh)
        z = im * boson_mesh[index]
        expected = iszero(z) ?
                   -sum(fixture.spin_residues) :
                   sum(energy .* residue ./ (z - energy)
                       for (energy, residue) in
                       zip(fixture.spin_energies, fixture.spin_residues))
        @test reconstructed_boson.data[:, :, index] ≈ expected atol=1e-12
    end
    @test !reconstructed_boson.statistics
    @test reconstructed_boson.metadata == boson_template.metadata

    real_mesh = ReFreq([-1.5, 0.0, 1.5])
    real_template = Gf(
        real_mesh; target_shape=(2, 2),
        data=zeros(ComplexF64, 2, 2, length(real_mesh)),
        statistics=true, component=:retarded,
        temperature=GreenFunc.ZeroTemperature(),
        target_labels=((:up, :down), (:up, :down)),
        metadata=(; fixture=:bathfit_report_real_axis),
    )
    real_points = ComplexF64[
        real_mesh[index] + 0.2im for index in eachindex(real_mesh)
    ]
    expected_real_samples = _bathfit_report_matrix_samples(
        fixture.spin_energies, fixture.spin_residues, real_points,
    )
    reconstructed_real = reconstruct_hybridization(
        result.bath, real_template; block=:spin, broadening=0.2,
    )
    @test reconstructed_real isa Gf
    for index in eachindex(expected_real_samples)
        @test reconstructed_real.data[:, :, index] ≈ expected_real_samples[index] atol=1e-12
    end
    @test reconstructed_real.target_labels == real_template.target_labels
    @test reconstructed_real.metadata == real_template.metadata
    @test_throws ArgumentError reconstruct_hybridization(
        result.bath, real_template; block=:spin,
    )
    @test_throws ArgumentError reconstruct_hybridization(
        result.bath, real_template; block=:spin, broadening=0.0,
    )

    eta = 0.2
    retarded_charge = _bathfit_report_scalar_samples(
        fixture.charge_energies, fixture.charge_residues,
        ComplexF64[real_mesh[index] + im * eta for index in eachindex(real_mesh)],
    )
    retarded_spin = _bathfit_report_matrix_samples(
        fixture.spin_energies, fixture.spin_residues,
        ComplexF64[real_mesh[index] + im * eta for index in eachindex(real_mesh)],
    )
    retarded_source = BlockGf(
        :charge => Gf(
            real_mesh; data=retarded_charge, statistics=true, component=:retarded,
            temperature=GreenFunc.ZeroTemperature(),
            metadata=(; fixture=:bathfit_report_retarded),
        ),
        :spin => Gf(
            real_mesh; target_shape=(2, 2),
            data=_bathfit_report_matrix_data(retarded_spin), statistics=true,
            component=:retarded, temperature=GreenFunc.ZeroTemperature(),
            target_labels=((:up, :down), (:up, :down)),
            metadata=(; fixture=:bathfit_report_retarded),
        ),
    )
    retarded_result = realize_bath(
        BathFitInput(fixture.layout, retarded_source), fixture.expansion,
        fixture.partition;
        orbital_order=(; charge=[:charge], spin=[:up, :down]), broadening=eta,
    )
    @test retarded_result.report.blocks.charge.spectral_weight_error <= 1e-12
    @test retarded_result.report.blocks.spin.spectral_weight_error <= 1e-12
    @test audit_bathfit(
        retarded_result.report, BathFitCriteria(max_spectral_weight_error=1e-10),
    ).passed
    spectral_charge = ComplexF64[
        (conj(value) - value) / (2pi * im) for value in retarded_charge
    ]
    spectral_spin = Matrix{ComplexF64}[
        (adjoint(value) - value) / (2pi * im) for value in retarded_spin
    ]
    spectral_source = BlockGf(
        :charge => Gf(
            real_mesh; data=spectral_charge, statistics=true, component=:spectral,
            temperature=GreenFunc.ZeroTemperature(),
            metadata=(; fixture=:bathfit_report_spectral),
        ),
        :spin => Gf(
            real_mesh; target_shape=(2, 2),
            data=_bathfit_report_matrix_data(spectral_spin), statistics=true,
            component=:spectral, temperature=GreenFunc.ZeroTemperature(),
            target_labels=((:up, :down), (:up, :down)),
            metadata=(; fixture=:bathfit_report_spectral),
        ),
    )
    spectral_input = BathFitInput(fixture.layout, spectral_source)
    spectral_result = realize_bath(
        spectral_input, fixture.expansion, fixture.partition;
        orbital_order=(; charge=[:charge], spin=[:up, :down]), broadening=eta,
    )
    @test spectral_result.report.broadening == eta
    @test spectral_result.report.reconstruction isa BathFitInput
    @test spectral_result.report.blocks.charge.spectral_weight_error <= 1e-12
    @test spectral_result.report.blocks.spin.spectral_weight_error <= 1e-12
    @test_throws ArgumentError realize_bath(
        spectral_input, fixture.expansion, fixture.partition;
        orbital_order=(; charge=[:charge], spin=[:up, :down]), broadening=0.0,
    )
    no_eta_result = realize_bath(
        spectral_input,
        PoleExpansion(fixture.expansion.poles; kernel=:synthetic,
                      trace=(; plan=fixture.plan, broadening=nothing)),
        fixture.partition;
        orbital_order=(; charge=[:charge], spin=[:up, :down]),
    )
    @test no_eta_result.report.reconstruction === nothing
    @test any(warning -> warning.code === :reconstruction_unavailable,
              no_eta_result.report.warnings)
    spacing_ok = audit_bathfit(
        spectral_result.report,
        BathFitCriteria(max_spacing_over_broadening=6.3),
    )
    @test spacing_ok.passed
    spacing_rejected = audit_bathfit(
        spectral_result.report,
        BathFitCriteria(max_spacing_over_broadening=6.0),
    )
    @test !spacing_rejected.passed
    @test any(item -> item.block === :spin &&
                     item.criterion === :spacing_over_broadening,
              spacing_rejected.violations)

    accepted = audit_bathfit(
        report,
        BathFitCriteria(
            max_absolute=1e-10,
            max_maximum=1e-10,
            max_l2=1e-10,
            max_relative_l2=1e-10,
            min_residue_eigenvalue=-1e-8,
            max_psd_cone_distance=1e-10,
            beta=8.0,
            max_beta_spacing=10.1,
            request_horizon=1.0,
            max_request_horizon_ratio=1.0,
            require_reconstruction=true,
            require_mountable=true,
        ),
    )
    @test accepted isa BathFitAudit
    @test accepted.passed
    @test isempty(accepted.violations)
    @test all(item -> item isa BathFitAuditItem && item.passed, accepted.items)

    rejected = audit_bathfit(
        report,
        BathFitCriteria(min_residue_eigenvalue=10.0, require_mountable=true),
    )
    @test rejected isa BathFitAudit
    @test !rejected.passed
    @test !isempty(rejected.violations)
    @test all(item -> item isa BathFitAuditItem && !item.passed,
              rejected.violations)
    @test any(item -> item.threshold == 10.0, rejected.violations)
    horizon_rejected = audit_bathfit(
        report, BathFitCriteria(request_horizon=10.0),
    )
    @test !horizon_rejected.passed
    @test any(item -> item.criterion === :request_horizon_to_revival,
              horizon_rejected.violations)

    nonmountable = realize_bath(
        fixture.input,
        PoleExpansion(
            BlockRealPoles(
                fixture.layout,
                fixture.partition,
                [-0.25, 0.5],
                Any[0.75, ComplexF64[1 2; 2 1]],
                [1, 2];
                statistics=:fermion,
            );
            kernel=:synthetic,
            trace=(; plan=fixture.plan, source=:nonmountable_report_fixture),
        ),
        fixture.partition;
        orbital_order=(; charge=[:charge], spin=[:up, :down]),
    )
    @test nonmountable isa NonMountablePoleFit
    @test nonmountable.report isa BathFitReport
    @test !nonmountable.report.mountable
    @test nonmountable.report.reconstruction === nothing
    @test nonmountable.report.blocks.spin.minimum_residue_eigenvalue < 0
    @test nonmountable.report.blocks.spin.psd_cone_distance > 0
    mountability_audit = audit_bathfit(
        nonmountable.report, BathFitCriteria(require_mountable=true),
    )
    @test !mountability_audit.passed
    @test !isempty(mountability_audit.violations)
    cone_audit = audit_bathfit(
        nonmountable.report, BathFitCriteria(max_psd_cone_distance=0.0),
    )
    @test !cone_audit.passed
end
