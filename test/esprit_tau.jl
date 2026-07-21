using Test
using LinearAlgebra: Hermitian, eigvals, norm
using GraftImpurity
import GreenFunc
import Graft

function _esprit_tau_layout(flavors::Vector{Symbol}; basis::Symbol=:esprit_tau_basis)
    return FlavorLayout(
        flavors, Dict(flavor => :impurity for flavor in flavors),
        Dict(:impurity => flavors); basis,
    )
end

function _esprit_tau_kernel(tau::Float64, energy::Float64, beta::Float64)
    return energy > 0 ?
        exp(-tau * energy) / (1 + exp(-beta * energy)) :
        exp(-(tau - beta) * energy) / (1 + exp(beta * energy))
end

function _esprit_tau_samples(energies, residues, taus, beta)
    return Matrix{ComplexF64}[
        -sum(residue .* _esprit_tau_kernel(tau, energy, beta)
             for (energy, residue) in zip(energies, residues))
        for tau in taus
    ]
end

function _esprit_tau_assert_decomposition(source::BathFitInput,
                                           reconstruction::BathFitInput,
                                           residual::BathFitInput;
                                           atol::Float64=1e-11)
    @test source.layout == residual.layout
    @test source.domain === residual.domain
    @test source.statistics === residual.statistics
    @test source.frequencies == residual.frequencies
    @test Tuple(keys(source.blocks)) == Tuple(keys(residual.blocks))
    @test source.target_labels == residual.target_labels
    @test source.metadata == residual.metadata
    for block in keys(source.blocks)
        for (target, fitted, remainder) in zip(
                getproperty(source.blocks, block),
                getproperty(reconstruction.blocks, block),
                getproperty(residual.blocks, block))
            @test target ≈ fitted + remainder atol=atol rtol=atol
        end
    end
end

@testset "imaginary-time ESPRIT bath-fitting kernel" begin
    @test_throws ArgumentError ESPRITTauKernel(n_poles=0)
    @test_throws ArgumentError ESPRITTauKernel(
        n_poles=1, pole_tolerance=0,
    )
    @test_throws ArgumentError ESPRITTauKernel(
        n_poles=1, projection_tolerance=0,
    )

    beta = 8.0
    taus = collect(range(0.0, beta; length=65))
    layout = _esprit_tau_layout([:up, :down]; basis=:esprit_tau_matrix)
    partition = Partition(:spin => [:up, :down])
    energies = [-0.7, 0.45]
    couplings = Vector{ComplexF64}[
        ComplexF64[0.65 + 0.15im, 0.2 - 0.3im],
        ComplexF64[0.1 + 0.25im, 0.55 - 0.05im],
    ]
    residues = Matrix{ComplexF64}[coupling * coupling' for coupling in couplings]
    samples = _esprit_tau_samples(energies, residues, taus, beta)
    input = BathFitInput(
        layout, taus, :spin => samples;
        domain=:imaginary_time, statistics=:fermion,
        metadata=(; source=:esprit_tau_fixture),
    )
    kernel = ESPRITTauKernel(
        n_poles=2, pole_tolerance=1e-8,
        projection_tolerance=1e-10, fit_tolerance=1e-7,
    )
    expansion = real_pole_bath_fit(input, kernel, partition)
    @test expansion.kernel === :esprit_tau
    @test expansion.trace.method === :imaginary_time_esprit
    @test expansion.trace.source_metadata == input.metadata
    @test isfinite(expansion.trace.fit_seconds) && expansion.trace.fit_seconds >= 0
    @test expansion.trace.fits[1].requested_poles == 2
    @test expansion.trace.fits[1].selected_poles == 2
    @test expansion.trace.fits[1].physical_error.relative_l2 < 1e-7
    @test expansion.poles.poles ≈ energies atol=1e-7 rtol=1e-7
    @test any(residue -> abs(residue[1, 2]) > 1e-5,
              expansion.poles.residues)
    @test all(minimum(eigvals(Hermitian(residue))) >= -1e-10
              for residue in expansion.poles.residues)

    result = realize_bath(
        input, expansion, partition; orbital_order=(; spin=[:up, :down]),
    )
    @test result isa DiscretizationResult
    @test result.report.reconstruction !== nothing
    @test result.report.kernel === :esprit_tau
    @test result.report.blocks.spin.residual.relative_l2 < 1e-7
    reconstructed = reconstruct_hybridization(result.bath, input)
    remainder = residual_hybridization(input, result.bath)
    report_remainder = residual_hybridization(result.report)
    _esprit_tau_assert_decomposition(input, reconstructed, remainder)
    @test all(isapprox(left, right; atol=1e-12, rtol=1e-12)
              for (left, right) in zip(
                  remainder.blocks.spin, report_remainder.blocks.spin,
              ))

    repeated = real_pole_bath_fit(input, kernel, partition)
    @test repeated.poles.poles ≈ expansion.poles.poles atol=1e-12 rtol=1e-12
    @test all(isapprox(left, right; atol=1e-12, rtol=1e-12)
              for (left, right) in zip(
                  repeated.poles.residues, expansion.poles.residues,
              ))

    partial = real_pole_bath_fit(
        input,
        ESPRITTauKernel(
            n_poles=1, pole_tolerance=1e-8, projection_tolerance=1e-10,
        ),
        partition,
    )
    partial_result = realize_bath(input, partial, partition)
    @test partial_result isa DiscretizationResult
    partial_reconstruction = reconstruct_hybridization(partial_result.bath, input)
    partial_residual = residual_hybridization(partial_result.report)
    @test any(norm(sample) > 1e-8 for sample in partial_residual.blocks.spin)
    _esprit_tau_assert_decomposition(
        input, partial_reconstruction, partial_residual,
    )

    scalar_layout = _esprit_tau_layout([:charge]; basis=:esprit_tau_typed)
    scalar_partition = Partition(:charge => [:charge])
    scalar_residues = Matrix{ComplexF64}[
        reshape(ComplexF64[0.49], 1, 1),
        reshape(ComplexF64[0.16], 1, 1),
    ]
    scalar_samples = _esprit_tau_samples(
        energies, scalar_residues, taus, beta,
    )
    tau_mesh = GreenFunc.ImTime(
        beta, true; Euv=4.0, rtol=1e-10, grid=taus,
    )
    source_gf = GreenFunc.Gf(
        tau_mesh;
        data=ComplexF64[only(sample) for sample in scalar_samples],
        statistics=true, component=:matsubara,
    )
    typed_input = BathFitInput(
        scalar_layout, source_gf, :charge; metadata=(; tag=:typed_tau),
    )
    @test typed_input.domain === :imaginary_time
    typed_expansion = real_pole_bath_fit(
        typed_input,
        ESPRITTauKernel(
            n_poles=2, pole_tolerance=1e-8,
            projection_tolerance=1e-10, fit_tolerance=1e-7,
        ),
        scalar_partition,
    )
    typed_result = realize_bath(typed_input, typed_expansion, scalar_partition)
    @test typed_result isa DiscretizationResult
    typed_residual = residual_hybridization(typed_result.report)
    @test typed_residual.source_template isa GreenFunc.Gf
    @test typed_residual.source_template.mesh == source_gf.mesh
    @test typed_residual.source_template.component === :matsubara
    @test typed_residual.source_template.temperature == source_gf.temperature
    @test typed_residual.metadata == typed_input.metadata

    topology = impurity_topology(
        T3NS(scalar_layout), scalar_partition, typed_result.bath,
    )
    mounted = mount_bath(
        topology, typed_result.bath; sector=ParticleNumberSector(),
    )
    operators = ImpurityOperators(
        scalar_layout; sector=ParticleNumberSector(),
    )
    interaction = DensityDensityInteraction(
        zeros(ComplexF64, 1, 1), scalar_layout,
    )
    lowered = lower_hamiltonian(
        mounted, interaction, operators;
        h_loc=ImpurityOneBody(zeros(ComplexF64, 1, 1), scalar_layout),
        compression_atol=1e-12,
    )
    @test lowered.operator isa Graft.TTNO

    nonuniform = copy(taus)
    nonuniform[20] += 1e-3
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            layout, nonuniform, :spin => samples;
            domain=:imaginary_time, statistics=:fermion,
        ),
        kernel, partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            layout, taus, :spin => samples;
            domain=:imaginary_time, statistics=:boson,
        ),
        kernel, partition,
    )
    bosonic_input = BathFitInput(
        scalar_layout, taus, :charge => scalar_samples;
        domain=:imaginary_time, statistics=:boson,
    )
    bosonic_bath = DiscreteBath(
        scalar_layout, scalar_partition, typed_result.bath.orbitals;
        statistics=:boson,
    )
    @test_throws ArgumentError reconstruct_hybridization(
        bosonic_bath, bosonic_input,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            layout, collect(0.1:0.1:1.0),
            :spin => samples[1:10];
            domain=:imaginary_time, statistics=:fermion,
        ),
        kernel, partition,
    )
end

@testset "residual_hybridization is fitter-independent" begin
    layout = _esprit_tau_layout([:charge]; basis=:common_residual)
    partition = Partition(:charge => [:charge])
    frequencies = collect(range(-1.0, 1.0; length=17))
    broadening = 0.15
    energy = 0.2
    residue = reshape(ComplexF64[0.36], 1, 1)
    retarded = ComplexF64[
        residue[1, 1] / (frequency - energy + im * broadening)
        for frequency in frequencies
    ]
    input = BathFitInput(
        layout, frequencies, :charge => retarded;
        domain=:real_axis, statistics=:fermion,
        metadata=(; component=:retarded, source=:ordinary_fit_fixture),
    )
    poles = BlockRealPoles(
        layout, partition, [energy], [residue], [1]; statistics=:fermion,
    )
    expansion = PoleExpansion(
        poles; kernel=:ordinary_fit_fixture,
        trace=(; plan=DiscretizationPlan(partition), broadening),
    )
    result = realize_bath(input, expansion, partition)
    @test result isa DiscretizationResult
    reconstruction = reconstruct_hybridization(
        result.bath, input; broadening,
    )
    residual = residual_hybridization(result.report)
    _esprit_tau_assert_decomposition(input, reconstruction, residual)
    @test maximum(norm, residual.blocks.charge) < 1e-12

    no_reconstruction = realize_bath(
        input,
        PoleExpansion(
            poles; kernel=:ordinary_fit_fixture,
            trace=(; plan=DiscretizationPlan(partition)),
        ),
        partition,
    )
    @test no_reconstruction isa DiscretizationResult
    @test no_reconstruction.report.reconstruction === nothing
    @test_throws ArgumentError residual_hybridization(no_reconstruction.report)

    wrong_grid = BathFitInput(
        layout, reverse(frequencies), :charge => reverse(retarded);
        domain=:real_axis, statistics=:fermion,
        metadata=(; component=:retarded, source=:ordinary_fit_fixture),
    )
    @test_throws ArgumentError residual_hybridization(input, wrong_grid)
end
