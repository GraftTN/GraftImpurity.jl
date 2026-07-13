using Test
using LinearAlgebra: norm
using GraftImpurity

function _kernel_layout(flavors::Vector{Symbol}; basis::Symbol=:kernel_basis)
    sites = Dict(flavor => :impurity for flavor in flavors)
    return FlavorLayout(flavors, sites, Dict(:impurity => flavors); basis)
end

function _block_residues(expansion::PoleExpansion, block_index_value::Int)
    indices = findall(==(block_index_value), expansion.poles.block_indices)
    return expansion.poles.poles[indices], expansion.poles.residues[indices]
end

function _expansion_value(expansion::PoleExpansion, block_index_value::Int, z)
    poles, residues = _block_residues(expansion, block_index_value)
    value = zeros(ComplexF64, size(first(residues))...)
    for (pole, residue) in zip(poles, residues)
        value .+= residue ./ (z - pole)
    end
    return value
end

function _bath_block_residue(bath::DiscreteBath, block_index_value::Int)
    flavors = block_flavors(
        bath_partition(bath), block_names(bath_partition(bath))[block_index_value],
    )
    residue = zeros(ComplexF64, length(flavors), length(flavors))
    for coupling_index in eachindex(bath.orbitals.energies)
        bath.orbitals.block_indices[coupling_index] == block_index_value || continue
        coupling = bath.orbitals.couplings[coupling_index]
        residue .+= coupling * coupling'
    end
    return residue
end

@testset "real-pole fitting kernels" begin
    scalar_layout = _kernel_layout([:charge]; basis=:scalar_kernel)
    scalar_partition = Partition(:charge => [:charge])
    frequencies = collect(-2.0:0.05:2.0)
    scalar_spectrum = ComplexF64[abs(frequency) <= 1 ? 0.5 : 0.0
                                  for frequency in frequencies]
    scalar_input = BathFitInput(
        scalar_layout, frequencies, :charge => scalar_spectrum;
        domain=:real_axis, statistics=:fermion,
    )

    allocated = BlockDiscretizationPlan(
        (-3.0, 3.0), [(-3.0, -2.0), (1.0, 3.0)], 5;
        forced_poles=[-2.5, 2.5], discarded_weight=0.02,
    )
    @test map(interval -> interval.modes, allocated.intervals) == (2, 3)
    @test allocated.intervals[1].forced_poles == (-2.5,)
    @test allocated.intervals[2].forced_poles == (2.5,)
    automatic = BlockDiscretizationPlan(
        frequencies, scalar_spectrum, 3; discarded_fraction=0.1,
    )
    @test automatic.weight_measure === :hermitian_trace
    @test automatic.discarded_weight <= 0.1 + 1e-12
    @test automatic.outer_bounds[1] > first(frequencies)
    @test automatic.outer_bounds[2] < last(frequencies)
    descending_automatic = BlockDiscretizationPlan(
        reverse(frequencies), reverse(scalar_spectrum), 3; discarded_fraction=0.1,
    )
    @test descending_automatic.outer_bounds == automatic.outer_bounds
    @test descending_automatic.discarded_weight ≈ automatic.discarded_weight
    @test_throws ArgumentError BlockDiscretizationPlan(
        frequencies, scalar_spectrum, 2;
        discarded_fraction=0, supports=[(-1.0, -0.4), (0.4, 1.0)],
    )
    weak_spectrum = fill(ComplexF64(1e-20), length(frequencies))
    @test_throws ArgumentError BlockDiscretizationPlan(
        frequencies, weak_spectrum, 2;
        discarded_fraction=0, supports=[(-1.0, -0.4), (0.4, 1.0)],
    )
    nonhermitian_tail = Matrix{ComplexF64}[
        frequency == 2.0 ? ComplexF64[0 1; 0 0] :
        abs(frequency) <= 0.2 ? ComplexF64[0.5 0; 0 0.5] :
        zeros(ComplexF64, 2, 2)
        for frequency in frequencies
    ]
    safe_automatic = BlockDiscretizationPlan(
        frequencies, nonhermitian_tail, 3; discarded_fraction=0,
    )
    @test safe_automatic.weight_measure === :frobenius_norm
    @test safe_automatic.outer_bounds[2] == last(frequencies)
    @test safe_automatic.discarded_weight ≈ 0.0
    forward_forced = SpectralInterval(-1.0, 1.0, 2; forced_poles=[0.8, 0.6])
    reverse_forced = SpectralInterval(-1.0, 1.0, 2; forced_poles=[0.6, 0.8])
    @test forward_forced.forced_poles == reverse_forced.forced_poles
    @test GraftImpurity._interval_bin_grid(forward_forced) ==
          GraftImpurity._interval_bin_grid(reverse_forced)
    close_forced = SpectralInterval(-1.0, 1.0, 3; forced_poles=[0.02, -0.02])
    _, close_poles = GraftImpurity._interval_bin_grid(close_forced)
    @test close_poles[1:2] ≈ [-0.02, 0.02]
    gap_frequencies = collect(-3.0:0.1:3.0)
    gap_input = BathFitInput(
        scalar_layout, gap_frequencies,
        :charge => fill(ComplexF64(0.25), length(gap_frequencies));
        domain=:real_axis, statistics=:fermion,
    )
    gap_plan = DiscretizationPlan(:charge => allocated; shared_grid=false)
    gap_expansion = real_pole_bath_fit(
        gap_input, QuadratureKernel(gap_plan), scalar_partition,
    )
    @test length(gap_expansion.poles) == 5
    @test -2.5 in gap_expansion.poles.poles
    @test 2.5 in gap_expansion.poles.poles
    @test all(pole -> pole <= -2 || pole >= 1, gap_expansion.poles.poles)

    scalar_plan = DiscretizationPlan(
        :charge => BlockDiscretizationPlan(
            (-1.0, 1.0), [(-1.0, 1.0)], 3; forced_poles=[0.0],
        );
        shared_grid=false,
    )
    quadrature = QuadratureKernel(scalar_plan)
    scalar_expansion = real_pole_bath_fit(
        scalar_input, quadrature, scalar_partition,
    )
    @test scalar_expansion.kernel === :quadrature
    @test scalar_expansion.trace.source_metadata == scalar_input.metadata
    @test isfinite(scalar_expansion.trace.fit_seconds) &&
          scalar_expansion.trace.fit_seconds >= 0
    @test 0.0 in scalar_expansion.poles.poles
    @test sum(real(residue[1, 1]) for residue in scalar_expansion.poles.residues) ≈ 1.0 atol=1e-12
    scalar_result = realize_bath(scalar_input, scalar_expansion, scalar_partition)
    @test scalar_result isa DiscretizationResult
    @test _bath_block_residue(scalar_result.bath, 1)[1, 1] ≈ 1.0 atol=1e-12
    @test scalar_result.report.reconstruction === nothing
    @test any(warning -> warning.code === :reconstruction_unavailable,
              scalar_result.report.warnings)
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            scalar_layout, frequencies, :charge => scalar_spectrum;
            domain=:real_axis, statistics=:boson,
        ),
        quadrature, scalar_partition,
    )

    matrix_layout = _kernel_layout([:up, :down]; basis=:matrix_kernel)
    matrix_partition = Partition(:spin => [:up, :down])
    vector = ComplexF64[1.0, 0.5im]
    spectral_residue = vector * vector'
    matrix_spectrum = Matrix{ComplexF64}[
        abs(frequency) <= 1 ? 0.5 .* spectral_residue : zeros(ComplexF64, 2, 2)
        for frequency in frequencies
    ]
    matrix_input = BathFitInput(
        matrix_layout, frequencies, :spin => matrix_spectrum;
        domain=:real_axis, statistics=:fermion,
    )
    matrix_plan = DiscretizationPlan(
        :spin => BlockDiscretizationPlan(
            (-1.0, -0.25), [( -1.0, -0.25)], 1;
            forced_poles=[-0.5],
        );
        shared_grid=true,
    )
    matrix_quadrature = real_pole_bath_fit(
        matrix_input, QuadratureKernel(matrix_plan), matrix_partition,
    )
    @test matrix_quadrature.poles.residues[1][1, 2] != 0
    matrix_result = realize_bath(
        matrix_input, matrix_quadrature, matrix_partition;
        orbital_order=(; spin=[:up, :down]),
    )
    @test matrix_result isa DiscretizationResult
    @test _bath_block_residue(matrix_result.bath, 1) ≈
          matrix_quadrature.poles.residues[1] atol=1e-12

    boundary_plan = DiscretizationPlan(
        :spin => BlockDiscretizationPlan(
            (-1.0, 1.0), [(-1.0, 1.0)], 2;
            forced_poles=[-0.5, 0.5],
        );
        shared_grid=true,
    )
    boundary = BoundaryFitKernel(
        boundary_plan; broadening=0.2, scan_scales=(1.0, 1.25),
    )
    boundary_expansion = real_pole_bath_fit(
        matrix_input, boundary, matrix_partition,
    )
    @test boundary_expansion.kernel === :boundary_fit
    @test boundary_expansion.trace.source_metadata == matrix_input.metadata
    @test isfinite(boundary_expansion.trace.fit_seconds) &&
          boundary_expansion.trace.fit_seconds >= 0
    @test length(boundary_expansion.trace.boundary_curve) == 2
    @test any(residue -> abs(residue[1, 2]) > 0,
              boundary_expansion.poles.residues)
    boundary_result = realize_bath(
        matrix_input, boundary_expansion, matrix_partition;
        orbital_order=(; spin=[:up, :down]),
    )
    @test boundary_result isa DiscretizationResult
    @test boundary_result.report.broadening == boundary.broadening
    @test length(boundary_result.report.blocks.spin.boundary_curve) == 2
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            matrix_layout, frequencies, :spin => matrix_spectrum;
            domain=:real_axis, statistics=:boson,
        ),
        boundary, matrix_partition,
    )

    gapped_boundary_plan = DiscretizationPlan(
        :spin => BlockDiscretizationPlan(
            (-1.0, 1.0), [(-1.0, -0.4), (0.4, 1.0)], 2;
            forced_poles=[-0.75, 0.75],
        );
        shared_grid=true,
    )
    gapped_boundary = BoundaryFitKernel(
        gapped_boundary_plan; broadening=0.2, scan_scales=(1.0, 1.25),
    )
    gapped_boundary_expansion = real_pole_bath_fit(
        matrix_input, gapped_boundary, matrix_partition,
    )
    @test all(candidate -> candidate.status === :evaluated,
              gapped_boundary_expansion.trace.boundary_curve)
    @test all(candidate -> begin
              intervals = candidate.plan.blocks.spin.intervals
              intervals[1].upper <= intervals[2].lower
          end, gapped_boundary_expansion.trace.boundary_curve)
    @test all(candidate -> candidate.plan.blocks.spin.intervals[1].forced_poles ==
                          (-0.75,),
              gapped_boundary_expansion.trace.boundary_curve)
    @test all(candidate -> candidate.plan.blocks.spin.discarded_weight > 0,
              gapped_boundary_expansion.trace.boundary_curve)

    separate_outer_plan = DiscretizationPlan(
        :spin => BlockDiscretizationPlan(
            [SpectralInterval(-1.0, 1.0, 1)]; outer_bounds=(-1.5, 1.5),
        );
        shared_grid=true,
    )
    @test_throws ArgumentError BoundaryFitKernel(separate_outer_plan; broadening=0.2)

    out_of_mesh_boundary = BoundaryFitKernel(
        boundary_plan; broadening=0.2, scan_scales=(1.0, 3.0),
    )
    out_of_mesh_expansion = real_pole_bath_fit(
        matrix_input, out_of_mesh_boundary, matrix_partition,
    )
    @test any(candidate -> candidate.status === :invalid,
              out_of_mesh_expansion.trace.boundary_curve)
    out_of_mesh_result = realize_bath(
        matrix_input, out_of_mesh_expansion, matrix_partition;
        orbital_order=(; spin=[:up, :down]),
    )
    @test any(warning -> warning.code === :boundary_candidate_invalid,
              out_of_mesh_result.report.warnings)

    singular_spectrum = Matrix{ComplexF64}[
        abs(frequency) <= 1 ? ComplexF64[0.5 0; 0 0] : zeros(ComplexF64, 2, 2)
        for frequency in frequencies
    ]
    singular_input = BathFitInput(
        matrix_layout, frequencies, :spin => singular_spectrum;
        domain=:real_axis, statistics=:fermion,
    )
    ordered_boundary = BoundaryFitKernel(
        DiscretizationPlan(
            :spin => BlockDiscretizationPlan((-1.0, 1.0), [(-1.0, 1.0)], 1);
            shared_grid=true,
        );
        broadening=0.2, orbital_order=(; spin=[:down, :up]),
    )
    ordered_expansion = real_pole_bath_fit(
        singular_input, ordered_boundary, matrix_partition,
    )
    @test first(ordered_expansion.trace.bin_diagnostics).pivots[1] == 0.0
    ordered_result = realize_bath(singular_input, ordered_expansion, matrix_partition)
    @test ordered_result isa DiscretizationResult
    @test ordered_result.bath.orbitals.associated_flavors == [:up]
    overridden_result = realize_bath(
        singular_input, ordered_expansion, matrix_partition;
        orbital_order=(; spin=[:up, :down]),
    )
    @test overridden_result.report.trace.realization_orbital_order ==
          (; spin=(:up, :down))
    @test overridden_result.report.diagnostics[1].pivots ≈ [1.0, 0.0]

    non_psd_spectrum = Matrix{ComplexF64}[
        abs(frequency) <= 1 ? ComplexF64[1 2; 2 1] : zeros(ComplexF64, 2, 2)
        for frequency in frequencies
    ]
    non_psd_input = BathFitInput(
        matrix_layout, frequencies, :spin => non_psd_spectrum;
        domain=:real_axis, statistics=:fermion,
    )
    non_psd_expansion = real_pole_bath_fit(
        non_psd_input, boundary, matrix_partition,
    )
    @test any(diagnostic -> diagnostic.status === :non_psd,
              non_psd_expansion.trace.bin_diagnostics)
    @test any(residue -> residue[1, 2] != 0,
              non_psd_expansion.poles.residues)
    non_psd_result = realize_bath(
        non_psd_input, non_psd_expansion, matrix_partition;
        orbital_order=(; spin=[:up, :down]),
    )
    @test non_psd_result isa NonMountablePoleFit

    retarded_frequencies = collect(-2.0:0.1:2.0)
    retarded_values = ComplexF64[
        1 / (frequency + 0.5 + 0.1im) + 0.5 / (frequency - 0.5 + 0.1im)
        for frequency in retarded_frequencies
    ]
    retarded_input = BathFitInput(
        scalar_layout, retarded_frequencies, :charge => retarded_values;
        domain=:real_axis, statistics=:fermion, metadata=(; component=:retarded),
    )
    retarded_plan = DiscretizationPlan(
        :charge => BlockDiscretizationPlan((-1.0, 1.0), [(-1.0, 1.0)], 2);
        shared_grid=false,
    )
    retarded_expansion = real_pole_bath_fit(
        retarded_input,
        BoundaryFitKernel(retarded_plan; broadening=0.1,
                          residue_solver=:least_squares),
        scalar_partition,
    )
    @test maximum(norm(_expansion_value(retarded_expansion, 1, frequency + 0.1im) .-
                       reshape(ComplexF64[retarded_value], 1, 1))
                  for (frequency, retarded_value) in
                  zip(retarded_frequencies, retarded_values)) < 1e-10

    matsubara_frequencies = collect(0.5:0.5:6.0)
    matsubara_values = ComplexF64[
        0.8 / (im * frequency + 0.75) + 0.4 / (im * frequency - 0.5)
        for frequency in matsubara_frequencies
    ]
    matsubara_input = BathFitInput(
        scalar_layout, matsubara_frequencies, :charge => matsubara_values;
        domain=:matsubara, statistics=:fermion,
    )
    pes_expansion = real_pole_bath_fit(
        matsubara_input,
        PESKernel(n_poles=2, solver=:sdp, maxiter=0,
                  min_support=4, max_support=4),
        scalar_partition,
    )
    @test pes_expansion.kernel === :pes
    @test pes_expansion.trace.source_metadata == matsubara_input.metadata
    @test isfinite(pes_expansion.trace.fit_seconds) &&
          pes_expansion.trace.fit_seconds >= 0
    @test pes_expansion.poles.block_indices == fill(1, length(pes_expansion.poles))
    @test length(pes_expansion.trace.fits) == 1
    pes_result = realize_bath(matsubara_input, pes_expansion, scalar_partition)
    @test pes_result isa DiscretizationResult
    @test maximum(norm(_expansion_value(pes_expansion, 1, im * frequency) .-
                       reshape(ComplexF64[value], 1, 1))
                  for (frequency, value) in zip(matsubara_frequencies, matsubara_values)) < 2e-2

    boson_energy = 0.75
    boson_residue = 0.6
    boson_values = ComplexF64[
        boson_energy * boson_residue / (im * frequency - boson_energy)
        for frequency in matsubara_frequencies
    ]
    boson_input = BathFitInput(
        scalar_layout, matsubara_frequencies, :charge => boson_values;
        domain=:matsubara, statistics=:boson,
    )
    boson_pes = real_pole_bath_fit(
        boson_input,
        PESKernel(n_poles=1, solver=:sdp, maxiter=0,
                  min_support=4, max_support=4),
        scalar_partition,
    )
    boson_result = realize_bath(boson_input, boson_pes, scalar_partition)
    @test boson_result isa DiscretizationResult
    @test boson_result.report.reconstruction isa BathFitInput
    @test boson_result.report.blocks.charge.residual.relative_l2 < 1e-6

    minipole_expansion = real_pole_bath_fit(
        matsubara_input, MiniPoleKernel(n_poles=2), scalar_partition,
    )
    @test minipole_expansion.kernel === :minipole
    @test minipole_expansion.trace.algorithm === :minipole_conformal_matrix_esprit
    @test minipole_expansion.trace.source_metadata == matsubara_input.metadata
    @test isfinite(minipole_expansion.trace.fit_seconds) &&
          minipole_expansion.trace.fit_seconds >= 0
    @test minipole_expansion.trace.fits[1].selected_poles == 2
    @test minipole_expansion.trace.fits[1].selected_attempt.training_error.relative_l2 < 1e-8
    @test sort(minipole_expansion.poles.poles) ≈ [-0.75, 0.5] atol=1e-8
    @test maximum(norm(_expansion_value(minipole_expansion, 1, im * frequency) .-
                       reshape(ComplexF64[value], 1, 1))
                  for (frequency, value) in zip(matsubara_frequencies, matsubara_values)) < 1e-8
    @test realize_bath(matsubara_input, minipole_expansion, scalar_partition) isa
          DiscretizationResult
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            scalar_layout, matsubara_frequencies, :charge => matsubara_values;
            domain=:matsubara, statistics=:boson,
        ),
        MiniPoleKernel(n_poles=2), scalar_partition,
    )
    heldout_minipole = real_pole_bath_fit(
        matsubara_input, MiniPoleKernel(n_poles=2, holdout_count=2),
        scalar_partition,
    )
    @test heldout_minipole.trace.fits[1].training_count == 10
    @test heldout_minipole.trace.fits[1].selected_attempt.holdout_error.relative_l2 < 1e-8
    tail_corrupted_values = copy(matsubara_values)
    tail_corrupted_values[end - 1:end] .+= ComplexF64[0.7 - 0.4im, -0.5 + 0.3im]
    tail_corrupted_input = BathFitInput(
        scalar_layout, matsubara_frequencies, :charge => tail_corrupted_values;
        domain=:matsubara, statistics=:fermion,
    )
    tail_heldout = real_pole_bath_fit(
        tail_corrupted_input, MiniPoleKernel(n_poles=2, holdout_count=2),
        scalar_partition,
    )
    @test tail_heldout.trace.fits[1].selected_attempt.training_error.relative_l2 < 1e-8
    @test tail_heldout.trace.fits[1].selected_attempt.holdout_error.relative_l2 > 0.1
    singular_map_values = ComplexF64[1 / (im * frequency + 1)
                                      for frequency in matsubara_frequencies]
    singular_map_input = BathFitInput(
        scalar_layout, matsubara_frequencies, :charge => singular_map_values;
        domain=:matsubara, statistics=:fermion,
    )
    singular_map_expansion = real_pole_bath_fit(
        singular_map_input,
        MiniPoleKernel(n_poles=1, conformal_scale=1.0), scalar_partition,
    )
    @test only(singular_map_expansion.poles.poles) ≈ -1.0 atol=1e-8
    nonuniform_matsubara = [0.5, 1.0, 1.75, 2.5]
    nonuniform_input = BathFitInput(
        scalar_layout, nonuniform_matsubara,
        :charge => ComplexF64[1 / (im * frequency + 1)
                              for frequency in nonuniform_matsubara];
        domain=:matsubara, statistics=:fermion,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        nonuniform_input, MiniPoleKernel(n_poles=1), scalar_partition,
    )
    overcomplete_minipole = real_pole_bath_fit(
        matsubara_input, MiniPoleKernel(n_poles=3), scalar_partition,
    )
    @test length(overcomplete_minipole.poles) == 2
    @test overcomplete_minipole.trace.fits[1].selected_poles == 2
    complex_pole_input = BathFitInput(
        scalar_layout, matsubara_frequencies,
        :charge => ComplexF64[1 / (im * frequency - (0.5 - 0.25im))
                              for frequency in matsubara_frequencies];
        domain=:matsubara, statistics=:fermion,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        complex_pole_input, MiniPoleKernel(n_poles=1), scalar_partition,
    )
    nearly_real_complex_input = BathFitInput(
        scalar_layout, matsubara_frequencies,
        :charge => ComplexF64[1 / (im * frequency - (0.5 - 0.01im))
                              for frequency in matsubara_frequencies];
        domain=:matsubara, statistics=:fermion,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        nearly_real_complex_input,
        MiniPoleKernel(n_poles=1, rank_tolerance=0.1), scalar_partition,
    )

    zero_scalar_input = BathFitInput(
        scalar_layout, matsubara_frequencies,
        :charge => zeros(ComplexF64, length(matsubara_frequencies));
        domain=:matsubara, statistics=:fermion,
    )
    zero_scalar_expansion = real_pole_bath_fit(
        zero_scalar_input, MiniPoleKernel(n_poles=1), scalar_partition,
    )
    @test length(zero_scalar_expansion.poles) == 0
    @test zero_scalar_expansion.trace.fits[1].selected_attempt.status === :zero_sequence
    @test realize_bath(zero_scalar_input, zero_scalar_expansion, scalar_partition) isa
          DiscretizationResult

    minipole_vectors = (ComplexF64[0.8, 0.3im], ComplexF64[0.2 - 0.4im, 0.6])
    minipole_energies = (-0.75, 0.5)
    minipole_matrix_values = Matrix{ComplexF64}[
        sum(vector * vector' ./ (im * frequency - energy)
            for (energy, vector) in zip(minipole_energies, minipole_vectors))
        for frequency in matsubara_frequencies
    ]
    minipole_matrix_input = BathFitInput(
        matrix_layout, matsubara_frequencies, :spin => minipole_matrix_values;
        domain=:matsubara, statistics=:fermion,
    )
    minipole_matrix_expansion = real_pole_bath_fit(
        minipole_matrix_input, MiniPoleKernel(n_poles=2), matrix_partition,
    )
    @test any(residue -> abs(residue[1, 2]) > 0,
              minipole_matrix_expansion.poles.residues)
    @test maximum(norm(_expansion_value(minipole_matrix_expansion, 1, im * frequency) .-
                       value)
                  for (frequency, value) in
                  zip(matsubara_frequencies, minipole_matrix_values)) < 1e-8
    @test realize_bath(
        minipole_matrix_input, minipole_matrix_expansion, matrix_partition;
        orbital_order=(; spin=[:up, :down]),
    ) isa DiscretizationResult
    zero_matrix_input = BathFitInput(
        matrix_layout, matsubara_frequencies,
        :spin => fill(zeros(ComplexF64, 2, 2), length(matsubara_frequencies));
        domain=:matsubara, statistics=:fermion,
    )
    zero_matrix_expansion = real_pole_bath_fit(
        zero_matrix_input, MiniPoleKernel(n_poles=1), matrix_partition,
    )
    @test length(zero_matrix_expansion.poles) == 0
    @test zero_matrix_expansion.trace.fits[1].selected_attempt.status === :zero_sequence
    indefinite_matsubara_values = Matrix{ComplexF64}[
        ComplexF64[1 2; 2 1] ./ (im * frequency + 0.4)
        for frequency in matsubara_frequencies
    ]
    indefinite_matsubara_input = BathFitInput(
        matrix_layout, matsubara_frequencies, :spin => indefinite_matsubara_values;
        domain=:matsubara, statistics=:fermion,
    )
    indefinite_minipole = real_pole_bath_fit(
        indefinite_matsubara_input, MiniPoleKernel(n_poles=1), matrix_partition,
    )
    @test indefinite_minipole.poles.residues[1][1, 2] != 0
    @test realize_bath(
        indefinite_matsubara_input, indefinite_minipole, matrix_partition;
        orbital_order=(; spin=[:up, :down]),
    ) isa NonMountablePoleFit

    mixed_layout = _kernel_layout([:charge, :up, :down]; basis=:mixed_kernel)
    mixed_partition = Partition(:charge => [:charge], :spin => [:up, :down])
    mixed_matsubara_input = BathFitInput(
        mixed_layout, matsubara_frequencies,
        :charge => zeros(ComplexF64, length(matsubara_frequencies)),
        :spin => minipole_matrix_values;
        domain=:matsubara, statistics=:fermion,
    )
    mixed_matsubara_expansion = real_pole_bath_fit(
        mixed_matsubara_input, MiniPoleKernel(n_poles=2), mixed_partition,
    )
    @test all(==(2), mixed_matsubara_expansion.poles.block_indices)
    @test mixed_matsubara_expansion.trace.fits[1].selected_poles == 0
    @test realize_bath(
        mixed_matsubara_input, mixed_matsubara_expansion, mixed_partition;
        orbital_order=(; charge=[:charge], spin=[:up, :down]),
    ) isa DiscretizationResult
    mixed_input = BathFitInput(
        mixed_layout, frequencies,
        :charge => scalar_spectrum,
        :spin => matrix_spectrum;
        domain=:real_axis, statistics=:fermion,
    )
    mixed_plan = DiscretizationPlan(
        :charge => BlockDiscretizationPlan((-1.0, 1.0), [(-1.0, 1.0)], 1),
        :spin => BlockDiscretizationPlan((-1.0, 1.0), [(-1.0, 1.0)], 1);
        shared_grid=true,
    )
    mixed_expansion = real_pole_bath_fit(
        mixed_input, QuadratureKernel(mixed_plan), mixed_partition,
    )
    @test mixed_expansion.poles.block_indices == [1, 2]
    @test mixed_expansion.poles.residues[1] isa Matrix{ComplexF64}
    @test mixed_expansion.poles.residues[2][1, 2] != 0
    @test realize_bath(mixed_input, mixed_expansion, mixed_partition) isa
          DiscretizationResult

    diagonal_layout = _kernel_layout([:left, :right]; basis=:diagonal_kernel)
    diagonal_partition = Partition(:left => [:left], :right => [:right])
    left_spectrum = ComplexF64[frequency < 0 ? 0.5 : 0.0 for frequency in frequencies]
    right_spectrum = ComplexF64[frequency > 0 ? 0.5 : 0.0 for frequency in frequencies]
    diagonal_input = BathFitInput(
        diagonal_layout, frequencies,
        :left => left_spectrum, :right => right_spectrum;
        domain=:real_axis, statistics=:fermion,
    )
    diagonal_plan = DiscretizationPlan(
        :left => BlockDiscretizationPlan((-1.0, -0.5), [(-1.0, -0.5)], 1),
        :right => BlockDiscretizationPlan((0.5, 1.0), [(0.5, 1.0)], 1);
        shared_grid=false,
    )
    diagonal_expansion = real_pole_bath_fit(
        diagonal_input, QuadratureKernel(diagonal_plan), diagonal_partition,
    )
    @test diagonal_expansion.poles.block_indices == [1, 2]
    @test diagonal_expansion.poles.poles[1] < 0 < diagonal_expansion.poles.poles[2]
end
