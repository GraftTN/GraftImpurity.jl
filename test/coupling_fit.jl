using Test
using LinearAlgebra: Hermitian, eigvals, norm
using GraftImpurity

struct _UnsupportedCouplingTie <: GraftImpurity.AbstractCouplingTieRelation end

function _coupling_layout(flavors::Vector{Symbol}; basis::Symbol=:coupling_basis)
    return FlavorLayout(
        flavors, Dict(flavor => :impurity for flavor in flavors),
        Dict(:impurity => flavors); basis,
    )
end

function _coupling_samples(energies, couplings, frequencies)
    return Matrix{ComplexF64}[
        sum(coupling * coupling' ./ (im * frequency - energy)
            for (energy, coupling) in zip(energies, couplings))
        for frequency in frequencies
    ]
end

function _coupling_expansion_value(expansion::PoleExpansion, block_index_value::Int,
                                   frequency::Float64)
    block = block_names(expansion.poles.partition)[block_index_value]
    dimension = length(block_flavors(expansion.poles.partition, block))
    value = zeros(ComplexF64, dimension, dimension)
    for index in eachindex(expansion.poles.poles)
        expansion.poles.block_indices[index] == block_index_value || continue
        value .+= expansion.poles.residues[index] ./
                  (im * frequency - expansion.poles.poles[index])
    end
    return value
end

@testset "direct coupling-space Matsubara fitting" begin
    layout = _coupling_layout([:up, :down]; basis=:coupling_matrix)
    partition = Partition(:spin => [:up, :down])
    frequencies = collect(0.15:0.15:4.2)
    energies = [-0.65, 0.45]
    couplings = Vector{ComplexF64}[
        ComplexF64[0.65 + 0.1im, 0.2 - 0.25im],
        ComplexF64[0.15 + 0.35im, 0.55 - 0.1im],
    ]
    samples = _coupling_samples(energies, couplings, frequencies)
    input = BathFitInput(
        layout, frequencies, :spin => samples;
        domain=:matsubara, statistics=:fermion, metadata=(; source=:coupling_fixture),
    )
    kernel = CouplingFitKernel(
        n_modes=2, alpha=1.0, frequency_window=(0.15, 3.0),
        energy_bounds=(-1.0, 1.0), allocation=SignedModeAllocation(1),
        maxiter=1_500, optimizer_tolerance=1e-9, fit_tolerance=1e-5,
    )
    expansion = real_pole_bath_fit(input, kernel, partition)
    @test expansion.kernel === :coupling_fit
    @test expansion.trace.source_metadata == input.metadata
    @test expansion.trace.fits[1].selected_modes == 2
    @test expansion.trace.fits[1].frequency_count < length(frequencies)
    @test expansion.trace.fits[1].error.relative_l2 < 1e-5
    @test count(<(0), expansion.poles.poles) == 1
    @test count(>(0), expansion.poles.poles) == 1
    @test any(residue -> abs(residue[1, 2]) > 1e-5,
              expansion.poles.residues)
    @test all(minimum(eigvals(Hermitian(residue))) >= -1e-10
              for residue in expansion.poles.residues)
    @test all(isapprox(residue, coupling * coupling'; atol=1e-10)
              for (residue, coupling) in zip(
                  expansion.poles.residues, expansion.trace.fits[1].couplings,
              ))
    @test expansion.poles.poles ≈ energies atol=2e-3
    @test all(isapprox(residue, coupling * coupling'; atol=2e-3)
              for (residue, coupling) in zip(expansion.poles.residues, couplings))
    @test maximum(norm(_coupling_expansion_value(expansion, 1, frequency) .- target)
                  for (frequency, target) in zip(frequencies, samples)) < 2e-4
    @test realize_bath(
        input, expansion, partition; orbital_order=(; spin=[:up, :down]),
    ) isa DiscretizationResult
    real_component_fit = real_pole_bath_fit(
        input,
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
            components=RealComponents(), maxiter=1_000,
            optimizer_tolerance=1e-8,
        ),
        partition,
    )
    @test all(all(iszero, imag.(coupling))
              for coupling in real_component_fit.trace.fits[1].couplings)

    scalar_layout = _coupling_layout([:charge]; basis=:coupling_scalar)
    scalar_partition = Partition(:charge => [:charge])
    scalar_energies = [-0.55, 0.35]
    scalar_couplings = Vector{ComplexF64}[ComplexF64[0.7], ComplexF64[0.35]]
    scalar_samples = _coupling_samples(scalar_energies, scalar_couplings, frequencies)
    scalar_input = BathFitInput(
        scalar_layout, frequencies,
        :charge => ComplexF64[only(sample) for sample in scalar_samples];
        domain=:matsubara, statistics=:fermion,
    )
    scalar_fit = real_pole_bath_fit(
        scalar_input,
        CouplingFitKernel(
            n_modes=2, alpha=0.5, energy_bounds=(-1.0, 1.0),
            allocation=SignedModeAllocation(1), maxiter=1_500,
            optimizer_tolerance=1e-9, fit_tolerance=1e-5,
        ),
        scalar_partition,
    )
    @test issorted(scalar_fit.poles.poles)
    @test scalar_fit.trace.fits[1].error.relative_l2 < 1e-5
    @test maximum(abs(_coupling_expansion_value(scalar_fit, 1, frequency)[1, 1] - target)
                  for (frequency, target) in
                  zip(frequencies, ComplexF64[only(sample) for sample in scalar_samples])) < 2e-4
    repeated = real_pole_bath_fit(input, kernel, partition)
    @test repeated.poles.poles ≈ expansion.poles.poles atol=1e-10
    @test all(isapprox(left, right; atol=1e-10)
              for (left, right) in zip(repeated.poles.residues, expansion.poles.residues))

    tied_layout = _coupling_layout([:left, :right]; basis=:coupling_tied)
    tied_partition = Partition(:left => [:left], :right => [:right])
    tied_energies = [-0.4]
    tied_couplings = Vector{ComplexF64}[ComplexF64[0.7]]
    tied_samples = _coupling_samples(tied_energies, tied_couplings, frequencies)
    tied_input = BathFitInput(
        tied_layout, frequencies, :left => tied_samples, :right => tied_samples;
        domain=:matsubara, statistics=:fermion,
    )
    tied_kernel = CouplingFitKernel(
        n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
        allocation=SignedModeAllocation(1), components=RealComponents(),
        maxiter=1_000, optimizer_tolerance=1e-10,
        fit_tolerance=1e-6,
        block_ties=(CouplingBlockTie(:left, :right, EqualTie()),),
    )
    tied = real_pole_bath_fit(tied_input, tied_kernel, tied_partition)
    @test tied.poles.block_indices == [1, 2]
    @test tied.poles.poles[1] ≈ tied.poles.poles[2] atol=1e-6
    @test tied.poles.residues[1] ≈ tied.poles.residues[2] atol=1e-8
    @test tied.trace.fits[2].status === :tied
    @test all(iszero(imag(only(coupling)))
              for coupling in tied.trace.fits[1].couplings)
    @test maximum(norm(_coupling_expansion_value(tied, 2, frequency) .- target)
                  for (frequency, target) in zip(frequencies, tied_samples)) < 2e-5
    @test realize_bath(tied_input, tied, tied_partition) isa DiscretizationResult

    zero_tied_samples = Matrix{ComplexF64}[
        zeros(ComplexF64, 1, 1) for _ in frequencies
    ]
    conflicting_tied_input = BathFitInput(
        tied_layout, frequencies,
        :left => tied_samples, :right => zero_tied_samples;
        domain=:matsubara, statistics=:fermion,
    )
    conflicting_tied = real_pole_bath_fit(
        conflicting_tied_input,
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
            allocation=SignedModeAllocation(1), components=RealComponents(),
            maxiter=1_000, optimizer_tolerance=1e-10,
            block_ties=(CouplingBlockTie(:left, :right, EqualTie()),),
        ),
        tied_partition,
    )
    @test conflicting_tied.trace.fits[1].source_error.relative_l2 > 0.1
    @test only(conflicting_tied.trace.fits[1].follower_errors).error.weighted_l2 > 0.1

    low_weight_samples = _coupling_samples(
        [-0.4], Vector{ComplexF64}[ComplexF64[1.0]], frequencies,
    )
    high_weight_samples = _coupling_samples(
        [-0.4], Vector{ComplexF64}[ComplexF64[10.0]], frequencies,
    )
    tie_quality_kernel = CouplingFitKernel(
        n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
        allocation=SignedModeAllocation(1), components=RealComponents(),
        maxiter=1_000, optimizer_tolerance=1e-10, fit_tolerance=0.8,
        block_ties=(CouplingBlockTie(:left, :right, EqualTie()),),
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            tied_layout, frequencies,
            :left => low_weight_samples, :right => high_weight_samples;
            domain=:matsubara, statistics=:fermion,
        ),
        tie_quality_kernel, tied_partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            tied_layout, frequencies,
            :left => high_weight_samples, :right => low_weight_samples;
            domain=:matsubara, statistics=:fermion,
        ),
        tie_quality_kernel, tied_partition,
    )

    conjugate_layout = _coupling_layout([:left_up, :left_down, :right_up, :right_down];
                                         basis=:coupling_conjugate)
    conjugate_partition = Partition(
        :left => [:left_up, :left_down], :right => [:right_up, :right_down],
    )
    conjugate_energies = [-0.35]
    conjugate_couplings = Vector{ComplexF64}[ComplexF64[0.55 + 0.15im,
                                                           0.2 - 0.3im]]
    conjugate_left = _coupling_samples(
        conjugate_energies, conjugate_couplings, frequencies,
    )
    conjugate_right = _coupling_samples(
        conjugate_energies,
        Vector{ComplexF64}[conj.(only(conjugate_couplings))], frequencies,
    )
    conjugate_input = BathFitInput(
        conjugate_layout, frequencies,
        :left => conjugate_left, :right => conjugate_right;
        domain=:matsubara, statistics=:fermion,
    )
    conjugate_fit = real_pole_bath_fit(
        conjugate_input,
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
            allocation=SignedModeAllocation(1), maxiter=1_000,
            optimizer_tolerance=1e-10, fit_tolerance=1e-5,
            block_ties=(CouplingBlockTie(:left, :right, ConjugateTie()),),
        ),
        conjugate_partition,
    )
    @test conjugate_fit.trace.fits[2].relation isa ConjugateTie
    @test conjugate_fit.trace.fits[1].error.relative_l2 < 1e-5
    @test conjugate_fit.poles.residues[2] ≈ conj.(conjugate_fit.poles.residues[1]) atol=1e-8
    @test maximum(norm(_coupling_expansion_value(conjugate_fit, 2, frequency) .- target)
                  for (frequency, target) in zip(frequencies, conjugate_right)) < 2e-4

    indefinite = Matrix{ComplexF64}[
        ComplexF64[1 2; 2 1] ./ (im * frequency + 0.3)
        for frequency in frequencies
    ]
    indefinite_input = BathFitInput(
        layout, frequencies, :spin => indefinite;
        domain=:matsubara, statistics=:fermion,
    )
    indefinite_fit = real_pole_bath_fit(
        indefinite_input,
        CouplingFitKernel(
            n_modes=1, alpha=1.0, energy_bounds=(-1.0, 1.0),
            maxiter=1_000, optimizer_tolerance=1e-8,
        ),
        partition,
    )
    @test any(residue -> abs(residue[1, 2]) > 1e-5,
              indefinite_fit.poles.residues)
    @test all(minimum(eigvals(Hermitian(residue))) >= -1e-10
              for residue in indefinite_fit.poles.residues)
    @test indefinite_fit.trace.fits[1].error.relative_l2 > 0.1
    @test realize_bath(indefinite_input, indefinite_fit, partition) isa
          DiscretizationResult
    @test_throws ArgumentError real_pole_bath_fit(
        indefinite_input,
        CouplingFitKernel(
            n_modes=1, alpha=1.0, energy_bounds=(-1.0, 1.0),
            maxiter=1_000, optimizer_tolerance=1e-8, fit_tolerance=0.1,
        ),
        partition,
    )

    endpoint_samples = _coupling_samples(
        [-1.0], Vector{ComplexF64}[ComplexF64[0.7]], frequencies,
    )
    endpoint_input = BathFitInput(
        scalar_layout, frequencies,
        :charge => ComplexF64[only(sample) for sample in endpoint_samples];
        domain=:matsubara, statistics=:fermion,
    )
    endpoint_fit = real_pole_bath_fit(
        endpoint_input,
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
            allocation=SignedModeAllocation(1), components=RealComponents(),
            maxiter=5_000, optimizer_tolerance=1e-10, fit_tolerance=1e-7,
        ),
        scalar_partition,
    )
    @test only(endpoint_fit.poles.poles) == -1.0
    @test only(endpoint_fit.trace.fits[1].boundary_snaps)
    @test endpoint_fit.trace.fits[1].error.relative_l2 < 1e-7
    @test maximum(abs(_coupling_expansion_value(endpoint_fit, 1, frequency)[1, 1] - target)
                  for (frequency, target) in
                  zip(frequencies, ComplexF64[only(sample) for sample in endpoint_samples])) < 1e-7

    reverse_order = collect(reverse(eachindex(frequencies)))
    unsorted_scalar_input = BathFitInput(
        scalar_layout, frequencies[reverse_order],
        :charge => ComplexF64[only(sample) for sample in scalar_samples][reverse_order];
        domain=:matsubara, statistics=:fermion,
    )
    unsorted_scalar_fit = real_pole_bath_fit(
        unsorted_scalar_input,
        CouplingFitKernel(
            n_modes=2, alpha=0.5, energy_bounds=(-1.0, 1.0),
            allocation=SignedModeAllocation(1), maxiter=1_500,
            optimizer_tolerance=1e-9, fit_tolerance=1e-5,
        ),
        scalar_partition,
    )
    @test unsorted_scalar_fit.trace.fits[1].error.relative_l2 < 1e-5

    zero_scalar_input = BathFitInput(
        scalar_layout, frequencies, :charge => zeros(ComplexF64, length(frequencies));
        domain=:matsubara, statistics=:fermion,
    )
    zero_scalar_fit = real_pole_bath_fit(
        zero_scalar_input, CouplingFitKernel(n_modes=1, alpha=0.0), scalar_partition,
    )
    @test isempty(zero_scalar_fit.poles.poles)
    @test zero_scalar_fit.trace.fits[1].status === :zero_sequence
    @test zero_scalar_fit.trace.allocation isa FreeModeAllocation
    @test zero_scalar_fit.trace.energy_bounds ==
          (-maximum(frequencies), maximum(frequencies))

    narrow_samples = _coupling_samples(
        [-5e-13], Vector{ComplexF64}[ComplexF64[0.7]], frequencies,
    )
    narrow_fit = real_pole_bath_fit(
        BathFitInput(
            scalar_layout, frequencies,
            :charge => ComplexF64[only(sample) for sample in narrow_samples];
            domain=:matsubara, statistics=:fermion,
        ),
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1e-12, 1e-12),
            allocation=SignedModeAllocation(1), components=RealComponents(),
            maxiter=100, optimizer_tolerance=1e-10,
        ),
        scalar_partition,
    )
    @test -1e-12 < only(narrow_fit.poles.poles) < 0

    positive_signed_fit = real_pole_bath_fit(
        scalar_input,
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(0.1, 1.0),
            allocation=SignedModeAllocation(0), components=RealComponents(),
            maxiter=100, optimizer_tolerance=1e-8,
        ),
        scalar_partition,
    )
    @test only(positive_signed_fit.poles.poles) > 0

    high_frequencies = Float64[
        floatmax(Float64) / 1.1, floatmax(Float64) / 1.2,
        floatmax(Float64) / 1.3, floatmax(Float64) / 1.4,
    ]
    high_frequency_fit = real_pole_bath_fit(
        BathFitInput(
            scalar_layout, high_frequencies,
            :charge => zeros(ComplexF64, length(high_frequencies));
            domain=:matsubara, statistics=:fermion,
        ),
        CouplingFitKernel(n_modes=1, alpha=0.0), scalar_partition,
    )
    @test high_frequency_fit.trace.energy_bounds ==
          (-floatmax(Float64) / 2, floatmax(Float64) / 2)
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            scalar_layout, high_frequencies,
            :charge => fill(2.0 + 0.0im, length(high_frequencies));
            domain=:matsubara, statistics=:fermion,
        ),
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
            components=RealComponents(),
        ),
        scalar_partition,
    )

    short_fit = real_pole_bath_fit(
        input,
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
            maxiter=1, optimizer_tolerance=1e-14,
        ),
        partition,
    )
    @test short_fit.trace.fits[1].status === :nonconverged
    @test !short_fit.trace.fits[1].optimizer.converged

    @test_throws ArgumentError CouplingFitKernel(n_modes=1, alpha=-1)
    @test_throws ArgumentError CouplingFitKernel(n_modes=1, alpha=1.1)
    @test_throws ArgumentError CouplingFitKernel(
        n_modes=1, allocation=SignedModeAllocation(2),
    )
    @test_throws ArgumentError CouplingFitKernel(
        n_modes=1, frequency_window=(-0.1, 1.0),
    )
    @test_throws ArgumentError CouplingFitKernel(
        n_modes=1, energy_bounds=(-floatmax(Float64), floatmax(Float64)),
    )
    @test_throws ArgumentError CouplingBlockTie(:spin, :spin)
    @test_throws ArgumentError CouplingBlockTie(:left, :right, _UnsupportedCouplingTie())
    @test_throws ArgumentError CouplingFitKernel(
        n_modes=1, block_ties=(CouplingBlockTie(:left, :right),
                                CouplingBlockTie(:spin, :right)),
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            layout, frequencies, :spin => samples;
            domain=:real_axis, statistics=:fermion,
        ),
        CouplingFitKernel(n_modes=1), partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            scalar_layout, vcat(0.0, frequencies[2:end]),
            :charge => ComplexF64[only(sample) for sample in scalar_samples];
            domain=:matsubara, statistics=:fermion,
        ),
        CouplingFitKernel(n_modes=1), scalar_partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            scalar_layout, [nextfloat(0.0), 0.2, 0.4, 0.6],
            :charge => zeros(ComplexF64, 4);
            domain=:matsubara, statistics=:fermion,
        ),
        CouplingFitKernel(n_modes=1, alpha=1.0), scalar_partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        input,
        CouplingFitKernel(n_modes=2, frequency_window=(0.15, 0.3)), partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        input,
        CouplingFitKernel(
            n_modes=2, energy_bounds=(0.1, 1.0),
            allocation=SignedModeAllocation(1),
        ),
        partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        tied_input,
        CouplingFitKernel(
            n_modes=1,
            block_ties=(CouplingBlockTie(:missing, :right),),
        ),
        tied_partition,
    )
    dimension_layout = _coupling_layout([:small, :wide_up, :wide_down];
                                         basis=:coupling_tie_dimensions)
    dimension_partition = Partition(
        :small => [:small], :wide => [:wide_up, :wide_down],
    )
    dimension_input = BathFitInput(
        dimension_layout, frequencies,
        :small => tied_samples, :wide => samples;
        domain=:matsubara, statistics=:fermion,
    )
    @test_throws DimensionMismatch real_pole_bath_fit(
        dimension_input,
        CouplingFitKernel(
            n_modes=1,
            block_ties=(CouplingBlockTie(:small, :wide),),
        ),
        dimension_partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        BathFitInput(
            layout, frequencies, :spin => samples;
            domain=:matsubara, statistics=:boson,
        ),
        CouplingFitKernel(n_modes=1), partition,
    )
    @test_throws ArgumentError real_pole_bath_fit(
        input,
        CouplingFitKernel(
            n_modes=1, alpha=0.0, energy_bounds=(-1.0, 1.0),
            components=RealComponents(), maxiter=1_000,
            optimizer_tolerance=1e-8, fit_tolerance=1e-12,
        ),
        partition,
    )
end
