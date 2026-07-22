using Test
using LinearAlgebra: Diagonal
using GraftImpurity
using GreenFunc

function _realization_layout()
    return FlavorLayout(
        [:up, :down],
        Dict(:up => :impurity, :down => :impurity),
        Dict(:impurity => [:up, :down]);
        basis=:realization_spin,
    )
end

function _realization_input(layout, partition)
    frequencies = [-3.0, -1.0, 1.0, 3.0]
    samples = [zeros(ComplexF64, 2, 2) for _ in frequencies]
    return BathFitInput(layout, frequencies, :spin => samples;
                        domain=:matsubara, statistics=:fermion)
end

function _realization_plan()
    interval = SpectralInterval(-2.0, 2.0, 1)
    return DiscretizationPlan(
        :spin => BlockDiscretizationPlan([interval]);
        shared_grid=true,
    )
end

function _local_residue(bath, pole_index)
    indices = findall(==(pole_index), bath.orbitals.pole_indices)
    dimension = length(bath_partition(bath).blocks.spin)
    residue = zeros(ComplexF64, dimension, dimension)
    for index in indices
        coupling = bath.orbitals.couplings[index]
        residue .+= coupling * coupling'
    end
    return residue
end

@testset "real-pole Hamiltonian realization" begin
    layout = _realization_layout()
    partition = Partition(:spin => [:up, :down])
    input = _realization_input(layout, partition)
    plan = _realization_plan()

    iw = ImFreq(8.0, true; grid=[-2, -1, 0, 1, 2])
    gf_data = zeros(ComplexF64, 2, 2, length(iw))
    gf_data[1, 2, :] .= 0.25im
    gf_data[2, 1, :] .= -0.25im
    gf = Gf(iw; target_shape=(2, 2), data=gf_data, statistics=true,
            component=:matsubara,
            target_labels=((:up, :down), (:up, :down)))
    greenfunc_input = BathFitInput(layout, BlockGf(:spin => gf))
    @test greenfunc_input.domain === :matsubara
    @test greenfunc_input.statistics === :fermion
    @test greenfunc_input.frequencies == Float64[iw[index] for index in eachindex(iw)]
    @test greenfunc_input.target_labels.spin == ((:up, :down), (:up, :down))
    @test GraftImpurity._validate_fit_input(greenfunc_input, partition) === greenfunc_input
    @test_throws ArgumentError BathFitInput(
        layout, gf, :spin; metadata=(; component=:spectral),
    )
    @test_throws ArgumentError BathFitInput(
        layout, BlockGf(:spin => gf);
        metadata=(; temperature=GraftImpurity.ZeroTemperature()),
    )
    bad_labels = Gf(iw; target_shape=(2, 2), data=gf_data, statistics=true,
                    component=:matsubara,
                    target_labels=((:down, :up), (:down, :up)))
    @test_throws ArgumentError GraftImpurity._validate_fit_input(
        BathFitInput(layout, BlockGf(:spin => bad_labels)), partition,
    )

    vector = ComplexF64[1.0, 0.5im]
    residue = vector * vector'
    raw = BlockRealPoles(layout, partition, [0.25], [residue], [1];
                          statistics=:fermion)
    expansion = PoleExpansion(raw; kernel=:synthetic,
                              trace=(; plan, source=:rank_one))
    realized = realize_bath(input, expansion, partition;
                            orbital_order=(; spin=[:up, :down]))
    @test realized isa DiscretizationResult
    @test realized.plan === plan
    @test length(realized.bath) == 1
    @test realized.bath.orbitals.associated_flavors == [:up]
    @test _local_residue(realized.bath, 1) ≈ residue atol=1e-12
    @test realized.report.diagnostics[1].status === :valid
    @test realized.report.diagnostics[1].minimum_eigenvalue >= -1e-12
    @test realized.plan.blocks.spin.intervals isa Tuple
    @test first(realized.plan.blocks.spin.intervals).forced_poles isa Tuple
    @test_throws ArgumentError realize_bath(
        input, expansion, partition;
        orbital_order=(; spin=[:up, :down]), broadening=0.1,
    )
    trace_broadening_result = realize_bath(
        input,
        PoleExpansion(expansion.poles; kernel=:synthetic,
                      trace=(; plan, broadening=0.1)),
        partition; orbital_order=(; spin=[:up, :down]),
    )
    @test trace_broadening_result isa DiscretizationResult
    @test trace_broadening_result.report.broadening === nothing
    mutable_order = Dict(:spin => [:up, :down])
    dictionary_order_result = realize_bath(
        input, expansion, partition; orbital_order=mutable_order,
    )
    mutable_order[:spin][1] = :down
    @test dictionary_order_result.report.trace.realization_orbital_order ==
          (; spin=(:up, :down))

    zero_pivot = ComplexF64[0 0; 0 1]
    singular = PoleExpansion(
        BlockRealPoles(layout, partition, [0.5], [zero_pivot], [1];
                       statistics=:fermion);
        kernel=:synthetic, trace=(; plan, source=:zero_pivot),
    )
    singular_result = realize_bath(
        input, singular, partition; orbital_order=(; spin=[:up, :down]),
    )
    @test singular_result isa DiscretizationResult
    @test length(singular_result.bath) == 1
    @test singular_result.bath.orbitals.associated_flavors == [:down]
    @test singular_result.report.diagnostics[1].pivots == [0.0, 1.0]
    @test _local_residue(singular_result.bath, 1) ≈ zero_pivot atol=1e-12

    tiny_layout = FlavorLayout(
        [:tiny], Dict(:tiny => :tiny_site), Dict(:tiny_site => [:tiny]);
        basis=:tiny_realization,
    )
    tiny_partition = Partition(:tiny => [:tiny])
    tiny = BlockRealPoles(tiny_layout, tiny_partition, [0.1], [1e-10], [1];
                           statistics=:fermion)
    @test length(factorize_residues(tiny)) == 1
    @test factorize_residues(tiny).couplings[1][1]^2 ≈ 1e-10
    tiny_input = BathFitInput(
        tiny_layout, [-1.0, 1.0], :tiny => ComplexF64[0.0, 0.0];
        domain=:matsubara, statistics=:fermion,
    )
    nearly_real = BlockRealPoles(
        tiny_layout, tiny_partition, [0.2], ComplexF64[1 + 1e-10im], [1];
        statistics=:fermion,
    )
    nearly_real_expansion = PoleExpansion(nearly_real; kernel=:synthetic)
    nearly_real_result = realize_bath(
        tiny_input, nearly_real_expansion, tiny_partition; rtol=1e-8,
    )
    @test nearly_real_result isa DiscretizationResult
    @test nearly_real_result.report.diagnostics[1].status === :numerical_symmetrization
    @test nearly_real_result.report.diagnostics[1].reconstruction_error ≈ 1e-10
    @test_throws ArgumentError factorize_residues(nearly_real; rtol=1e-8)

    reversed = PoleExpansion(
        BlockRealPoles(layout, partition, [0.5], [ComplexF64[1 0; 0 0]], [1];
                       statistics=:fermion);
        kernel=:synthetic, trace=(; plan, source=:reversed),
    )
    reversed_result = realize_bath(
        input, reversed, partition; orbital_order=(; spin=[:down, :up]),
    )
    @test reversed_result isa DiscretizationResult
    @test reversed_result.bath.orbitals.associated_flavors == [:up]
    @test reversed_result.report.diagnostics[1].pivots == [0.0, 1.0]

    lower = ComplexF64[1 0; 0.25im 1]
    diagonal = Diagonal([4.0, 1.0])
    permuted_residue = lower * diagonal * lower'
    native_residue = zeros(ComplexF64, 2, 2)
    native_residue[[2, 1], [2, 1]] .= permuted_residue
    full_rank = PoleExpansion(
        BlockRealPoles(layout, partition, [0.75], [native_residue], [1];
                       statistics=:fermion);
        kernel=:synthetic, trace=(; plan, source=:full_rank_reordered),
    )
    full_rank_result = realize_bath(
        input, full_rank, partition; orbital_order=(; spin=[:down, :up]),
    )
    @test full_rank_result isa DiscretizationResult
    @test full_rank_result.bath.orbitals.associated_flavors == [:down, :up]
    first_column = full_rank_result.bath.orbitals.couplings[1][[2, 1]]
    second_column = full_rank_result.bath.orbitals.couplings[2][[2, 1]]
    @test first_column ≈ ComplexF64[2.0, 0.5im] atol=1e-12
    @test second_column ≈ ComplexF64[0.0, 1.0] atol=1e-12
    @test _local_residue(full_rank_result.bath, 1) ≈ native_residue atol=1e-12
    @test full_rank_result.report.diagnostics[1].minimum_eigenvalue > 0

    nonhermitian = PoleExpansion(
        BlockRealPoles(layout, partition, [0.0], [ComplexF64[1 1; 0 1]], [1];
                       statistics=:fermion);
        kernel=:boundary, trace=(; plan, source=:nonhermitian),
    )
    nonhermitian_result = realize_bath(input, nonhermitian, partition;
                                        orbital_order=(; spin=[:up, :down]))
    @test nonhermitian_result isa NonMountablePoleFit
    @test nonhermitian_result.expansion === nonhermitian
    @test nonhermitian_result.report.diagnostics[1].status === :nonhermitian
    @test nonhermitian_result.expansion.poles.residues[1][1, 2] == 1
    @test nonhermitian_result.expansion.poles.residues[1][2, 1] == 0

    large_nonhermitian = PoleExpansion(
        BlockRealPoles(
            layout, partition, [0.0], [ComplexF64[1e12 2e12; 1e12 1e12]], [1];
            statistics=:fermion,
        );
        kernel=:boundary, trace=(; plan, source=:large_nonhermitian),
    )
    large_nonhermitian_result = realize_bath(
        input, large_nonhermitian, partition;
        orbital_order=(; spin=[:up, :down]),
    )
    @test large_nonhermitian_result isa NonMountablePoleFit
    @test large_nonhermitian_result.report.diagnostics[1].status === :nonhermitian
    @test_throws ArgumentError realize_bath(
        input, expansion, partition; orbital_order=(; spin=[:up, :down]), atol=Inf,
    )

    indefinite = PoleExpansion(
        BlockRealPoles(layout, partition, [0.0], [ComplexF64[1 2; 2 1]], [1];
                       statistics=:fermion);
        kernel=:boundary, trace=(; plan, source=:indefinite),
    )
    indefinite_result = realize_bath(input, indefinite, partition;
                                      orbital_order=(; spin=[:up, :down]))
    @test indefinite_result isa NonMountablePoleFit
    @test indefinite_result.report.diagnostics[1].status === :non_psd
    @test indefinite_result.report.diagnostics[1].minimum_eigenvalue < 0

    mixed_layout = FlavorLayout(
        [:left, :up, :down],
        Dict(:left => :left_site, :up => :impurity, :down => :impurity),
        Dict(:left_site => [:left], :impurity => [:up, :down]);
        basis=:mixed_realization,
    )
    mixed_partition = Partition(:left => [:left], :spin => [:up, :down])
    mixed = BlockRealPoles(
        mixed_layout,
        mixed_partition,
        [-0.2, 0.4],
        Any[0.25, ComplexF64[0 0; 0 1]],
        [1, 2];
        statistics=:fermion,
    )
    @test eltype(mixed.residues) == Union{Float64,ComplexF64,Matrix{ComplexF64}}
    @test length(factorize_residues(
        mixed; orbital_order=(; left=[:left], spin=[:up, :down]),
    )) == 2
end
