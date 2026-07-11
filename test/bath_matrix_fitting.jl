using Test
using GraftImpurity
using LinearAlgebra: Hermitian, I, eigmin, ishermitian, norm

@testset "bath value-object invariants" begin
    source_blocks = [[:a], [:b]]
    P = Partition(source_blocks)
    original_hash = hash(P)
    source_blocks[1][1] = :mutated
    push!(source_blocks, [:c])
    @test P.blocks == [[:a], [:b]]
    @test hash(P) == original_hash
    exposed_blocks = P.blocks
    push!(exposed_blocks[1], :mutated_copy)
    @test P.blocks == [[:a], [:b]]
    @test hash(P) == original_hash
    @test P == Partition([[:a], [:b]])
    @test_throws ArgumentError Partition(Vector{Vector{Symbol}}())
    @test_throws ArgumentError Partition([Symbol[]])

    @test_throws ArgumentError RealPoles(
        [1.0, 2.0], [0.1, 0.2], [[:a]], [1:1], (; source=:test))
    I2 = Matrix{ComplexF64}(I, 2, 2)
    @test_throws ArgumentError MatrixRealPoles(
        [1.0, 2.0], [I2, I2], [[:a, :b]], [1:1], (; source=:test))
end

@testset "real-axis matrix midpoint discretization" begin
    P = Partition([[:a, :b]])
    u = ComplexF64[1.0 + 0.2im, 0.4 - 0.1im]
    v = ComplexF64[0.2 - 0.3im, 0.8 + 0.1im]
    spectral(ω) = (0.10 + 0.02ω) .* (u * u') +
                  (0.03 + 0.01ω) .* (v * v')
    bath = fit_bath(spectral, P; domain=:real_axis, method=:equal_spacing,
                    nmodes=2, ωmin=0.5, ωmax=1.5)
    @test bath isa MatrixRealPoles
    @test bath.poles ≈ [0.75, 1.25]
    @test bath.block_ranges == [1:2]
    @test bath.residues[1] ≈ 0.5 .* spectral(0.75) atol=1e-12
    @test bath.residues[2] ≈ 0.5 .* spectral(1.25) atol=1e-12
    @test all(ishermitian, bath.residues)
    @test all(R -> eigmin(Hermitian(R)) >= -1e-12, bath.residues)

    modes = factorize_residues(bath; rtol=1e-12)
    @test modes.energies ≈ [0.75, 0.75, 1.25, 1.25]
    for k in eachindex(bath.residues)
        selected = findall(==(k), modes.pole_indices)
        recovered = sum((modes.factors[j] * modes.factors[j]' for j in selected);
                        init=zeros(ComplexF64, 2, 2))
        @test recovered ≈ bath.residues[k] atol=1e-12
    end

    P2 = Partition([[:a, :b], [:c, :d]])
    spectral2(ω) = (0.5 + 0.1ω) .* spectral(ω)
    grouped = fit_bath([spectral, spectral2], P2; domain=:real_axis,
                       nmodes=1, ωmin=0.5, ωmax=1.5)
    @test grouped isa MatrixRealPoles
    @test grouped.block_ranges == [1:1, 2:2]
    @test grouped.residues[1] ≈ spectral(1.0) atol=1e-12
    @test grouped.residues[2] ≈ spectral2(1.0) atol=1e-12
    @test_throws ArgumentError fit_bath(spectral, P2; domain=:real_axis,
                                        nmodes=1, ωmin=0.5, ωmax=1.5)

    nonpsd(ω) = ComplexF64[1 0.2im; -0.2im -0.1]
    @test_throws ArgumentError fit_bath(nonpsd, P; domain=:real_axis,
                                        nmodes=1, ωmin=0.5, ωmax=1.5)
    nonhermitian(ω) = ComplexF64[1 0.2; 0 1]
    @test_throws ArgumentError fit_bath(nonhermitian, P; domain=:real_axis,
                                        nmodes=1, ωmin=0.5, ωmax=1.5)
end

@testset "matrix PSD bath fitting and realization" begin
    P = Partition([[:a, :b]])
    νs = [0.0, 0.3, 0.7, 1.2, 2.0, 4.0, 8.0]
    poles = [0.75, 1.25]
    u = ComplexF64[0.2 + 0.1im, 0.3 - 0.2im]
    R1 = u * u' # rank one
    R2 = ComplexF64[0.08 0.01im; -0.01im 0.05] # rank two, PSD
    residues = [R1, R2]
    values = [sum(2 * poles[k] / (ν^2 + poles[k]^2) .* residues[k]
                  for k in eachindex(poles)) for ν in νs]

    bath = fit_bath((; frequencies=im .* νs, values), P;
                    domain=:matsubara, nmodes=2, ωmin=0.5, ωmax=1.5,
                    solver=:sdp)
    @test bath isa MatrixRealPoles
    @test bath.diagnostics.solver == :psd
    @test bath.poles ≈ poles
    @test bath.residues[1] ≈ R1 atol=1e-12
    @test bath.residues[2] ≈ R2 atol=1e-12
    @test all(ishermitian, bath.residues)
    @test all(R -> eigmin(Hermitian(R)) >= -1e-12, bath.residues)

    reconstructed = matsubara_reconstruct(bath, νs)
    @test size(reconstructed) == (length(νs), 2, 2)
    @test all(n -> isapprox(reconstructed[n, :, :], values[n]; atol=1e-11),
              eachindex(νs))

    modes = factorize_residues(bath; rtol=1e-10)
    @test modes.energies ≈ [0.75, 1.25, 1.25]
    @test length(modes.factors) == 3
    @test modes.block_indices == [1, 1, 1]
    for k in eachindex(bath.residues)
        selected = findall(==(k), modes.pole_indices)
        recovered = sum((modes.factors[j] * modes.factors[j]' for j in selected);
                        init=zeros(ComplexF64, 2, 2))
        @test recovered ≈ bath.residues[k] atol=1e-12
    end
end

@testset "diagonal NNLS and matrix guards" begin
    P = Partition([[:a, :b]])
    νs = [0.0, 0.4, 1.0, 2.0, 4.0, 8.0]
    poles = [0.75, 1.25]
    residues = [ComplexF64[0.04 0; 0 0.02],
                ComplexF64[0.01 0; 0 0.03]]
    values = [sum(2 * poles[k] / (ν^2 + poles[k]^2) .* residues[k]
                  for k in eachindex(poles)) for ν in νs]
    bath = fit_bath((; frequencies=νs, values), P;
                    domain=:matsubara, nmodes=2, ωmin=0.5, ωmax=1.5,
                    solver=:nnls)
    @test bath isa MatrixRealPoles
    @test bath.residues ≈ residues atol=1e-12
    @test bath.diagnostics.block_diagnostics[1].relative_residual < 1e-12

    offdiag = deepcopy(values)
    offdiag[1][1, 2] = 0.01im
    offdiag[1][2, 1] = -0.01im
    @test_throws ArgumentError fit_bath((; frequencies=νs, values=offdiag), P;
                                        domain=:matsubara, nmodes=2,
                                        ωmin=0.5, ωmax=1.5, solver=:nnls)
    nonhermitian = deepcopy(values)
    nonhermitian[1][1, 2] = 0.01
    @test_throws ArgumentError fit_bath((; frequencies=νs, values=nonhermitian), P;
                                        domain=:matsubara, nmodes=2,
                                        ωmin=0.5, ωmax=1.5, solver=:psd)
    @test_throws ArgumentError MatrixRealPoles(
        [1.0], [ComplexF64[1 0; 0 -0.1]], P.blocks, [1:1], (; source=:test))
    @test_throws ArgumentError fit_bath((; frequencies=νs, values), P;
                                        domain=:matsubara, nmodes=2,
                                        ωmin=0.5, ωmax=1.5, solver=:clip)
end

@testset "active-set NNLS beats least-squares clipping" begin
    P = Partition([[:imp]])
    νs = [0.0, 0.25, 0.6, 1.1, 2.0, 4.0, 8.0]
    source_poles = [1.2818206933030747, 1.3802101361800228]
    source_residues = [0.8908786980927811, 0.19090669902576285]
    values = [sum(2 * source_poles[k] * source_residues[k] /
                  (ν^2 + source_poles[k]^2) for k in eachindex(source_poles))
              for ν in νs]
    bath = fit_bath((; frequencies=νs, values), P;
                    domain=:matsubara, nmodes=3, ωmin=0.5, ωmax=2.0,
                    solver=:nnls)
    A = [2 * ω / (ν^2 + ω^2) for ν in νs, ω in bath.poles]
    clipped = max.(A \ values, 0.0)
    nnls_residual = norm(A * bath.residues - values)
    clipped_residual = norm(A * clipped - values)
    @test all(>=(0), bath.residues)
    @test nnls_residual < 0.1 * clipped_residual
    @test bath.diagnostics.block_diagnostics[1].solver_iterations > 0
end

@testset "adaptive poles improve off-grid fit at fixed mode count" begin
    P = Partition([[:imp]])
    νs = [0.0, 0.2, 0.5, 0.9, 1.5, 2.5, 4.0, 7.0, 12.0]
    source_poles = [0.68, 1.43]
    source_residues = [0.09, 0.035]
    values = [sum(2 * source_poles[k] * source_residues[k] /
                  (ν^2 + source_poles[k]^2) for k in eachindex(source_poles))
              for ν in νs]
    equal = fit_bath((; frequencies=νs, values), P;
                     domain=:matsubara, nmodes=2, ωmin=0.5, ωmax=2.0)
    adaptive = fit_bath((; frequencies=νs, values), P;
                        domain=:matsubara, nmodes=2, ωmin=0.5, ωmax=2.0,
                        method=:adaptive)
    equal_error = norm(matsubara_reconstruct(equal, νs) - values)
    adaptive_error = norm(matsubara_reconstruct(adaptive, νs) - values)
    @test adaptive_error < 0.1 * equal_error
    @test norm(adaptive.poles - source_poles) < norm(equal.poles - source_poles)
    @test adaptive.diagnostics.method == :adaptive
    @test adaptive.diagnostics.block_diagnostics[1].pole_selection ==
          :bounded_coordinate_refinement
end

@testset "adaptive matrix PSD smoke" begin
    P = Partition([[:a, :b]])
    νs = [0.0, 0.25, 0.6, 1.0, 1.8, 3.0, 5.0, 9.0]
    source_poles = [0.68, 1.43]
    u = ComplexF64[0.25 + 0.1im, 0.30 - 0.15im]
    R1 = u * u'
    R2 = ComplexF64[0.06 0.012im; -0.012im 0.04]
    values = [sum(2 * source_poles[k] / (ν^2 + source_poles[k]^2) .* R
                  for (k, R) in enumerate((R1, R2))) for ν in νs]
    equal = fit_bath((; frequencies=νs, values), P;
                     domain=:matsubara, nmodes=2, ωmin=0.5, ωmax=2.0,
                     solver=:psd)
    adaptive = fit_bath((; frequencies=νs, values), P;
                        domain=:matsubara, nmodes=2, ωmin=0.5, ωmax=2.0,
                        method=:adaptive, solver=:psd)
    equal_loss = equal.diagnostics.block_diagnostics[1].relative_residual
    adaptive_loss = adaptive.diagnostics.block_diagnostics[1].relative_residual
    @test adaptive_loss <= equal_loss + 1e-10
    @test all(ishermitian, adaptive.residues)
    @test all(R -> eigmin(Hermitian(R)) >= -1e-12, adaptive.residues)
end
