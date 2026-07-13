abstract type AbstractBathFitReport end

"""
    PoleBinDiagnostic

Numerical evidence for one raw pole residue at the common realization gate.
The original expansion is never changed to make this diagnostic pass.
"""
struct PoleBinDiagnostic
    pole_index::Int
    block::Symbol
    hermiticity_error::Float64
    minimum_eigenvalue::Float64
    tolerance::Float64
    pivots::Vector{Float64}
    reconstruction_error::Float64
    status::Symbol
end

struct PendingBathFitReport{T<:NamedTuple} <: AbstractBathFitReport
    trace::T
    diagnostics::Vector{PoleBinDiagnostic}
end

"""
    DiscretizationResult(expansion, bath, plan, report)

Successful Hamiltonian realization of a `PoleExpansion`. The M3 report slot is
typed but intentionally carries only kernel trace and residue-gate evidence;
M4 replaces it with the concrete block-preserving BathFitReport.
"""
struct DiscretizationResult{E<:PoleExpansion,B<:DiscreteBath,
                            P<:DiscretizationPlan,R<:AbstractBathFitReport}
    expansion::E
    bath::B
    plan::P
    report::R
end

"""
    NonMountablePoleFit(expansion, plan, report)

Typed result for a finite real-pole expansion that cannot be mounted as a
Hamiltonian bath. It retains all raw residues, kernel trace, and per-bin
diagnostics instead of projecting, dropping off-diagonal entries, or falling
back to a diagonal bath.
"""
struct NonMountablePoleFit{E<:PoleExpansion,P<:DiscretizationPlan,
                           R<:AbstractBathFitReport}
    expansion::E
    plan::P
    report::R
end

function _residue_tolerance(residue::AbstractMatrix, atol::Float64,
                            rtol::Float64)
    return atol + rtol * max(opnorm(residue), 1.0)
end

function _residue_tolerance(residue::Number, atol::Float64,
                            rtol::Float64)
    return atol + rtol * max(abs(residue), 1.0)
end

function _zero_pivot_tolerance(residue::AbstractMatrix, atol::Float64)
    return max(atol, eps(Float64) * max(opnorm(residue), 1.0))
end

function _zero_pivot_tolerance(residue::Number, atol::Float64)
    return max(atol, eps(Float64) * max(abs(residue), 1.0))
end

function _resolved_orbital_order(expansion::BlockRealPoles, block::Symbol,
                                 orbital_order)
    declared = block_flavors(expansion.partition, block)
    raw = if orbital_order === nothing
        declared
    elseif orbital_order isa NamedTuple
        hasproperty(orbital_order, block) || throw(KeyError(block))
        getproperty(orbital_order, block)
    elseif orbital_order isa AbstractDict
        haskey(orbital_order, block) || throw(KeyError(block))
        orbital_order[block]
    else
        throw(ArgumentError("orbital_order must be nothing, a NamedTuple, or a dictionary"))
    end
    order = Tuple(Symbol.(raw))
    length(order) == length(declared) &&
        allunique(order) && Set(order) == Set(declared) ||
        throw(ArgumentError(
            "orbital_order for block $block must be a permutation of its declared flavors",
        ))
    return order
end

function _unpivoted_semidefinite_ldl(residue::Matrix{ComplexF64},
                                     psd_tolerance::Float64,
                                     zero_tolerance::Float64)
    dimension = size(residue, 1)
    lower = Matrix{ComplexF64}(I, dimension, dimension)
    pivots = zeros(Float64, dimension)
    work = copy(residue)
    numerical_zero = false
    for column in 1:dimension
        pivot = real(work[column, column])
        if pivot < -psd_tolerance
            return (; lower, pivots, status=:non_psd,
                    reconstruction_error=Inf)
        elseif pivot < 0
            numerical_zero = true
            pivots[column] = 0.0
            remaining = column < dimension ?
                maximum(abs, @view work[(column + 1):end, column]; init=0.0) :
                0.0
            remaining <= zero_tolerance ||
                return (; lower, pivots, status=:zero_pivot_conflict,
                        reconstruction_error=Inf)
            continue
        elseif abs(pivot) <= zero_tolerance
            numerical_zero |= !iszero(pivot)
            pivots[column] = 0.0
            remaining = column < dimension ?
                maximum(abs, @view work[(column + 1):end, column]; init=0.0) :
                0.0
            remaining <= zero_tolerance ||
                return (; lower, pivots, status=:zero_pivot_conflict,
                        reconstruction_error=Inf)
            continue
        end

        pivots[column] = pivot
        for row in (column + 1):dimension
            lower[row, column] = work[row, column] / pivot
        end
        for row in (column + 1):dimension, col in (column + 1):dimension
            work[row, col] -= lower[row, column] * pivot *
                              conj(lower[col, column])
        end
        for row in (column + 1):dimension
            work[row, row] = ComplexF64(real(work[row, row]))
            for col in (row + 1):dimension
                work[row, col] = conj(work[col, row])
            end
        end
    end
    reconstructed = lower * Diagonal(pivots) * adjoint(lower)
    error = norm(reconstructed - residue)
    return (; lower, pivots,
            status=numerical_zero ? :numerical_zero : :valid,
            reconstruction_error=error)
end

function _matrix_factorization(residue::AbstractMatrix, pole_index::Int,
                               block::Symbol, declared_order::Tuple,
                               requested_order::Tuple, atol::Float64,
                               rtol::Float64)
    matrix = Matrix{ComplexF64}(residue)
    tolerance = _residue_tolerance(matrix, atol, rtol)
    zero_tolerance = _zero_pivot_tolerance(matrix, atol)
    hermiticity_error = norm(matrix - adjoint(matrix))
    symmetrized = (matrix + adjoint(matrix)) / 2
    minimum_eigenvalue = minimum(real.(eigvals(Hermitian(symmetrized))))
    if hermiticity_error > tolerance
        diagnostic = PoleBinDiagnostic(
            pole_index, block, hermiticity_error, minimum_eigenvalue, tolerance,
            Float64[], Inf, :nonhermitian,
        )
        return (; vectors=nothing, diagnostic)
    end
    if minimum_eigenvalue < -tolerance
        diagnostic = PoleBinDiagnostic(
            pole_index, block, hermiticity_error, minimum_eigenvalue, tolerance,
            Float64[], Inf, :non_psd,
        )
        return (; vectors=nothing, diagnostic)
    end

    permutation = Int[findfirst(==(flavor), declared_order)
                      for flavor in requested_order]
    permuted = symmetrized[permutation, permutation]
    decomposition = _unpivoted_semidefinite_ldl(
        permuted, tolerance, zero_tolerance,
    )
    if decomposition.status in (:non_psd, :zero_pivot_conflict)
        diagnostic = PoleBinDiagnostic(
            pole_index, block, hermiticity_error, minimum_eigenvalue, tolerance,
            decomposition.pivots, decomposition.reconstruction_error,
            decomposition.status,
        )
        return (; vectors=nothing, diagnostic)
    end

    vectors = Tuple{Vector{ComplexF64},Symbol}[]
    for column in eachindex(decomposition.pivots)
        decomposition.pivots[column] > zero_tolerance || continue
        permuted_vector = decomposition.lower[:, column] .*
                          sqrt(decomposition.pivots[column])
        vector = zeros(ComplexF64, length(declared_order))
        vector[permutation] .= permuted_vector
        push!(vectors, (vector, requested_order[column]))
    end
    reconstructed = zeros(ComplexF64, length(declared_order),
                          length(declared_order))
    for (vector, _) in vectors
        reconstructed .+= vector * vector'
    end
    reconstruction_error = norm(reconstructed - matrix)
    if reconstruction_error > tolerance
        diagnostic = PoleBinDiagnostic(
            pole_index, block, hermiticity_error, minimum_eigenvalue, tolerance,
            decomposition.pivots, reconstruction_error, :reconstruction_failure,
        )
        return (; vectors=nothing, diagnostic)
    end
    status = if hermiticity_error > 0 && decomposition.status === :numerical_zero
        :numerical_symmetrization_and_zero
    elseif hermiticity_error > 0
        :numerical_symmetrization
    else
        decomposition.status
    end
    diagnostic = PoleBinDiagnostic(
        pole_index, block, hermiticity_error, minimum_eigenvalue, tolerance,
        decomposition.pivots, reconstruction_error, status,
    )
    return (; vectors, diagnostic)
end

function _scalar_factorization(residue::Number, pole_index::Int,
                               block::Symbol, owner::Symbol, atol::Float64,
                               rtol::Float64)
    value = ComplexF64(residue)
    tolerance = _residue_tolerance(value, atol, rtol)
    zero_tolerance = _zero_pivot_tolerance(value, atol)
    hermiticity_error = abs(imag(value))
    weight = real(value)
    if hermiticity_error > tolerance
        diagnostic = PoleBinDiagnostic(
            pole_index, block, hermiticity_error, weight, tolerance,
            Float64[], Inf, :nonhermitian,
        )
        return (; vectors=nothing, diagnostic)
    elseif weight < -tolerance
        diagnostic = PoleBinDiagnostic(
            pole_index, block, hermiticity_error, weight, tolerance,
            Float64[], Inf, :non_psd,
        )
        return (; vectors=nothing, diagnostic)
    elseif weight <= zero_tolerance
        reconstruction_error = abs(value)
        reconstruction_error <= tolerance ||
            return (; vectors=nothing, diagnostic=PoleBinDiagnostic(
                pole_index, block, hermiticity_error, weight, tolerance,
                [0.0], reconstruction_error, :reconstruction_failure,
            ))
        status = if hermiticity_error > 0 && !iszero(weight)
            :numerical_symmetrization_and_zero
        elseif hermiticity_error > 0
            :numerical_symmetrization
        elseif iszero(weight)
            :valid
        else
            :numerical_zero
        end
        diagnostic = PoleBinDiagnostic(
            pole_index, block, hermiticity_error, weight, tolerance,
            [0.0], reconstruction_error, status,
        )
        return (; vectors=Tuple{Vector{ComplexF64},Symbol}[], diagnostic)
    end
    vector = ComplexF64[sqrt(weight)]
    reconstruction_error = abs(value - weight)
    status = hermiticity_error > 0 ? :numerical_symmetrization : :valid
    diagnostic = PoleBinDiagnostic(
        pole_index, block, hermiticity_error, weight, tolerance,
        [weight], reconstruction_error, status,
    )
    return (; vectors=[(vector, owner)], diagnostic)
end

function _attempt_factorization(expansion::BlockRealPoles;
                                orbital_order=nothing,
                                atol::Real=0.0,
                                rtol::Real=sqrt(eps(Float64)))
    absolute = Float64(atol)
    relative = Float64(rtol)
    isfinite(absolute) && absolute >= 0 ||
        throw(ArgumentError("factorization atol must be finite and nonnegative"))
    isfinite(relative) && relative >= 0 ||
        throw(ArgumentError("factorization rtol must be finite and nonnegative"))

    energies = Float64[]
    couplings = Vector{ComplexF64}[]
    pole_indices = Int[]
    block_indices = Int[]
    owners = Symbol[]
    diagnostics = PoleBinDiagnostic[]
    valid = true
    for pole_index in eachindex(expansion.poles)
        block_index_value = expansion.block_indices[pole_index]
        block = block_names(expansion.partition)[block_index_value]
        declared_order = block_flavors(expansion.partition, block)
        requested_order = _resolved_orbital_order(
            expansion, block, orbital_order)
        residue = expansion.residues[pole_index]
        factor = if residue isa Number
            _scalar_factorization(residue, pole_index, block,
                                  only(requested_order), absolute, relative)
        else
            _matrix_factorization(residue, pole_index, block, declared_order,
                                  requested_order, absolute, relative)
        end
        push!(diagnostics, factor.diagnostic)
        if factor.vectors === nothing
            valid = false
            continue
        end
        for (vector, owner) in factor.vectors
            push!(energies, expansion.poles[pole_index])
            push!(couplings, vector)
            push!(pole_indices, pole_index)
            push!(block_indices, block_index_value)
            push!(owners, owner)
        end
    end
    if !valid
        return (; orbitals=nothing, diagnostics)
    end
    orbitals = BathOrbitals(energies, couplings, pole_indices, block_indices,
                            owners; layout=expansion.layout,
                            partition=expansion.partition)
    return (; orbitals, diagnostics)
end

"""
    factorize_residues(expansion; orbital_order=nothing, atol=0,
                       rtol=sqrt(eps())) -> BathOrbitals

Factor a validated real-pole expansion into canonical bath modes. Matrix
residues use unpivoted, zero-pivot-safe LDL-dagger in the caller-declared
`orbital_order`; no eigensystem factorization, pivoting, or hidden arm choice
is performed. The default is the explicit within-block order stored in the
named Partition.
"""
function factorize_residues(expansion::BlockRealPoles;
                            orbital_order=nothing,
                            atol::Real=0.0,
                            rtol::Real=sqrt(eps(Float64)))
    attempted = _attempt_factorization(expansion; orbital_order, atol, rtol)
    attempted.orbitals === nothing &&
        throw(ArgumentError("BlockRealPoles is not Hermitian positive semidefinite"))
    all(diagnostic -> diagnostic.status === :valid, attempted.diagnostics) ||
        throw(ArgumentError(
            "factorize_residues requires exact PSD residues; use realize_bath to retain tolerance diagnostics",
        ))
    return attempted.orbitals
end

function _realization_plan(expansion::PoleExpansion, partition::Partition)
    trace = expansion.trace
    hasproperty(trace, :plan) || return DiscretizationPlan(partition)
    plan = getproperty(trace, :plan)
    plan isa DiscretizationPlan ||
        throw(ArgumentError("PoleExpansion trace.plan must be a DiscretizationPlan"))
    Tuple(keys(plan.blocks)) == block_names(partition) ||
        throw(ArgumentError("PoleExpansion plan must match the named Partition"))
    return plan
end

function _realization_orbital_order(expansion::PoleExpansion, orbital_order)
    orbital_order !== nothing && return orbital_order
    hasproperty(expansion.trace, :orbital_order) || return nothing
    inherited = getproperty(expansion.trace, :orbital_order)
    inherited === nothing || inherited isa NamedTuple ||
        throw(ArgumentError("PoleExpansion trace orbital_order must be a NamedTuple"))
    return inherited
end

function _canonical_realization_orbital_order(expansion::PoleExpansion,
                                              orbital_order)
    names = Tuple(block_names(expansion.poles.partition))
    orders = Tuple(_resolved_orbital_order(
        expansion.poles, block, orbital_order,
    ) for block in names)
    return NamedTuple{names}(orders)
end

"""
    realize_bath(input, expansion, partition; orbital_order=nothing,
                 atol=0, rtol=sqrt(eps()))

Run the common Hamiltonian-realizability gate for a kernel-produced real-pole
expansion. A valid expansion becomes a canonical `DiscreteBath`; a finite but
non-Hermitian or non-PSD expansion becomes `NonMountablePoleFit` with retained
raw data and per-bin evidence.
"""
function realize_bath(input::BathFitInput, expansion::PoleExpansion,
                      partition::Partition;
                      orbital_order=nothing,
                      atol::Real=0.0,
                      rtol::Real=sqrt(eps(Float64)))
    _validate_fit_input(input, partition)
    expansion.poles.layout == input.layout ||
        throw(ArgumentError("PoleExpansion FlavorLayout does not match BathFitInput"))
    expansion.poles.partition == partition ||
        throw(ArgumentError("PoleExpansion Partition does not match realization Partition"))
    expansion.poles.statistics == input.statistics ||
        throw(ArgumentError("PoleExpansion statistics do not match BathFitInput"))

    plan = _realization_plan(expansion, partition)
    requested_order = _realization_orbital_order(expansion, orbital_order)
    resolved_order = _canonical_realization_orbital_order(
        expansion, requested_order,
    )
    attempted = _attempt_factorization(
        expansion.poles; orbital_order=resolved_order, atol, rtol,
    )
    report = PendingBathFitReport(
        (; kernel=expansion.kernel, expansion_trace=expansion.trace,
           realization_orbital_order=resolved_order),
        attempted.diagnostics,
    )
    if attempted.orbitals === nothing
        return NonMountablePoleFit(expansion, plan, report)
    end
    bath = DiscreteBath(expansion.poles.layout, partition, attempted.orbitals;
                        statistics=expansion.poles.statistics)
    return DiscretizationResult(expansion, bath, plan, report)
end
