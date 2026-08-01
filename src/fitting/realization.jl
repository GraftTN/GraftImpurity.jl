"""
    DiscretizationResult(expansion, bath, plan, report)

Successful Hamiltonian realization of a `PoleExpansion` with its concrete,
block-preserving `BathFitReport`.
"""
struct DiscretizationResult{E<:PoleExpansion,B<:DiscreteBath,
                            P<:DiscretizationPlan,R<:BathFitReport}
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
                           R<:BathFitReport}
    expansion::E
    plan::P
    report::R
end

function _validate_bathfit_health_compatibility(
    report::BathFitReport, health::BathFitHealthReport,
)
    health.layout == report.source.layout || throw(ArgumentError(
        "BathFitHealthReport layout does not match the bath-fit result",
    ))
    health.statistics === report.source.statistics || throw(ArgumentError(
        "BathFitHealthReport statistics do not match the bath-fit result",
    ))
    hasproperty(health.provenance, :training_source) || throw(ArgumentError(
        "BathFitHealthReport provenance must contain training_source::BathFitInput",
    ))
    training_source = getproperty(health.provenance, :training_source)
    training_source isa BathFitInput || throw(ArgumentError(
        "BathFitHealthReport provenance training_source must be a BathFitInput",
    ))
    _bathfit_input_target_identical(training_source, report.source) ||
        throw(ArgumentError(
            "BathFitHealthReport training source does not match the bath-fit result source",
        ))
    return nothing
end

function _bathfit_input_target_identical(
    left::BathFitInput, right::BathFitInput,
)
    left.layout == right.layout || return false
    left.domain === right.domain || return false
    left.statistics === right.statistics || return false
    left.frequencies == right.frequencies || return false
    Tuple(keys(left.blocks)) == Tuple(keys(right.blocks)) || return false
    left.target_labels == right.target_labels || return false
    for name in keys(left.blocks)
        left_samples = getproperty(left.blocks, name)
        right_samples = getproperty(right.blocks, name)
        length(left_samples) == length(right_samples) || return false
        for index in eachindex(left_samples, right_samples)
            size(left_samples[index]) == size(right_samples[index]) ||
                return false
            left_samples[index] == right_samples[index] || return false
        end
    end
    return true
end

function _bathfit_report_with_health(
    report::BathFitReport, health::BathFitHealthReport,
)
    _validate_bathfit_health_compatibility(report, health)
    return BathFitReport(
        report.source, report.reconstruction, report.blocks, report.plan,
        report.kernel, report.mountable, report.broadening, report.diagnostics,
        report.warnings, report.timing, report.trace; health,
    )
end

"""
    attach_bathfit_health(result, health)

Return a new realization result whose `BathFitReport` carries `health`.
Expansion, canonical bath (when present), and discretization plan are retained
by identity; the input result and its report are not mutated.

Attachment fails closed unless `health.provenance.training_source` is a
`BathFitInput` whose layout, domain, statistics, frequencies, ordered block
names, per-sample shapes and complex entries, and target labels exactly match
`result.report.source`. Source metadata and `source_template` object identity
are deliberately excluded: copied provenance with the same numerical target
and label contract is accepted.
"""
function attach_bathfit_health(
    result::DiscretizationResult, health::BathFitHealthReport,
)
    report = _bathfit_report_with_health(result.report, health)
    return DiscretizationResult(
        result.expansion, result.bath, result.plan, report,
    )
end

function attach_bathfit_health(
    result::NonMountablePoleFit, health::BathFitHealthReport,
)
    report = _bathfit_report_with_health(result.report, health)
    return NonMountablePoleFit(result.expansion, result.plan, report)
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
                 atol=0, rtol=sqrt(eps()), broadening=nothing)

Run the common Hamiltonian-realizability gate for a kernel-produced real-pole
expansion. A valid expansion becomes a canonical `DiscreteBath`; a finite but
non-Hermitian or non-PSD expansion becomes `NonMountablePoleFit` with retained
raw data and per-bin evidence.
"""
function realize_bath(input::BathFitInput, expansion::PoleExpansion,
                      partition::Partition;
                      orbital_order=nothing,
                      atol::Real=0.0,
                      rtol::Real=sqrt(eps(Float64)),
                      broadening=nothing)
    _validate_fit_input(input, partition)
    expansion.poles.layout == input.layout ||
        throw(ArgumentError("PoleExpansion FlavorLayout does not match BathFitInput"))
    expansion.poles.partition == partition ||
        throw(ArgumentError("PoleExpansion Partition does not match realization Partition"))
    expansion.poles.statistics == input.statistics ||
        throw(ArgumentError("PoleExpansion statistics do not match BathFitInput"))
    _validate_realization_broadening(input, expansion, broadening)

    plan = _realization_plan(expansion, partition)
    requested_order = _realization_orbital_order(expansion, orbital_order)
    resolved_order = _canonical_realization_orbital_order(
        expansion, requested_order,
    )
    started = time_ns()
    attempted = _attempt_factorization(
        expansion.poles; orbital_order=resolved_order, atol, rtol,
    )
    realization_seconds = (time_ns() - started) / 1e9
    if attempted.orbitals === nothing
        report = _bathfit_report(
            expansion, input, plan, nothing, attempted.diagnostics,
            resolved_order, realization_seconds; broadening,
        )
        return NonMountablePoleFit(expansion, plan, report)
    end
    bath = DiscreteBath(expansion.poles.layout, partition, attempted.orbitals;
                        statistics=expansion.poles.statistics)
    report = _bathfit_report(
        expansion, input, plan, bath, attempted.diagnostics,
        resolved_order, realization_seconds; broadening,
    )
    return DiscretizationResult(expansion, bath, plan, report)
end

function _validate_discretization_mount_ownership(
    result::DiscretizationResult,
)
    report = result.report
    bath = result.bath
    report.mountable || throw(ArgumentError(
        "mount_bath requires a mountable BathFitReport",
    ))
    report.plan === result.plan || throw(ArgumentError(
        "DiscretizationResult report does not own its discretization plan",
    ))
    report.kernel === result.expansion.kernel || throw(ArgumentError(
        "DiscretizationResult report kernel does not match its pole expansion",
    ))
    report.source.layout == bath_layout(bath) || throw(ArgumentError(
        "DiscretizationResult report layout does not match its canonical bath",
    ))
    report.source.statistics === bath_statistics(bath) || throw(ArgumentError(
        "DiscretizationResult report statistics do not match its canonical bath",
    ))
    result.expansion.poles.layout == bath_layout(bath) || throw(ArgumentError(
        "DiscretizationResult pole-expansion layout does not match its canonical bath",
    ))
    result.expansion.poles.partition == bath_partition(bath) || throw(ArgumentError(
        "DiscretizationResult pole expansion does not match its canonical bath partition",
    ))
    result.expansion.poles.statistics === bath_statistics(bath) ||
        throw(ArgumentError(
            "DiscretizationResult pole-expansion statistics do not match its canonical bath",
        ))
    return nothing
end

"""
    mount_bath(topology, result::DiscretizationResult;
               carry_bathfit_health=false, kwargs...)

Mount the canonical bath owned by a realization result. By default no health
metadata is carried. With `carry_bathfit_health=true`, the result must have an
attached health report and only its compact immutable summary is added to the
mounted diagnostics. Mounting always uses the existing canonical-bath path and
does not refit, gate, or alter the symbolic Hamiltonian.
"""
function mount_bath(
    topology::TreeTopology, result::DiscretizationResult;
    carry_bathfit_health::Bool=false,
    site_labels=nothing,
    sector::AbstractFermionSector=ParticleNumberSector(),
    diagnostics::NamedTuple=(;),
)
    :bathfit_health in keys(diagnostics) && throw(ArgumentError(
        "bathfit_health diagnostics are owned by DiscretizationResult.report",
    ))
    _validate_discretization_mount_ownership(result)
    mounted_diagnostics = if carry_bathfit_health
        health = result.report.health
        health === nothing && throw(ArgumentError(
            "carry_bathfit_health=true requires an attached BathFitHealthReport",
        ))
        merge(diagnostics, (; bathfit_health=_bathfit_health_mount_summary(health)))
    else
        diagnostics
    end
    return mount_bath(
        topology, result.bath; site_labels, sector,
        diagnostics=mounted_diagnostics,
    )
end
