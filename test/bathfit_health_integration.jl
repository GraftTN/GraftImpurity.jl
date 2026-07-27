using Test
using Graft: TreeTopology
using GraftImpurity

function _health_integration_fixture(; residue=1.0)
    layout = FlavorLayout(
        [:orbital],
        Dict(:orbital => :impurity),
        Dict(:impurity => [:orbital]);
        basis=:bathfit_health_integration,
    )
    partition = Partition(:orbital => [:orbital])
    frequencies = [-2.0, -1.0, 1.0, 2.0]
    samples = ComplexF64[
        residue / (im * frequency - 0.25) for frequency in frequencies
    ]
    input = BathFitInput(
        layout, frequencies, :orbital => samples;
        domain=:matsubara, statistics=:fermion,
    )
    plan = DiscretizationPlan(partition)
    expansion = PoleExpansion(
        BlockRealPoles(
            layout, partition, [0.25], [residue], [1];
            statistics=:fermion,
        );
        kernel=:health_integration,
        trace=(; plan, requested_order=1),
    )
    return (;
        layout, partition, input, plan, expansion,
        result=realize_bath(input, expansion, partition),
    )
end

function _health_integration_report(
    layout::FlavorLayout, training_source;
    statistics::Symbol=:fermion,
    provenance::NamedTuple=(;
        source=:integration_test,
        training_source,
        validation_source=training_source,
    ),
)
    metric = GraftImpurity.BathFitMetricSummary([1.0])
    order = GraftImpurity.BathFitOrderHealth(
        1, 1, 0, metric, metric, metric, metric, metric, metric, metric,
        nothing, nothing, 1.0, 0.0, nothing, 0.0,
        (; overall=:healthy), (;), String[],
    )
    thresholds = GraftImpurity.BathFitHealthThresholds(
        0.95, 1e-12, 1, 1e-12, 0.0, 1.0, 1.0, 1.0, (1.0,),
        (-1.0, 1.0),
    )
    return GraftImpurity.BathFitHealthReport(
        layout, statistics, (; calibrated=true, method=:synthetic), [order],
        1, Dict(1 => 1.0), (1, 1), 1.0, thresholds,
        (; overall=:healthy, prediction=:stable), String[],
        provenance,
    )
end

function _health_opsum_signature(H)
    return Tuple(
        (
            term.coeff,
            Tuple(
                (
                    operator.site, operator.name, operator.charge,
                    convert(Array, operator.op),
                )
                for operator in term.ops
            ),
        )
        for term in H.terms
    )
end

@testset "BathFitHealthReport realization and mount integration" begin
    fixture = _health_integration_fixture()
    result = fixture.result
    @test result isa DiscretizationResult
    @test result.report.health === nothing

    health = _health_integration_report(fixture.layout, fixture.input)
    attached = attach_bathfit_health(result, health)
    @test attached !== result
    @test attached.report !== result.report
    @test result.report.health === nothing
    @test attached.report.health === health
    @test attached.expansion === result.expansion
    @test attached.bath === result.bath
    @test attached.plan === result.plan
    @test attached.report.source === result.report.source
    @test attached.report.trace === result.report.trace
    @test attached.report.warnings === result.report.warnings
    @test attached.report.timing === result.report.timing

    copied_samples = [
        copy(sample) for sample in fixture.input.blocks.orbital
    ]
    copied_training_source = BathFitInput(
        fixture.layout, copy(fixture.input.frequencies),
        :orbital => copied_samples;
        domain=fixture.input.domain, statistics=fixture.input.statistics,
        metadata=(; provenance_copy=true),
    )
    copied_health = _health_integration_report(
        fixture.layout, copied_training_source,
    )
    @test attach_bathfit_health(result, copied_health).report.health ===
          copied_health

    missing_source = _health_integration_report(
        fixture.layout, fixture.input;
        provenance=(; source=:integration_test),
    )
    @test_throws ArgumentError attach_bathfit_health(result, missing_source)
    malformed_source = _health_integration_report(
        fixture.layout, :not_a_bathfit_input,
    )
    @test_throws ArgumentError attach_bathfit_health(result, malformed_source)

    wrong_samples = copy(copied_samples)
    wrong_samples[1] = copy(wrong_samples[1])
    wrong_samples[1][1, 1] += 1
    wrong_source = BathFitInput(
        fixture.layout, fixture.input.frequencies,
        :orbital => wrong_samples;
        domain=fixture.input.domain, statistics=fixture.input.statistics,
    )
    @test_throws ArgumentError attach_bathfit_health(
        result, _health_integration_report(fixture.layout, wrong_source),
    )

    wrong_layout = FlavorLayout(
        [:orbital],
        Dict(:orbital => :impurity),
        Dict(:impurity => [:orbital]);
        basis=:incompatible_health,
    )
    @test_throws ArgumentError attach_bathfit_health(
        result, _health_integration_report(wrong_layout, fixture.input),
    )
    @test_throws ArgumentError attach_bathfit_health(
        result,
        _health_integration_report(
            fixture.layout, fixture.input; statistics=:boson,
        ),
    )

    nonmountable_fixture = _health_integration_fixture(residue=-1.0)
    nonmountable = nonmountable_fixture.result
    @test nonmountable isa NonMountablePoleFit
    attached_nonmountable = attach_bathfit_health(
        nonmountable,
        _health_integration_report(
            nonmountable_fixture.layout, nonmountable_fixture.input,
        ),
    )
    @test attached_nonmountable isa NonMountablePoleFit
    @test attached_nonmountable !== nonmountable
    @test attached_nonmountable.expansion === nonmountable.expansion
    @test attached_nonmountable.plan === nonmountable.plan
    @test attached_nonmountable.report.health !== nothing
    @test nonmountable.report.health === nothing

    topology = TreeTopology(:impurity, Pair{Symbol,Symbol}[])
    raw_mounted = mount_bath(topology, result.bath)
    default_mounted = mount_bath(
        topology, attached; diagnostics=(; caller_tag=:preserved),
    )
    @test default_mounted.diagnostics.caller_tag === :preserved
    @test !hasproperty(default_mounted.diagnostics, :bathfit_health)
    @test_throws ArgumentError mount_bath(
        topology, result; carry_bathfit_health=true,
    )
    @test_throws ArgumentError mount_bath(
        topology, attached;
        diagnostics=(; bathfit_health=(; forged=true)),
    )

    carried = mount_bath(
        topology, attached; carry_bathfit_health=true,
        diagnostics=(; caller_tag=:preserved),
    )
    @test carried.diagnostics.caller_tag === :preserved
    @test carried.diagnostics.bathfit_health == (
        calibration=true,
        selected_order=1,
        selection_stability=1.0,
        selection_interval=(1, 1),
        verdicts=:healthy,
    )
    @test carried.diagnostics.bathfit_health isa NamedTuple
    @test _health_opsum_signature(carried.H) ==
          _health_opsum_signature(raw_mounted.H)
    @test GraftImpurity._opsum_integrity_hash(carried.H) ==
          GraftImpurity._opsum_integrity_hash(raw_mounted.H)
    @test carried.certificate.hamiltonian_hash ==
          raw_mounted.certificate.hamiltonian_hash
    @test carried.certificate.parametrization_hash ==
          raw_mounted.certificate.parametrization_hash
    @test carried.diagnostics.hamiltonian_hash ==
          raw_mounted.diagnostics.hamiltonian_hash
    @test carried.diagnostics.ownership_hash ==
          raw_mounted.diagnostics.ownership_hash
end
