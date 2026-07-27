using Test
using LinearAlgebra
using GraftImpurity

function _pure_health_fixture(dimension::Integer; basis::Symbol)
    flavors = [Symbol(:orbital_, index) for index in 1:dimension]
    layout = FlavorLayout(
        flavors,
        Dict(flavor => :impurity for flavor in flavors),
        Dict(:impurity => flavors);
        basis,
    )
    return (; layout, partition=Partition(:block => flavors))
end

function _pure_health_input(
    layout::FlavorLayout, frequencies::AbstractVector{<:Real}, samples;
    statistics::Symbol=:fermion,
)
    return BathFitInput(
        layout, frequencies, :block => samples;
        domain=:matsubara, statistics,
    )
end

function _pure_health_scalar_shift(input::BathFitInput, shift::Number)
    samples = ComplexF64[
        only(sample) + shift for sample in input.blocks.block
    ]
    return BathFitInput(
        input.layout, input.frequencies, :block => samples;
        domain=input.domain, statistics=input.statistics,
        metadata=input.metadata,
    )
end

function _pure_health_covariance(input::BathFitInput)
    dimension = 2sum(
        length(sample)
        for samples in values(input.blocks)
        for sample in samples
    )
    return Matrix{Float64}(I, dimension, dimension)
end

function _pure_health_expansion(
    fixture, poles::AbstractVector{<:Real}, residues;
    statistics::Symbol=:fermion,
)
    raw = BlockRealPoles(
        fixture.layout, fixture.partition, poles, residues,
        fill(1, length(poles));
        statistics,
    )
    return PoleExpansion(raw; kernel=:bathfit_health_test)
end

function _pure_health_candidate(
    order::Integer, training::BathFitInput,
    validation_prediction::BathFitInput;
    replica::Integer=1,
    start::Integer=1,
    training_prediction::BathFitInput=training,
    expansion::Union{Nothing,PoleExpansion}=nothing,
    mountable::Bool=expansion !== nothing,
)
    count = expansion === nothing ? order : length(expansion.poles)
    return BathFitHealthCandidate(
        order;
        replica,
        start,
        expansion,
        returned_poles=count,
        mode_count=count,
        mountable,
        training_target=training,
        training_prediction,
        validation_prediction,
    )
end

function _pure_health_order(report::BathFitHealthReport, order::Integer)
    return only(item for item in report.orders if item.order == order)
end
