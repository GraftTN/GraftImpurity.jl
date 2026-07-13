"""
    BathFitInput(layout, frequencies, blocks...; domain, statistics, metadata=(;))

Validated, layout-bearing input to a real-pole bath-fit kernel. `blocks` are
named scalar or square-matrix samples indexed by the common frequency grid.
The input owns no algorithm state: it records only the caller's basis token,
source samples, and source metadata needed to preserve the named block
contract through fitting and realization.
"""
struct BathFitInput{B<:NamedTuple,L<:NamedTuple,M<:NamedTuple,S}
    layout::FlavorLayout
    domain::Symbol
    statistics::Symbol
    frequencies::Vector{Float64}
    blocks::B
    target_labels::L
    metadata::M
    source_template::S

    function BathFitInput(layout::FlavorLayout, domain::Symbol,
                          statistics::Symbol, frequencies::Vector{Float64},
                          blocks::B, target_labels::L, metadata::M,
                          source_template::S,
                          ::Val{:validated}) where {B<:NamedTuple,L<:NamedTuple,
                                                     M<:NamedTuple,S}
        new{B,L,M,S}(layout, domain, statistics, frequencies, blocks,
                     target_labels, metadata, source_template)
    end
end

function _fit_sample_matrices(values, nfrequency::Int)
    length(values) == nfrequency ||
        throw(DimensionMismatch("bath-fit samples need one value per frequency"))
    samples = Matrix{ComplexF64}[]
    if all(value -> value isa Number, values)
        for value in values
            scalar = ComplexF64(value)
            isfinite(real(scalar)) && isfinite(imag(scalar)) ||
                throw(ArgumentError("bath-fit samples must be finite"))
            push!(samples, reshape(ComplexF64[scalar], 1, 1))
        end
    elseif all(value -> value isa AbstractMatrix, values)
        isempty(values) && throw(ArgumentError("bath-fit samples may not be empty"))
        dimension = size(first(values))
        length(dimension) == 2 && dimension[1] == dimension[2] ||
            throw(ArgumentError("bath-fit matrix samples must be square"))
        for (index, value) in enumerate(values)
            size(value) == dimension ||
                throw(DimensionMismatch(
                    "bath-fit matrix sample $index has an inconsistent dimension",
                ))
            matrix = Matrix{ComplexF64}(value)
            all(entry -> isfinite(real(entry)) && isfinite(imag(entry)), matrix) ||
                throw(ArgumentError("bath-fit samples must be finite"))
            push!(samples, matrix)
        end
    else
        throw(ArgumentError(
            "bath-fit samples must be all scalar values or all square matrices",
        ))
    end
    return samples
end

function _fit_sample_matrices(values::AbstractArray{<:Number,3},
                              nfrequency::Int)
    size(values, 3) == nfrequency ||
        throw(DimensionMismatch("bath-fit matrix sample axis must be last"))
    size(values, 1) == size(values, 2) ||
        throw(ArgumentError("bath-fit matrix samples must be square"))
    return _fit_sample_matrices(
        [Matrix{ComplexF64}(@view values[:, :, index])
         for index in axes(values, 3)],
        nfrequency,
    )
end

function BathFitInput(layout::FlavorLayout,
                      frequencies::AbstractVector{<:Real}, blocks::Pair...;
                      domain::Symbol,
                      statistics::Symbol,
                      metadata::NamedTuple=(;))
    domain in (:real_axis, :matsubara) ||
        throw(ArgumentError("BathFitInput domain must be :real_axis or :matsubara"))
    statistics in (:fermion, :boson) ||
        throw(ArgumentError("BathFitInput statistics must be :fermion or :boson"))
    isempty(frequencies) && throw(ArgumentError("BathFitInput needs frequencies"))
    values = Float64.(frequencies)
    all(isfinite, values) ||
        throw(ArgumentError("BathFitInput frequencies must be finite"))
    allunique(values) ||
        throw(ArgumentError("BathFitInput frequencies must be distinct"))
    isempty(blocks) && throw(ArgumentError("BathFitInput needs named blocks"))
    names = Symbol[first(block) for block in blocks]
    allunique(names) || throw(ArgumentError("BathFitInput block names must be unique"))
    samples = Tuple(_fit_sample_matrices(block.second, length(values))
                    for block in blocks)
    canonical = NamedTuple{Tuple(names)}(samples)
    labels = NamedTuple{Tuple(names)}(ntuple(_ -> nothing, length(names)))
    return BathFitInput(layout, domain, statistics, values, canonical, labels,
                        metadata, nothing, Val(:validated))
end

function _greenfunc_domain(gf::GreenFunc.Gf)
    length(gf.mesh) == 1 ||
        throw(ArgumentError("bath fitting supports one physical frequency mesh"))
    mesh = only(gf.mesh)
    if mesh isa GreenFunc.ImFreq
        gf.component === :matsubara ||
            throw(ArgumentError("an ImFreq bath-fit input needs component=:matsubara"))
        return :matsubara
    elseif mesh isa GreenFunc.ReFreq
        gf.component in (:spectral, :retarded) ||
            throw(ArgumentError(
                "a ReFreq bath-fit input needs component=:spectral or :retarded",
            ))
        return :real_axis
    end
    throw(ArgumentError("bath fitting supports ImFreq and ReFreq GreenFunc meshes"))
end

function _greenfunc_samples(gf::GreenFunc.Gf)
    nfrequency = length(only(gf.mesh))
    if gf.target_ndim == 0
        return _fit_sample_matrices(collect(gf.data), nfrequency)
    elseif gf.target_ndim == 2
        size(gf.data, 3) == nfrequency ||
            throw(DimensionMismatch("GreenFunc frequency axis must follow target axes"))
        return _fit_sample_matrices(gf.data, nfrequency)
    end
    throw(ArgumentError("bath fitting needs a scalar or matrix GreenFunc target"))
end

function _greenfunc_labels(gf::GreenFunc.Gf)
    gf.target_ndim == 0 && return nothing
    gf.target_ndim == 2 || return nothing
    return gf.target_labels
end

# GreenFunc.ImFreq exposes physical real Matsubara frequencies through indexed
# access while its iterable element declaration follows the stored integer grid.
# Indexed extraction therefore preserves data order without relying on
# `collect(mesh)`'s element-type allocation path.
_greenfunc_frequency_values(mesh) = Float64[mesh[index] for index in eachindex(mesh)]

_copy_greenfunc_template(gf::GreenFunc.Gf) = copy(gf)

function _copy_greenfunc_template(blocks::GreenFunc.BlockGf)
    return GreenFunc.BlockGf(
        (name => copy(blocks[name]) for name in keys(blocks))...,
    )
end

function _greenfunc_source_metadata(component::Symbol, temperature,
                                    metadata::NamedTuple)
    protected = (:source, :component, :temperature)
    collision = findfirst(name -> hasproperty(metadata, name), protected)
    collision === nothing || throw(ArgumentError(
        "GreenFunc source metadata key $(protected[collision]) is reserved",
    ))
    return merge(
        (; source=:greenfunc, component, temperature), metadata,
    )
end

function BathFitInput(layout::FlavorLayout, gf::GreenFunc.Gf,
                      block::Symbol; metadata::NamedTuple=(;))
    domain = _greenfunc_domain(gf)
    mesh = only(gf.mesh)
    frequencies = _greenfunc_frequency_values(mesh)
    samples = _greenfunc_samples(gf)
    statistics = gf.statistics ? :fermion : :boson
    source_metadata = _greenfunc_source_metadata(
        gf.component, gf.temperature, metadata,
    )
    input = BathFitInput(layout, frequencies, block => samples;
                          domain, statistics,
                          metadata=source_metadata)
    labels = NamedTuple{(block,)}((_greenfunc_labels(gf),))
    return BathFitInput(layout, input.domain, input.statistics,
                        input.frequencies, input.blocks, labels,
                        input.metadata, _copy_greenfunc_template(gf),
                        Val(:validated))
end

function BathFitInput(layout::FlavorLayout, blocks::GreenFunc.BlockGf;
                      metadata::NamedTuple=(;))
    first_block = first(values(blocks))
    domain = _greenfunc_domain(first_block)
    statistics = blocks.statistics ? :fermion : :boson
    frequencies = _greenfunc_frequency_values(only(blocks.mesh))
    names = Tuple(keys(blocks))
    samples = Tuple(_greenfunc_samples(blocks[name]) for name in names)
    labels = Tuple(_greenfunc_labels(blocks[name]) for name in names)
    canonical = NamedTuple{names}(samples)
    target_labels = NamedTuple{names}(labels)
    source_metadata = _greenfunc_source_metadata(
        blocks.component, blocks.temperature, metadata,
    )
    return BathFitInput(layout, domain, statistics, frequencies, canonical,
                        target_labels, source_metadata,
                        _copy_greenfunc_template(blocks), Val(:validated))
end

function _validate_fit_input(input::BathFitInput, partition::Partition)
    validate_partition(partition, input.layout)
    Tuple(keys(input.blocks)) == block_names(partition) ||
        throw(ArgumentError(
            "BathFitInput block names and order must match the named Partition",
        ))
    for name in block_names(partition)
        expected = length(block_flavors(partition, name))
        samples = getproperty(input.blocks, name)
        all(sample -> size(sample) == (expected, expected), samples) ||
            throw(DimensionMismatch(
                "BathFitInput block $name does not match its Partition dimension",
            ))
        labels = getproperty(input.target_labels, name)
        labels === nothing && continue
        labels == (block_flavors(partition, name), block_flavors(partition, name)) ||
            throw(ArgumentError(
                "BathFitInput GreenFunc target labels must match block $name flavor order",
            ))
    end
    return input
end

_fit_block_samples(input::BathFitInput, block::Symbol) =
    getproperty(input.blocks, block)
