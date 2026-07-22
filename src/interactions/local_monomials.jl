"""
    ImpurityOperators(layout; sector=ParticleNumberSector())

Layout-owned local fermion operator collection. Its sector marker fixes one
concrete abelian physical-space representation for every impurity site; a
Hamiltonian assembly rejects mounted baths built in a different representation.
"""
struct ImpurityOperators{S<:AbstractFermionSector,L<:FlavorLayout,N<:NamedTuple}
    layout::L
    sector::S
    sites::N
end

function ImpurityOperators(layout::FlavorLayout;
                           sector::AbstractFermionSector=ParticleNumberSector())
    names = layout_sites(layout)
    carriers = Tuple(FermionSiteOperators(layout, site; sector) for site in names)
    return ImpurityOperators(layout, sector, NamedTuple{names}(carriers))
end

function site_operators(operators::ImpurityOperators, site::Symbol)
    hasproperty(operators.sites, site) || throw(KeyError(site))
    return getproperty(operators.sites, site)
end

struct _FermionFactor
    flavor::Symbol
    kind::Symbol

    function _FermionFactor(flavor::Symbol, kind::Symbol)
        kind in (:C, :Cd) || throw(ArgumentError(
            "fermion monomial factors must be :C or :Cd",
        ))
        new(flavor, kind)
    end
end

_annihilator(flavor::Symbol) = _FermionFactor(flavor, :C)
_creator(flavor::Symbol) = _FermionFactor(flavor, :Cd)

function _sort_factors!(factors::Vector{_FermionFactor}, layout::FlavorLayout;
                        reverse::Bool)
    sign = 1
    for right in 2:length(factors)
        left = right
        while left > 1
            previous = flavor_index(layout, factors[left - 1].flavor)
            current = flavor_index(layout, factors[left].flavor)
            ordered = reverse ? previous > current : previous < current
            ordered && break
            factors[left - 1], factors[left] = factors[left], factors[left - 1]
            sign = -sign
            left -= 1
        end
    end
    return sign
end

"""Canonical normal ordering with its exact fermionic sign, or `nothing` for zero."""
function _canonical_monomial(layout::FlavorLayout,
                             factors::AbstractVector{_FermionFactor})
    isempty(factors) && throw(ArgumentError("fermion monomial may not be empty"))
    all(factor -> factor.flavor in flavors(layout), factors) || throw(ArgumentError(
        "fermion monomial has a flavor absent from its FlavorLayout",
    ))
    creation = _FermionFactor[]
    annihilation = _FermionFactor[]
    sign = 1
    for (position, factor) in enumerate(factors)
        if factor.kind === :Cd
            for earlier in @view factors[1:(position - 1)]
                earlier.kind === :C || continue
                earlier.flavor == factor.flavor && throw(ArgumentError(
                    "non-normal local `C*Cd` product needs an explicit one-body term",
                ))
                sign = -sign
            end
            push!(creation, factor)
        else
            push!(annihilation, factor)
        end
    end
    for collection in (creation, annihilation)
        seen = Set{Symbol}()
        for factor in collection
            factor.flavor in seen && return nothing
            push!(seen, factor.flavor)
        end
    end
    sign *= _sort_factors!(creation, layout; reverse=false)
    sign *= _sort_factors!(annihilation, layout; reverse=true)
    canonical = Tuple((factor.flavor, factor.kind)
                      for factor in (creation..., annihilation...))
    return canonical, sign
end

function _factors_from_key(key::Tuple)
    return _FermionFactor[_FermionFactor(Symbol(factor[1]), Symbol(factor[2]))
                          for factor in key]
end

function _add_monomial!(coefficients::Dict{Tuple,ComplexF64},
                        layout::FlavorLayout, coefficient::Number,
                        factors::AbstractVector{_FermionFactor})
    iszero(coefficient) && return coefficients
    canonical = _canonical_monomial(layout, factors)
    canonical === nothing && return coefficients
    key, sign = canonical
    updated = get(coefficients, key, 0.0 + 0.0im) + ComplexF64(sign * coefficient)
    if iszero(updated)
        delete!(coefficients, key)
    else
        coefficients[key] = updated
    end
    return coefficients
end

function _adjoint_factors(factors::AbstractVector{_FermionFactor})
    return _FermionFactor[
        _FermionFactor(factor.flavor, factor.kind === :C ? :Cd : :C)
        for factor in Iterators.reverse(factors)
    ]
end

function _validate_hermitian_monomials(coefficients::Dict{Tuple,ComplexF64},
                                       layout::FlavorLayout,
                                       name::AbstractString)
    tolerance = _interaction_tolerance(values(coefficients))
    for (key, coefficient) in coefficients
        canonical = _canonical_monomial(layout, _adjoint_factors(_factors_from_key(key)))
        canonical === nothing && throw(ArgumentError(
            "$name contains an algebraically zero monomial after adjunction",
        ))
        adjoint_key, sign = canonical
        expected = sign * conj(coefficient)
        actual = get(coefficients, adjoint_key, 0.0 + 0.0im)
        isapprox(actual, expected; atol=tolerance, rtol=0) || throw(ArgumentError(
            "$name does not lower to a Hermitian fermionic operator",
        ))
    end
    return coefficients
end

function _local_tensor_matrix(operator::AbstractTensorMap)
    array = convert(Array, operator)
    ndims(array) in (2, 3) || throw(ArgumentError(
        "local fermion operators must have one physical input and at most one charge leg",
    ))
    return ndims(array) == 2 ? Matrix{ComplexF64}(array) :
        Matrix{ComplexF64}(@view array[:, :, 1])
end

function _local_product(operators::FermionSiteOperators,
                        factors::AbstractVector{_FermionFactor})
    dimension = dim(operators.P)
    matrix = Matrix{ComplexF64}(I, dimension, dimension)
    charge = 0
    for factor in factors
        operator = factor.kind === :Cd ?
            local_creator(operators, factor.flavor) :
            local_annihilator(operators, factor.flavor)
        matrix *= _local_tensor_matrix(operator)
        charge += factor.kind === :Cd ? 1 : -1
    end
    return matrix, charge
end

function _local_siteop(layout::FlavorLayout, site::Symbol,
                       operators::FermionSiteOperators,
                       factors::AbstractVector{_FermionFactor})
    matrix, charge = _local_product(operators, factors)
    encoded = join((string(lowercase(String(factor.kind)), "_",
                           flavor_index(layout, factor.flavor))
                    for factor in factors), "__")
    name = Symbol("local_", encoded)
    if operators.sector isa ParticleNumberSector
        iszero(charge) && return SiteOp(site, name, TensorMap(matrix, operators.P ← operators.P))
        parity = FermionParity(mod(charge, 2))
        S = typeof(parity ⊠ U1Irrep(charge))
        Q = Vect[S]((parity ⊠ U1Irrep(charge)) => 1)
        return SiteOp(site, name, _charged_local_tensor(matrix, operators.P, Q))
    end
    iseven(length(factors)) && return SiteOp(
        site, name, TensorMap(matrix, operators.P ← operators.P),
    )
    Q = Vect[FermionParity](FermionParity(1) => 1)
    return SiteOp(site, name, _charged_local_tensor(matrix, operators.P, Q))
end

function _monomial_siteops(layout::FlavorLayout, operators::ImpurityOperators,
                           factors::AbstractVector{_FermionFactor})
    layout == operators.layout || throw(ArgumentError(
        "fermion monomial layout must match ImpurityOperators.layout",
    ))
    groups = Dict{Symbol,Vector{_FermionFactor}}()
    for factor in factors
        site = physical_site(layout, factor.flavor)
        if !haskey(groups, site)
            groups[site] = _FermionFactor[]
        end
        push!(groups[site], factor)
    end

    site_rank = Dict(site => rank for (rank, site) in enumerate(layout_sites(layout)))
    order = sort!(collect(keys(groups)); by=site -> begin
        net_charge = sum(factor.kind === :Cd ? 1 : -1 for factor in groups[site])
        charge_class = net_charge > 0 ? 0 : net_charge < 0 ? 1 : 2
        return charge_class, site_rank[site]
    end)

    # A Graft Term is a site-labelled tensor product.  Its braid certificate
    # interprets charged factors in class-normal order: creations,
    # annihilations, then neutral local products, with the declared physical
    # site order inside each class.  Grouping a canonical CAR word into local
    # products can permute odd factors, so retain that exact fermionic sign.
    grouped = reduce(vcat, (groups[site] for site in order); init=_FermionFactor[])
    positions = Dict((factor.flavor, factor.kind) => index
                     for (index, factor) in enumerate(factors))
    permutation = [positions[(factor.flavor, factor.kind)] for factor in grouped]
    sign = 1
    for left in 1:(length(permutation) - 1), right in (left + 1):length(permutation)
        permutation[left] > permutation[right] && (sign = -sign)
    end

    siteops = SiteOp[_local_siteop(layout, site, site_operators(operators, site),
                                   groups[site])
                     for site in order]
    return siteops, sign
end

function _compile_fermion_monomials(layout::FlavorLayout,
                                    operators::ImpurityOperators,
                                    coefficients::Dict{Tuple,ComplexF64})
    H = OpSum()
    for (key, coefficient) in sort!(collect(coefficients); by=entry -> string(entry.first))
        factors = _factors_from_key(key)
        siteops, grouping_sign = _monomial_siteops(layout, operators, factors)
        H += Term(grouping_sign * coefficient, siteops...)
    end
    return H
end
