"""
    FlavorLayout(flavors, flavor_sites, site_modes; basis=:canonical)

Immutable basis and physical-site ownership token for an impurity problem.
flavors fixes the global spin-orbital order. flavor_sites maps each flavor to
its physical site, while site_modes fixes the canonical fermion mode order
within every site. A basis rotation must create a new layout and transform all
layout-bearing one-body, bath, and interaction data with it.
"""
struct FlavorLayout
    flavors::Tuple{Vararg{Symbol}}
    flavor_sites::Tuple{Vararg{Pair{Symbol,Symbol}}}
    site_modes::Tuple{Vararg{Pair{Symbol,Tuple{Vararg{Symbol}}}}}
    basis::Symbol

    function FlavorLayout(flavors::AbstractVector{<:Symbol},
                          flavor_sites::AbstractDict{Symbol,<:Symbol},
                          site_modes::AbstractDict{Symbol,<:AbstractVector{<:Symbol}};
                          basis::Symbol=:canonical)
        ordered_flavors = Tuple(Symbol.(flavors))
        isempty(ordered_flavors) &&
            throw(ArgumentError("FlavorLayout needs at least one flavor"))
        allunique(ordered_flavors) ||
            throw(ArgumentError("FlavorLayout flavors must be unique"))
        isempty(String(basis)) &&
            throw(ArgumentError("FlavorLayout basis identity must be nonempty"))

        all(haskey(flavor_sites, flavor) for flavor in ordered_flavors) ||
            throw(ArgumentError("FlavorLayout needs one physical site per flavor"))
        Set(keys(flavor_sites)) == Set(ordered_flavors) ||
            throw(ArgumentError("FlavorLayout flavor-to-site map has unknown or missing flavors"))

        ordered_sites = Symbol[]
        flavor_site_pairs = Pair{Symbol,Symbol}[]
        for flavor in ordered_flavors
            site = Symbol(flavor_sites[flavor])
            push!(flavor_site_pairs, flavor => site)
            site in ordered_sites || push!(ordered_sites, site)
        end
        Set(keys(site_modes)) == Set(ordered_sites) ||
            throw(ArgumentError("FlavorLayout site mode map must cover exactly its physical sites"))

        site_mode_pairs = Pair{Symbol,Tuple{Vararg{Symbol}}}[]
        flattened_modes = Symbol[]
        for site in ordered_sites
            modes = Tuple(Symbol.(site_modes[site]))
            isempty(modes) &&
                throw(ArgumentError("FlavorLayout site $site must carry at least one flavor"))
            push!(site_mode_pairs, site => modes)
            append!(flattened_modes, modes)
            for flavor in modes
                flavor_sites[flavor] == site ||
                    throw(ArgumentError("FlavorLayout flavor $flavor is assigned to the wrong site mode order"))
            end
        end
        length(flattened_modes) == length(ordered_flavors) &&
            Set(flattened_modes) == Set(ordered_flavors) &&
            allunique(flattened_modes) ||
            throw(ArgumentError("FlavorLayout site mode orders must contain every flavor exactly once"))

        return new(
            ordered_flavors,
            Tuple(flavor_site_pairs),
            Tuple(site_mode_pairs),
            basis,
        )
    end
end

"""
    FlavorLayout(flavor_sites::Pair...; site_modes, basis=:canonical)

Convenience constructor whose pair order establishes the global flavor order.
site_modes remains explicit because physical-site mode order is part of the
fermionic basis identity rather than an incidental dictionary order.
"""
function FlavorLayout(flavor_sites::Pair{Symbol,Symbol}...;
                      site_modes::AbstractDict{Symbol,<:AbstractVector{<:Symbol}},
                      basis::Symbol=:canonical)
    isempty(flavor_sites) &&
        throw(ArgumentError("FlavorLayout needs at least one flavor-to-site pair"))
    flavors = Symbol[first(pair) for pair in flavor_sites]
    locations = Dict{Symbol,Symbol}(flavor_sites)
    return FlavorLayout(flavors, locations, site_modes; basis)
end

"""Ordered global spin-orbital flavor labels."""
flavors(layout::FlavorLayout) = layout.flavors

"""Canonical index of flavor in the global fermion ordering."""
function flavor_index(layout::FlavorLayout, flavor::Symbol)
    index = findfirst(==(flavor), layout.flavors)
    index === nothing && throw(KeyError(flavor))
    return index
end

"""Physical site which owns flavor in this layout."""
function physical_site(layout::FlavorLayout, flavor::Symbol)
    return layout.flavor_sites[flavor_index(layout, flavor)].second
end

"""Canonical local fermion mode order for site."""
function site_modes(layout::FlavorLayout, site::Symbol)
    for pair in layout.site_modes
        pair.first == site && return pair.second
    end
    throw(KeyError(site))
end

"""Physical sites in deterministic layout order."""
layout_sites(layout::FlavorLayout) = Tuple(pair.first for pair in layout.site_modes)

"""Stable basis identity supplied when constructing the layout."""
basis_identity(layout::FlavorLayout) = layout.basis

Base.:(==)(left::FlavorLayout, right::FlavorLayout) =
    left.flavors == right.flavors &&
    left.flavor_sites == right.flavor_sites &&
    left.site_modes == right.site_modes &&
    left.basis == right.basis

Base.hash(layout::FlavorLayout, seed::UInt) = hash(
    (layout.flavors, layout.flavor_sites, layout.site_modes, layout.basis),
    hash(:FlavorLayout, seed),
)
