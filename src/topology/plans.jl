function _validated_plan_order(layout::FlavorLayout,
                               flavor_order::AbstractVector{<:Symbol})
    ordered = Tuple(Symbol.(flavor_order))
    length(ordered) == length(flavors(layout)) &&
        allunique(ordered) &&
        Set(ordered) == Set(flavors(layout)) ||
        throw(ArgumentError("topology plan flavor order must contain every layout flavor exactly once"))
    return ordered
end

"""
    T3NS(layout; flavor_order=flavors(layout))

Topology-plan value for the production minimal three-legged impurity route.
It records basis ownership and declared flavor order only. The ownership-
preserving builder and its MT3N provenance comments land in M5.
"""
struct T3NS <: AbstractImpurityTopologyPlan
    layout::FlavorLayout
    flavor_order::Tuple{Vararg{Symbol}}

    function T3NS(layout::FlavorLayout, flavor_order::Tuple{Vararg{Symbol}},
                  ::Val{:validated})
        new(layout, flavor_order)
    end
end

function T3NS(layout::FlavorLayout;
              flavor_order::AbstractVector{<:Symbol}=collect(flavors(layout)))
    order = _validated_plan_order(layout, flavor_order)
    return T3NS(layout, order, Val(:validated))
end

"""
    FTPS(layout; flavor_order=flavors(layout))

Topology-plan value for fork tensor-product states. Each ordered flavor will
own one spine node and one bath tooth in the M5 builder; this plan itself does
not construct geometry or remap bath modes.
"""
struct FTPS <: AbstractImpurityTopologyPlan
    layout::FlavorLayout
    flavor_order::Tuple{Vararg{Symbol}}

    function FTPS(layout::FlavorLayout, flavor_order::Tuple{Vararg{Symbol}},
                  ::Val{:validated})
        new(layout, flavor_order)
    end
end

function FTPS(layout::FlavorLayout;
              flavor_order::AbstractVector{<:Symbol}=collect(flavors(layout)))
    order = _validated_plan_order(layout, flavor_order)
    return FTPS(layout, order, Val(:validated))
end

Base.:(==)(left::T3NS, right::T3NS) =
    left.layout == right.layout && left.flavor_order == right.flavor_order
Base.hash(plan::T3NS, seed::UInt) =
    hash((plan.layout, plan.flavor_order), hash(:T3NS, seed))

Base.:(==)(left::FTPS, right::FTPS) =
    left.layout == right.layout && left.flavor_order == right.flavor_order
Base.hash(plan::FTPS, seed::UInt) =
    hash((plan.layout, plan.flavor_order), hash(:FTPS, seed))
