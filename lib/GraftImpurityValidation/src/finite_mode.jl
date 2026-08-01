"""
    FiniteModeAction(; beta, mu=0, static_interaction=0, n0=0,
                     bath_energies, bath_couplings,
                     boson_frequencies, boson_couplings,
                     orbital_convention=:spinless,
                     density_convention=:shifted)

Finite Anderson-Holstein action shared by Graft and dense ED references. The
fermionic and bosonic baths are kept as explicit real-frequency modes so the
Matsubara kernels can be generated without fitting.
"""
struct FiniteModeAction
    beta::Float64
    mu::Float64
    static_interaction::Float64
    n0::Float64
    bath_energies::Vector{Float64}
    bath_couplings::Vector{ComplexF64}
    boson_frequencies::Vector{Float64}
    boson_couplings::Vector{Float64}
    orbital_convention::Symbol
    density_convention::Symbol
end

function FiniteModeAction(; beta::Real,
                          mu::Real=0,
                          static_interaction::Real=0,
                          n0::Real=0,
                          bath_energies=Float64[],
                          bath_couplings=ComplexF64[],
                          boson_frequencies=Float64[],
                          boson_couplings=Float64[],
                          orbital_convention::Symbol=:spinless,
                          density_convention::Symbol=:shifted)
    β = Float64(beta)
    μ = Float64(mu)
    U = Float64(static_interaction)
    shift = Float64(n0)
    eps = Float64.(collect(bath_energies))
    hybridizations = ComplexF64.(collect(bath_couplings))
    omegas = Float64.(collect(boson_frequencies))
    couplings = Float64.(collect(boson_couplings))

    isfinite(β) && β > 0 ||
        throw(ArgumentError("beta must be finite and positive"))
    all(isfinite, (μ, U, shift)) ||
        throw(ArgumentError("mu, static_interaction, and n0 must be finite"))
    length(eps) == length(hybridizations) ||
        throw(ArgumentError("bath_energies and bath_couplings must have equal length"))
    length(omegas) == length(couplings) ||
        throw(ArgumentError("boson_frequencies and boson_couplings must have equal length"))
    all(isfinite, eps) ||
        throw(ArgumentError("bath energies must be finite"))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), hybridizations) ||
        throw(ArgumentError("bath couplings must be finite"))
    all(isfinite, omegas) && all(>(0), omegas) ||
        throw(ArgumentError("boson frequencies must be finite and positive"))
    all(isfinite, couplings) ||
        throw(ArgumentError("boson couplings must be finite"))
    orbital_convention in (:spinless, :spinful) ||
        throw(ArgumentError("orbital_convention must be :spinless or :spinful"))
    density_convention in (:shifted, :unshifted) ||
        throw(ArgumentError("density_convention must be :shifted or :unshifted"))

    return FiniteModeAction(
        β, μ, U, shift, eps, hybridizations, omegas, couplings,
        orbital_convention, density_convention)
end

"""
    finite_mode_hash(action)

Stable FNV-1a fingerprint of every action parameter and convention, used to
verify that compared benchmark cells share one action.
"""
function finite_mode_hash(action::FiniteModeAction)
    fields = String[
        bitstring(action.beta),
        bitstring(action.mu),
        bitstring(action.static_interaction),
        bitstring(action.n0),
        String(action.orbital_convention),
        String(action.density_convention),
    ]
    append!(fields, bitstring.(action.bath_energies))
    for coupling in action.bath_couplings
        push!(fields, bitstring(real(coupling)), bitstring(imag(coupling)))
    end
    append!(fields, bitstring.(action.boson_frequencies))
    append!(fields, bitstring.(action.boson_couplings))

    hash = UInt64(0xcbf29ce484222325)
    for byte in codeunits(join(fields, "|"))
        hash = (hash ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return string(hash; base=16, pad=16)
end

fermionic_frequency(action::FiniteModeAction, n::Integer) =
    (2Int(n) + 1) * pi / action.beta

bosonic_frequency(action::FiniteModeAction, n::Integer) =
    2Int(n) * pi / action.beta

"""
    hybridization_iw(action, indices=0:15)

Exact finite-mode hybridization
`Delta(i*omega_n) = sum_p |V_p|^2 / (i*omega_n - epsilon_p)`.
"""
function hybridization_iw(action::FiniteModeAction, indices=0:15)
    ns = Int.(collect(indices))
    frequencies = fermionic_frequency.(Ref(action), ns)
    values = ComplexF64[
        sum(abs2(V) / (im * frequency - epsilon)
            for (epsilon, V) in zip(
                action.bath_energies, action.bath_couplings);
            init=0.0 + 0.0im)
        for frequency in frequencies
    ]
    return MatsubaraSeries(
        ns, frequencies, values,
        (; beta=action.beta, statistics=:fermionic,
           kernel=:hybridization, action_hash=finite_mode_hash(action)))
end

"""
    retarded_interaction_iv(action, indices=0:15)

Exact finite-mode retarded interaction
`U_ret(i*nu_n) = -sum_l 2*g_l^2*omega_l/(nu_n^2+omega_l^2)`.
"""
function retarded_interaction_iv(action::FiniteModeAction, indices=0:15)
    ns = Int.(collect(indices))
    frequencies = bosonic_frequency.(Ref(action), ns)
    values = ComplexF64[
        -sum(2g^2 * omega / (frequency^2 + omega^2)
             for (omega, g) in zip(
                 action.boson_frequencies, action.boson_couplings);
             init=0.0)
        for frequency in frequencies
    ]
    return MatsubaraSeries(
        ns, frequencies, values,
        (; beta=action.beta, statistics=:bosonic,
           kernel=:retarded_interaction, action_hash=finite_mode_hash(action)))
end
