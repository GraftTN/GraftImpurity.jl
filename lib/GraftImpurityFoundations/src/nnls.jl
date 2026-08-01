"""
    _nnls(A, b; maxiter, tol) -> coefficients, iterations

Lawson-Hanson active-set nonnegative least squares used internally by
independent pole and Lorentzian fitting algorithms. It is intentionally kept
private: public bath fitting enters through executable kernel types in M3.
"""
function _nnls(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real};
               maxiter::Int=max(100, 30size(A, 2)),
               tol::Float64=max(size(A)...) * eps(Float64) *
                            max(opnorm(A), 1.0) * max(norm(b), 1.0))
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("NNLS right-hand side length mismatch"))
    x = zeros(Float64, n)
    passive = falses(n)
    w = Float64.(A' * (b - A * x))
    iterations = 0
    while any(i -> !passive[i] && w[i] > tol, 1:n)
        iterations += 1
        iterations <= maxiter ||
            throw(ErrorException("NNLS active-set iteration limit exceeded"))
        candidate = argmax([passive[i] ? -Inf : w[i] for i in 1:n])
        passive[candidate] = true
        while true
            z = zeros(Float64, n)
            indices = findall(passive)
            z[indices] = A[:, indices] \ b
            all(z[index] > tol for index in indices) && (x = z; break)
            ratios = [x[index] / (x[index] - z[index]) for index in indices
                      if z[index] <= tol && x[index] - z[index] > tol]
            alpha = isempty(ratios) ? 0.0 : minimum(ratios)
            x .+= alpha .* (z .- x)
            for index in indices
                if x[index] <= tol
                    x[index] = 0.0
                    passive[index] = false
                end
            end
        end
        w = Float64.(A' * (b - A * x))
    end
    x .= max.(x, 0.0)
    return x, iterations
end
