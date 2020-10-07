using UnicodePlots

include("./optimizers.jl")
include("./lowerbounds.jl")

n = 100

tavg = 5
trad = 4

ω = 2*(2π)

A = (n^2 * spdiagm(-1 => ones(n-1), 0 => -2ones(n), 1 => ones(n-1))/ω^2 + tavg*I)/trad
b = zeros(n)
b[n÷2] = -1/trad
b[n÷2+1] = -1/trad

σ = .5

x = range(-1, 1, length=n)

z_hat = sin.(fr*x) .* exp.(-x.^2/σ^2)

ipopt_optimize(A, b, ones(n), z_hat)
greedy_round_robin(A, b, ones(n), z_hat)
sign_flip_descent(A, b, ones(n), z_hat)