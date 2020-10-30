using JuMP, Ipopt
using Evolutionary
using LinearAlgebra, SparseArrays
using Random
using Plots
using TimerOutputs

include("./utilities.jl")
include("./optimizers.jl")
include("./lowerbounds.jl")

to = TimerOutput()

# for n in [11, 51, 101, 501, 1001]
n = 251

tmin = 1
tmax = 1.5

tavg = (tmax + tmin)/2
trad = (tmax - tmin)/2

ω = 3*(2π)
Δ = kron(spdiagm(-1 => ones(n-1), 0 => -2ones(n), 1 => ones(n-1)), I(n)) + kron(I(n), spdiagm(-1 => ones(n-1), 0 => -2ones(n), 1 => ones(n-1)))

A = (n * Δ/ω^2 + tavg*I/n)/trad
b = zeros(n^2)
b[(n+1)^2÷2] = 2/(trad*n)
# b[n÷2+1] = 1/(trad)
# b[1] = 1/(trad*n)

σ = .5

l = range(-1, 1, length=n)
x = [x for x in l for y in l]
y = [y for x in l for y in l]

z_hat = cos.(ω*x) .* cos.(ω*y) .* exp.(-(x.^2 + y.^2)/σ^2) .* (x .<= 0)


heatmap(reshape(z_hat, n, n))
savefig("z_hat_map.pdf")
closeall()

@timeit to "Optimization and Lower bounds with n = $(n^2)" begin
# @time greedy_round_robin(A, b, ones(n), z_hat)
# @timeit to "Genetic algorithm" ga_obj = genetic_algorithm(A, b, ones(n^2), z_hat, verbose=true)
@timeit to "IPOPT" ipopt_obj = ipopt_optimize(A, b, ones(n^2), z_hat, verbose=true)
@timeit to "SFD" sfd_obj = sign_flip_descent(A, b, ones(n^2), z_hat, tol=1e-6, verbose=true)
# @timeit to "Power lower bound" plb_obj = sparse_power_lower_bound(A, b, ones(n), z_hat, to)
@timeit to "Diagonal design lower bound" dlb_obj = diagonal_design_lower_bound(A, b, ones(n^2), z_hat)
# @time global_solver(A, b, ones(n), z_hat)  # Note that this can take a very long time!
end


# ns = 10^-9

# names = ["Genetic algorithm", "IPOPT", "SFD", "Power lower bound"]
# names = ["IPOPT", "SFD", "Power lower bound"]
# times = [TimerOutputs.time(to["Optimization and Lower bounds with n = $(n)"][name])*ns for name in names]
# values = [ipopt_obj, sfd_obj, plb_obj]


# plot()
# for (name, time, value) in zip(names, times, values)
#     plot!([time], [value], seriestype=:scatter, label=name, xaxis=:log10, yaxis=:log10)
# end

# xlabel!("Time (s)")
# ylabel!("Objective value")

# savefig("output.pdf")
# closeall()