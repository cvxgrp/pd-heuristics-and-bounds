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
n = 1001

tmin = 1
tmax = 1.5

tavg = (tmax + tmin)/2
trad = (tmax - tmin)/2

ω = 3*(2π)

A = (n * spdiagm(-1 => ones(n-1), 0 => -2ones(n), 1 => ones(n-1))/ω^2 + tavg*I/n)/trad
b = zeros(n)
b[n÷2+1] = 2/(trad*n)
# b[n÷2+1] = 1/(trad)
# b[1] = 1/(trad*n)

σ = .5

x = range(-1, 1, length=n)

z_hat = cos.(ω*x) .* exp.(-x.^2/σ^2)
z_hat[n÷2 + 1:end] .= 0

@timeit to "Optimization and Lower bounds with n = $(n)" begin
# @time greedy_round_robin(A, b, ones(n), z_hat)
@timeit to "Genetic algorithm" ga_obj = genetic_algorithm(A, b, ones(n), z_hat)
@timeit to "IPOPT" ipopt_obj = ipopt_optimize(A, b, ones(n), z_hat)
@timeit to "SFD" sfd_obj = sign_flip_descent(A, b, ones(n), z_hat)
@timeit to "Power lower bound" plb_obj = sparse_power_lower_bound(A, b, ones(n), z_hat, to)
@timeit to "Diagonal design lower bound" dlb_obj = diagonal_design_lower_bound(A, b, ones(n), z_hat)
# @time global_solver(A, b, ones(n), z_hat)  # Note that this can take a very long time!
end


ns = 10^-9

names = ["Genetic algorithm", "IPOPT", "SFD", "Power lower bound", "Diagonal design lower bound"]
times = [TimerOutputs.time(to["Optimization and Lower bounds with n = $(n)"][name])*ns for name in names]
values = [ga_obj, ipopt_obj, sfd_obj, plb_obj, dlb_obj]


plot()
for (name, time, value) in zip(names, times, values)
    plot!([time], [value], seriestype=:scatter, label=name, xaxis=:log10, yaxis=:log10)
end

xlabel!("Time (s)")
ylabel!("Objective value")

savefig("output.pdf")
closeall()