using JuMP, Ipopt, MosekTools, Dualization
using LinearAlgebra, SparseArrays

# Direct dual bound from formulation in Kuang and Miller 2020
function power_dual_lower_bound(A, b, w, z_hat)
    P = 2sparse(Diagonal(w.^2))
    q = -2 * (w.^2 .* z_hat)
    r = norm(w .* z_hat)^2

    # Renormalize to Θ = {0, 1}
    A_inv = inv(Matrix((A - I)/2)) # Only for illustrative purposes. Please don't actually do this in practice.
    b_prime = A_inv * (b/2)


    P_bar = A_inv' * P * A_inv
    q_bar = -A_inv' * (P * b_prime + q)
    r_bar = (b_prime' * P * b_prime)/2 + q' * b_prime + r

    n = size(A, 1)

    model = Model(with_dual_optimizer(Mosek.Optimizer))

    @variable(model, t)
    @variable(model, λ[1:n] >= 0)

    @show r_bar

    @objective(model, Max, r_bar - t/2)
    
    D = sparse(Diagonal(λ))
    v = q_bar - D * b_prime/2
    T = P_bar + D + (D*A_inv + A_inv'*D)/2

    @constraint(model,
        [t v'
         v T] in PSDCone()
    )

    optimize!(model)

    @show objective_value(model)
    return r_bar, JuMP.value.(T), JuMP.value.(v), JuMP.value(t), A_inv, λ
end

# Dual of power bound, formulation in Angeris, Vučković, Boyd 2020
function sparse_power_dual_lower_bound(A, b, w, z_hat, to)
    P = 2sparse(Diagonal(w.^2))
    q = -2 * (w.^2 .* z_hat)
    r = norm(w .* z_hat)^2

    n = size(A, 1)

    model = Model(dual_optimizer(Mosek.Optimizer))

    @timeit to "Model building" begin

    @variable(model, t)
    @variable(model, λ[1:n] >= 0)
    
    D = sparse(Diagonal(λ))

    v = q - A' * D * b
    T = P + A' * D * A - D
    u = b' * D * b + r

    @objective(model, Max, u - t/2)

    @constraint(model,
        [t v'
         v T] in PSDCone()
    )

    end

    optimize!(model)

    @show objective_value(model)
end

# Primal formulation of dual bound; this is faster with Mosek
# since it does not support Chordal sparsity
function sparse_power_lower_bound(A, b, w, z_hat, to)
    P = 2sparse(Diagonal(w.^2))
    q = -2 * (w.^2 .* z_hat)
    r = norm(w .* z_hat)^2

    n = size(A, 1)

    model = Model(Mosek.Optimizer)

    @timeit to "Model building" begin

    @variable(model, S[1:n+1, 1:n+1] in PSDCone())

    Z = S[2:end, 2:end]
    z = S[2:end, 1]

    @constraint(model, S[1, 1] == 1)

    @objective(model, Min, sparse_mat_inner_prod(P, Z)/2 + q'*z + r)
    
    for i=1:n
        D_i = diag_unit(n, i)
        @constraint(model, sparse_mat_inner_prod(A'*D_i*A - D_i, Z) <= 2*z'*A'*D_i*b - b'* D_i * b)
    end

    end

    @timeit to "Model optimization" optimize!(model)

    @show objective_value(model)
end

# Diagonal design lower bound from Angeris, Vučković, and Boyd 2019
function diagonal_design_lower_bound(A, b, w, z_hat)
    @info "Diagonal design lower bound"
    n = size(A, 1)

    model = Model(Mosek.Optimizer)

    @variable(model, ν[1:n])
    @variable(model, t[1:n])

    @objective(model, Max, -sum(t) - 2*(ν' * b) + norm(w .* z_hat)^2)
    
    Atν = A' * ν

    for i=1:n
        quad_over_lin(model, t[i], Atν[i] - ν[i] - w[i]^2*z_hat[i], w[i]^2)
        quad_over_lin(model, t[i], Atν[i] + ν[i] - w[i]^2*z_hat[i], w[i]^2)
    end

    @info "Finished building model"

    optimize!(model)

    @info primal_status(model)
    @info dual_status(model)

    @show objective_value(model)
end

function special_lower_bound(A, b, S, w, z_hat; list=[])
    # TODO
end