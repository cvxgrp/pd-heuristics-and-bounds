using JuMP, Ipopt, OSQP, MosekTools
using Evolutionary
using LinearAlgebra, SparseArrays
using Random

function ipopt_optimize(A, b, w, z_hat)
    @info "IPOPT"
    model = Model(Ipopt.Optimizer)

    n = size(A, 1)

    @variable(model, z[1:n])
    @variable(model, -1 <= θ[1:n] <= 1)

    @objective(model, Min, sum((w[i]*(z[i]-z_hat[i]))^2 for i in 1:n))
    
    for i=1:n
        @NLconstraint(model, sum(v * z[j] for (j, v) in zip(findnz(A[i, :])...)) + θ[i]*z[i] == b[i])
    end

    @time optimize!(model)
    @info "Objective value (IPOPT): $(objective_value(model))"

    return JuMP.value.(θ), JuMP.value.(z)
end

function greedy_round_robin(A, b, w, z_hat; max_iter=1000)
    @info "Greedy round robin"
    n = size(A, 1)

    θ_curr = ones(n)
    z_curr = zeros(n)
    # XXX: For large design, update this to be faster; e.g., cache last result
    z_old = (A + sparse(Diagonal(θ_curr))) \ b

    f(z) = norm(w .* (z - z_hat))^2

    curr_obj, old_obj = Inf, f(z_old)
    
    for i in 1:max_iter
        idx = (i-1)%n + 1
        θ_curr[idx] *= -1
        z_curr .= (A + sparse(Diagonal(θ_curr))) \ b

        curr_obj = f(z_curr)

        if curr_obj > old_obj
            # Objective did not improve, reverse change
            θ_curr[idx] *= -1
        else
            old_obj = curr_obj
            @info "New smaller objective found: $(curr_obj)"
        end
    end

    z_curr = (A + sparse(Diagonal(θ_curr))) \ b

    @info "Objective value (Greedy): $(old_obj)"

    return θ_curr, z_curr
end

function sign_flip_descent(A, b, w, z_hat; max_iter=20, tol=1e-4)
    @info "Sign flip descent"
    n = size(A, 1)

    s_curr = sign.((A + I) \ b)
    s_curr[s_curr .== 0] .= 1
    z_curr = zeros(n)

    old_obj = Inf

    @time for curr_iter=1:max_iter
        model = Model(Mosek.Optimizer)
        MOI.set(model, MOI.Silent(), true)
        @variable(model, z[1:n])

        @objective(model, Min, sum((w[i]*(z[i]-z_hat[i]))^2 for i in 1:n))
        
        for i=1:n
            @constraint(model, A[i, :]'*z - b[i] <= s_curr[i]*z[i])
            @constraint(model, A[i, :]'*z - b[i] >= -s_curr[i]*z[i])
        end

        @time optimize!(model)

        z_curr .= JuMP.value.(z)

        s_curr[abs.(z_curr) .<= tol] .*= -1 # Flip all small signs

        @info "Objective value: $(objective_value(model))"

        if old_obj - objective_value(model) <= 1e-2
            @info "Tolerance reached at iteration $(curr_iter), exiting"
            break
        end

        old_obj = objective_value(model)

    end

    return z_curr
end

function genetic_algorithm(A, b, w, z_hat)
    function obj_fun(θ)
        if any(abs.(θ) .> 1)
            return Inf
        end
        z = (A + sparse(Diagonal(θ))) \ b
        return norm(w .* (z - z_hat))^2
    end
    

end

function global_solver(A, b, w, z_hat)
    # TODO
end