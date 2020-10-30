Random.seed!(1234)

function ipopt_optimize(A, b, w, z_hat; verbose=false)
    @info "IPOPT"
    model = Model(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), !verbose)

    n = size(A, 1)

    @variable(model, z[1:n])
    @variable(model, -1 <= θ[1:n] <= 1)

    @objective(model, Min, sum((w[i]*(z[i]-z_hat[i]))^2 for i in 1:n))

    # Note that we assume matrix is symmetric for speed!
    for i=1:n
        @NLconstraint(model, sum(v * z[j] for (j, v) in zip(findnz(A[:, i])...)) + θ[i]*z[i] == b[i])
    end

    @info "Finished building model"

    optimize!(model)
    @info "Objective value (IPOPT): $(objective_value(model))"

    return objective_value(model)
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
        end
    end

    z_curr = (A + sparse(Diagonal(θ_curr))) \ b

    @info "Objective value (Greedy): $(old_obj)"

    return old_obj
end

function sign_flip_descent(A, b, w, z_hat; max_iter=100, tol=1e-5, verbose=false)
    @info "Sign flip descent"
    n = size(A, 1)

    # Initialize to signs of desired field
    s_curr = sign.(z_hat)
    s_curr[s_curr .== 0] .= 1
    z_curr = zeros(n)

    old_obj = Inf

    for curr_iter=1:max_iter
        model = Model(Mosek.Optimizer)
        MOI.set(model, MOI.Silent(), true)
        @variable(model, z[1:n])

        @objective(model, Min, sum((w[i]*(z[i]-z_hat[i]))^2 for i in 1:n))
        
        @constraint(model, A*z - b .<= s_curr .* z)
        @constraint(model, A*z - b .>= -s_curr .* z)

        optimize!(model)

        z_curr .= JuMP.value.(z)

        s_curr[abs.(z_curr) .<= tol] .*= -1 # Flip all small signs

        if verbose
            @info "Iteration $curr_iter"
            @info "Objective value $(objective_value(model))"
        end

        if old_obj - objective_value(model) <= 1e-5
            @info "Objective value (SFD): $(objective_value(model)) at iteration $curr_iter"
            break
        end

        old_obj = objective_value(model)
    end

    return old_obj
end

function genetic_algorithm(A, b, w, z_hat; verbose=false)
    @info "Genetic algorithm"
    A_fac = ldlt(A)
    function obj_fun(θ)
        if any(abs.(θ) .> 1)
            return Inf
        end
        ldlt!(A_fac, A + sparse(Diagonal(2*θ .- 1)))

        return norm(w .* (A_fac\b - z_hat))^2
    end

    n = size(A, 1)

    init_point = rand(n)

    function flip_0_1(recombinant; prob=.05)
        for i=1:length(recombinant)
            if rand() < prob
                recombinant[i] = rand() > .5 ? 0 : 1
            end
        end
    end
    
    result = Evolutionary.optimize(
        obj_fun,
        init_point,
        GA(
            selection=rouletteinv,
            mutation=flip_0_1,
            crossover=line(),
            mutationRate=0.1,
            crossoverRate=0.5,
            ɛ=0.1,
            populationSize=250
        ),
        Evolutionary.Options(show_trace=verbose)
    )

    @info "Objective value (GA): $(Evolutionary.minimum(result))"

    return Evolutionary.minimum(result)
end

function global_solver(A, b, w, z_hat)
    @info "Mosek, Global solution"
    model = Model(Mosek.Optimizer)

    n = size(A, 1)

    @variable(model, y_plus[1:n])
    @variable(model, y_minus[1:n])
    @variable(model, s[1:n], Bin)
    @variable(model, t_plus[1:n])
    @variable(model, t_minus[1:n])

    z = y_plus + y_minus
    y = y_plus - y_minus

    @objective(model, Min, sum(
            w[i]^2*(quad_over_lin(model, t_plus[i], y_plus[i], s[i]) + quad_over_lin(model, t_minus[i], y_minus[i], 1-s[i])
            - 2*z[i]*z_hat[i] 
            + z_hat[i]^2) for i in 1:n
        )
    )
    
    @constraint(model, A*z + y .== b)

    optimize!(model)
    @info "Objective value (Global): $(objective_value(model))"

    return objective_value(model)
end