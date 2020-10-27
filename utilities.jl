function quad_over_lin(m, t, x, y)
    @constraint(m, [y+t, 2x, y-t] ∈ SecondOrderCone())
end

function diag_unit(n, i)
    a = spzeros(n, n)
    a[i,i] = 1.0
    return a
end

function sparse_mat_inner_prod(A, B)
    return sum(v*B[i,j] for (i,j,v) in zip(findnz(A)...))
end