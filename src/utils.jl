function matrix_sqrt(A)
    e = eigen(A)

    for (i,ee) in enumerate(e.values)
        if ee < 0.0
            e.values[i] = 1.0e-3
        end
    end
    # return e.vectors*Diagonal(sqrt.(e.values))*inv(e.vectors)
    return Diagonal(sqrt.(e.values))*e.vectors'
end

function fastsqrt(A)
    #FASTSQRT computes the square root of a matrix A with Denman-Beavers iteration

    #S = sqrtm(A);
    Ep = 1e-8*Matrix(I,size(A))

    # if count(diag(A) .> 0.0) != size(A,1)
    #     S = diagm(sqrt.(diag(A)));
    #     return S
    # end

    In = Matrix(1.0*I,size(A));
    S = A;
    T = Matrix(1.0*I,size(A));

    T = .5*(T + inv(S+Ep));
    S = .5*(S+In);
    for k = 1:4
        Snew = .5*(S + inv(T+Ep));
        T = .5*(T + inv(S+Ep));
        S = Snew;
    end
    return S
end

# linear interpolation
function linear_interp(x0,xf,T)
    n = length(x0)
    X = [copy(Array(x0)) for t = 1:T]
    for t = 1:T
        for i = 1:n
            X[t][i] = (xf[i]-x0[i])/(T-1)*(t-1) + x0[i]
        end
    end

    return X
end

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

function row_col_cartesian!(row,col,r,c)
    for i = 1:length(r)
        push!(row,convert(Int,r[i]))
        push!(col,convert(Int,c[i]))
    end
    return row, col
end
