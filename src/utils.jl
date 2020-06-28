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
