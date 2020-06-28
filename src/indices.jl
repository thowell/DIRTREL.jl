mutable struct Indices
    x
    u
    h
end

function init_indices(n,m,T)
    x = []
    u = []
    h = []

    for t = 1:T-1
        push!(x,(t-1)*(n+m+1) .+ (1:n))
        push!(u,(t-1)*(n+m+1)+n .+ (1:m))
        push!(h,(t-1)*(n+m+1)+n+m + 1)
    end
    push!(x,(T-1)*(n+m+1) .+ (1:n))

    return Indices(x,u,h)
end
