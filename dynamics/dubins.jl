mutable struct Dubins
    f::Function
    nx::Int
    nu::Int
    nw::Int
end

function dubins_dynamics(model::Dubins,x,u,w)
    @SVector [(u[1] + w[1])*cos(x[3]), (u[1] + w[1])*sin(x[3]), u[2]]
end

nx,nu,nw = 3,2,1
model = Dubins(dubins_dynamics,nx,nu,nw)

function circle_obs(x,y,xc,yc,r)
    (x-xc)^2 + (y-yc)^2 - r^2
end
