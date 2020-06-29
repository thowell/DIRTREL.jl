mutable struct Pendulum{T}
    m::T  # mass
    b::T  # friction
    lc::T # length to center of mass
    I::T  # inertia
    g::T  # gravity
end

# function dynamics(model::Pendulum,x,u,w)
#     @SVector [x[2],
#               (u[1] - model.m*model.g*model.lc*sin(x[1]) - model.b*x[2])/model.I]
# end

function dynamics(model::Pendulum,x,u,w)
    @SVector [x[2],
              u[1]/((model.m + w[1])*model.lc*model.lc) - model.g*sin(x[1])/model.lc - model.b*x[2]/((model.m + w[1])*model.lc*model.lc)]
end

n,m = 2,1
model = Pendulum(1.0,0.1,0.5,0.25,9.81)
