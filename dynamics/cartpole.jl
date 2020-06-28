mutable struct Cartpole{T}
    mc::T # mass of the cart in kg (10)
    mp::T # mass of the pole (point mass at the end) in kg
    l::T  # length of the pole in m
    g::T  # gravity m/s^2
end

function dynamics(model::Cartpole, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[2]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]
    qdd = SVector{2}(-H\(C*view(x,1:2) + G - B*u[1]))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

n, m = 4, 1
model = Cartpole(1.0,0.2,0.5,9.81)
