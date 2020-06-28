function midpoint(model,z,u,Δt)
    z + Δt[1]*dynamics(model,z + 0.5*Δt[1]*dynamics(model,z,u),u)
end

function rk3(model,z,u,Δt)
    k1 = k2 = k3 = zero(z)
    k1 = Δt[1]*dynamics(model,z,u)
    k2 = Δt[1]*dynamics(model,z + 0.5*k1,u)
    k3 = Δt[1]*dynamics(model,z - k1 + 2.0*k2,u)
    z + (k1 + 4.0*k2 + k3)/6.0
end
