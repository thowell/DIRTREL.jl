function xm_rk3_implicit(model,x⁺,x,u,w,h)
    0.5*(x⁺ + x) + h[1]/8.0*(dynamics(model,x,u,w) - dynamics(model,x⁺,u,w))
end

function rk3_implicit(model,x⁺,x,u,w,h)
    xm = xm_rk3_implicit(model,x⁺,x,u,w,h)
    x⁺ - x - h[1]/6*dynamics(model,x,u,w) - 4*h[1]/6*dynamics(model,xm,u,w) - h[1]/6*dynamics(model,x⁺,u,w)
end
