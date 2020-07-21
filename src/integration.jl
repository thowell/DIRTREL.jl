function xm_rk3_implicit(model,x⁺,x,u⁺,u,w,h)
    0.5*(x⁺ + x) + h[1]/8.0*(model.f(model,x,u,w) - model.f(model,x⁺,u⁺,w))
end

function u_midpoint(u⁺,u)
    0.5*(u⁺ + u)
end

function rk3_implicit(model,x⁺,x,u⁺,u,w,h)
    xm = xm_rk3_implicit(model,x⁺,x,u⁺,u,w,h)
    um = u_midpoint(u⁺,u)
    x⁺ - x - h[1]/6*model.f(model,x,u,w) - 4*h[1]/6*model.f(model,xm,um,w) - h[1]/6*model.f(model,x⁺,u⁺,w)
end
