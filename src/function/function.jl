function instantiate_function(f, x, ::AbstractADType, p, num_cons = 0)
    grad   = f.grad   === nothing ? nothing : (G,x)->f.grad(G,x,p)
    hess   = f.hess   === nothing ? nothing : (H,x)->f.hess(H,x,p)
    hv     = f.hv     === nothing ? nothing : (H,x,v)->f.hv(H,x,v,p)
    cons   = f.cons   === nothing ? nothing : (x)->f.cons(x,p)
    cons_j = f.cons_j === nothing ? nothing : (res,x)->f.cons_j(res,x,p)
    cons_h = f.cons_h === nothing ? nothing : (res,x)->f.cons_h(res,x,p)
    return OptimizationFunction{true}(f.f, SciMLBase.NoAD(); grad=grad, hess=hess, hv=hv,
        cons=cons, cons_j=cons_j, cons_h=cons_h,
        hess_prototype=f.hess_prototype, cons_jac_prototype=f.cons_jac_prototype, cons_hess_prototype=f.cons_hess_prototype)
end
