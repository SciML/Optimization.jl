"""
instantiate_function(f, x, ::AbstractADType, p, num_cons = 0)::OptimizationFunction

This function is used internally by GalacticOptim.jl to construct
the necessary extra functions (gradients, Hessians, etc.) before
optimization. Each of the ADType dispatches use the supplied automatic
differentiation type in order to specify how the construction process
occurs. 

If no ADType is given, then the default `NoAD` dispatch simply
defines closures on any supplied gradient function to enclose the
parameters to match the interfaces for the specific optimization
libraries (i.e. (G,x)->f.grad(G,x,p)). If a function is not given
and the `NoAD` dispatch is used, or if the AD dispatch is currently
not capable of defining said derivative, then the constructed
`OptimizationFunction` will simply use `nothing` to specify and undefined
function.

The return of `instantiate_function` is an `OptimizationFunction` which
is then used in the optimization process. If an optimizer requires a
function that is not defined, an error is thrown.

For more information on the use of automatic differentiation, see the
documentation of the `AbstractADType` types.
"""
function instantiate_function(f, x, ::AbstractADType, p, num_cons = 0)
    grad   = f.grad   === nothing ? nothing : (G,x)->f.grad(G,x,p)
    hess   = f.hess   === nothing ? nothing : (H,x)->f.hess(H,x,p)
    hv     = f.hv     === nothing ? nothing : (H,x,v)->f.hv(H,x,v,p)
    cons   = f.cons   === nothing ? nothing : (x)->f.cons(x,p)
    cons_j = f.cons_j === nothing ? nothing : (res,x)->f.cons_j(res,x,p)
    cons_h = f.cons_h === nothing ? nothing : (res,x)->f.cons_h(res,x,p)

    return OptimizationFunction{true}(f.f, SciMLBase.NoAD(); grad=grad, hess=hess, hv=hv, 
        cons=cons, cons_j=cons_j, cons_h=cons_h,
        hess_prototype=nothing, cons_jac_prototype=nothing, cons_hess_prototype=nothing)
end
