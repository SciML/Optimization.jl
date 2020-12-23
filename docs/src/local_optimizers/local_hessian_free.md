# Local Hessian-Free Second Order Optimization

Hessian-free methods are methods which perform second order optimization
by direct computation of Hessian-vector products (`Hv`) without requiring
the construction of the full Hessian. As such, these methods can perform
well for large second order optimization problems, but can require
special case when considering conditioning of the Hessian.

## Recommended Methods

`KrylovTrustRegion`

## Optim.jl

`KrylovTrustRegion`
