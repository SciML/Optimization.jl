# QuadDIRECT.jl
[`QuadDIRECT`](https://github.com/timholy/QuadDIRECT.jl) is a Julia package implementing **QuadDIRECT algorithm (inspired by DIRECT and MCS)**. 

The QuadDIRECT algorithm is called using `QuadDirect()`. 

Also note that `QuadDIRECT` should (for now) be installed by doing:

`] add https://github.com/timholy/QuadDIRECT.jl.git`


## Global Optimizer
### Without Constraint Equations
The algorithm in [`QuadDIRECT`](https://github.com/timholy/QuadDIRECT.jl) is performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

Furthermore, `QuadDirect` requires `splits` which is a list of 3-vectors with initial locations at which to evaluate the function (the values must be in strictly increasing order and lie within the specified bounds) such that
`solve(problem, QuadDirect(), splits)`.

The Rosenbrock function can optimized using the `QuadDirect()` as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = GalacticOptim.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
solve(prob, QuadDirect(), splits = ([-0.9, 0, 0.9], [-0.8, 0, 0.8]))
```
    
    

