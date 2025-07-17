# v4 Breaking changes

 1. The main change in this breaking release has been the way mini-batching is handled. The data argument in the solve call and the implicit iteration of that in the callback has been removed,
    the stochastic solvers (Optimisers.jl and Sophia) now handle it explicitly. You would now pass in a DataLoader to OptimizationProblem as the second argument to the objective etc (p) if you
    want to do minibatching, else for full batch just pass in the full data.

 2. The support for extra returns from objective function has been removed. Now the objective should only return a scalar loss value, hence callback doesn't take extra arguments other than the state and loss value.
