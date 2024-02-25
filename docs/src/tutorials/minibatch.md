# Data Iterators and Minibatching

It is possible to solve an optimization problem with batches using a `Flux.Data.DataLoader`, which is passed to `Optimization.solve` with `ncycles`. All data for the batches need to be passed as a tuple of vectors.

!!! note
    
    This example uses the OptimizationOptimisers.jl package. See the
    [Optimisers.jl page](@ref optimisers) for details on the installation and usage.

```@example
using Flux, Optimization, OptimizationOptimisers, OrdinaryDiffEq, SciMLSensitivity

function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k * (temp - temp_m)
end

function true_sol(du, u, p, t)
    true_p = [log(2) / 8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end

ann = Chain(Dense(1, 8, tanh), Dense(8, 1, tanh))
pp, re = Flux.destructure(ann)

function dudt_(u, p, t)
    re(p)(u) .* u
end

callback = function (state, l, pred; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
        pl = scatter(t, ode_data[1, :], label = "data")
        scatter!(pl, t, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return false
end

u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 1.5f0)

t = range(tspan[1], tspan[2], length = datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat = t))

prob = ODEProblem{false}(dudt_, u0, tspan, pp)

function predict_adjoint(fullp, time_batch)
    Array(solve(prob, Tsit5(), p = fullp, saveat = time_batch))
end

function loss_adjoint(fullp, batch, time_batch)
    pred = predict_adjoint(fullp, time_batch)
    sum(abs2, batch .- pred), pred
end

k = 10
# Pass the data for the batches as separate vectors wrapped in a tuple
train_loader = Flux.Data.DataLoader((ode_data, t), batchsize = k)

numEpochs = 300
l1 = loss_adjoint(pp, train_loader.data[1], train_loader.data[2])[1]

optfun = OptimizationFunction(
    (θ, p, batch, time_batch) -> loss_adjoint(θ, batch,
        time_batch),
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, pp)
using IterTools: ncycle
res1 = Optimization.solve(optprob, Optimisers.ADAM(0.05), ncycle(train_loader, numEpochs),
    callback = callback)
```
