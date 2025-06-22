using Optimization, OrdinaryDiffEqTsit5, OptimizationOptimisers,
      SciMLSensitivity, Lux, Random, ComponentArrays, MLUtils
using Test

rng = Random.default_rng()

function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k * (temp - temp_m)
end

function true_sol(du, u, p, t)
    true_p = [log(2) / 8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end

function dudt_(u, p, t)
    ann(u, p, st)[1] .* u
end

function callback(state, l) #callback function to observe training
    display(l)
    return l < 1e-2
end

u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 1.5f0)

t = range(tspan[1], tspan[2], length = datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat = t))

ann = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1, tanh))
pp, st = Lux.setup(rng, ann)
pp = ComponentArray(pp)

prob = ODEProblem{false}(dudt_, u0, tspan, pp)

function predict_adjoint(fullp, time_batch)
    Array(solve(prob, Tsit5(), p = fullp, saveat = time_batch))
end

function loss_adjoint(fullp, p)
    (batch, time_batch) = p
    pred = predict_adjoint(fullp, time_batch)
    sum(abs2, batch .- pred)
end

k = 10
train_loader = MLUtils.DataLoader((ode_data, t), batchsize = k)

numEpochs = 300
l1 = loss_adjoint(pp, (train_loader.data[1], train_loader.data[2]))[1]

optfun = OptimizationFunction(loss_adjoint,
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, pp, train_loader)

res1 = Optimization.solve(optprob,
    Optimization.Sophia(), callback = callback,
    maxiters = 2000)
@test 10res1.objective < l1

optfun = OptimizationFunction(loss_adjoint,
    Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optfun, pp, train_loader)

res1 = Optimization.solve(optprob, Optimisers.Adam(0.05),
    callback = callback, maxiters = numEpochs)
@test 10res1.objective < l1

optfun = OptimizationFunction(
    (θ, p) -> loss_adjoint(θ, batch,
        time_batch),
    Optimization.AutoModelingToolkit())
optprob = OptimizationProblem(optfun, pp)
using IterTools: ncycle
@test_broken res1 = Optimization.solve(optprob, Optimisers.Adam(0.05),
    ncycle(train_loader, numEpochs),
    callback = callback, maxiters = numEpochs)
# @test 10res1.objective < l1

function loss_grad(res, fullp, p)
    (batch, time_batch) = p
    pred = solve(prob, Tsit5(), p = fullp, saveat = time_batch)
    res .= Array(adjoint_sensitivities(pred, Tsit5(); t = time_batch, p = fullp,
        dgdu_discrete = (out, u, p, t, i) -> (out .= -2 *
                                                     (batch[i] .-
                                                      u[1])),
        sensealg = InterpolatingAdjoint())[2]')
end

function callback(st, l, pred; doplot = false)
    display(l)
    if doplot
        pl = scatter(t, ode_data[1, :], label = "data")
        scatter!(pl, t, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return l < 1e-3
end

optfun = OptimizationFunction(loss_adjoint,
    grad = loss_grad)
optprob = OptimizationProblem(optfun, pp, train_loader)

res1 = Optimization.solve(optprob, Optimisers.Adam(0.05),
    callback = callback, maxiters = numEpochs)
@test 10res1.objective < l1
