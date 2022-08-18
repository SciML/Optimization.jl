using DiffEqFlux, Optimization, OrdinaryDiffEq, OptimizationOptimisers

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
    ann(u, p) .* u
end

callback = function (p, l, pred; doplot = false) #callback function to observe training
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

ann = FastChain(FastDense(1, 8, tanh), FastDense(8, 1, tanh))
pp = initial_params(ann)
prob = ODEProblem{false}(dudt_, u0, tspan, pp)

function predict_adjoint(fullp, time_batch)
    Array(solve(prob, Tsit5(), p = fullp, saveat = time_batch))
end

function loss_adjoint(fullp, batch, time_batch)
    pred = predict_adjoint(fullp, time_batch)
    sum(abs2, batch .- pred), pred
end

k = 10
train_loader = Flux.Data.DataLoader((ode_data, t), batchsize = k)

numEpochs = 300
l1 = loss_adjoint(pp, train_loader.data[1], train_loader.data[2])[1]

optfun = OptimizationFunction((θ, p, batch, time_batch) -> loss_adjoint(θ, batch,
                                                                        time_batch),
                              Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, pp)
using IterTools: ncycle
res1 = Optimization.solve(optprob, Optimisers.ADAM(0.05), ncycle(train_loader, numEpochs),
                          callback = callback, maxiters = numEpochs)
@test 10res1.minimum < l1
