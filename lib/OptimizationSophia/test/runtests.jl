using OptimizationBase, Optimization
using OptimizationBase.SciMLBase: solve, OptimizationFunction, OptimizationProblem
using OptimizationSophia
using Lux, MLUtils, Random, ComponentArrays
using SciMLSensitivity
using Test
using Zygote
using OrdinaryDiffEqTsit5

function dudt_(u, p, t)
    ann(u, p, st)[1] .* u
end

function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k * (temp - temp_m)
end

function true_sol(du, u, p, t)
    true_p = [log(2) / 8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end

function callback(state, l) #callback function to observe training
    display(l)
    return l < 1e-2
end

function predict_adjoint(fullp, time_batch)
    Array(solve(prob, Tsit5(), p = fullp, saveat = time_batch))
end

function loss_adjoint(fullp, p)
    (batch, time_batch) = p
    pred = predict_adjoint(fullp, time_batch)
    sum(abs2, batch .- pred)
end

u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
rng = Random.default_rng()

ann = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1, tanh))
pp, st = Lux.setup(rng, ann)
pp = ComponentArray(pp)

prob = ODEProblem{false}(dudt_, u0, tspan, pp)

t = range(tspan[1], tspan[2], length = datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat = t))

k = 10
train_loader = MLUtils.DataLoader((ode_data, t), batchsize = k)

l1 = loss_adjoint(pp, (train_loader.data[1], train_loader.data[2]))[1]

optfun = OptimizationFunction(loss_adjoint,
    OptimizationBase.AutoZygote())
optprob = OptimizationProblem(optfun, pp, train_loader)

res1 = solve(optprob,
    OptimizationSophia.Sophia(), callback = callback,
    maxiters = 2000)
@test 10res1.objective < l1

# Test Sophia with ComponentArrays + Enzyme (shadow generation fix)
using ComponentArrays
x0_comp = ComponentVector(a = 0.0, b = 0.0)
rosenbrock_comp(x, p = nothing) = (1 - x.a)^2 + 100 * (x.b - x.a^2)^2

optf_sophia = OptimizationFunction(rosenbrock_comp, AutoEnzyme())
prob_sophia = OptimizationProblem(optf_sophia, x0_comp)
res_sophia = solve(prob_sophia, Optimization.Sophia(η=0.01, k=5), maxiters = 50)
@test res_sophia.objective < rosenbrock_comp(x0_comp)  # Test optimization progress
@test res_sophia.retcode == Optimization.SciMLBase.ReturnCode.Success
