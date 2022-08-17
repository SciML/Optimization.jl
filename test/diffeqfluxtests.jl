using OrdinaryDiffEq, DiffEqFlux, Optimization, OptimizationOptimJL, OptimizationOptimisers,
      ForwardDiff

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode = ODEProblem(lotka_volterra!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5())

function predict_adjoint(p)
    return Array(solve(prob_ode, Tsit5(), p = p, saveat = tsteps))
end

function loss_adjoint(p)
    prediction = predict_adjoint(p)
    loss = sum(abs2, x - 1 for x in prediction)
    return loss, prediction
end

iter = 0
callback = function (p, l, pred)
    global iter
    iter += 1

    display(l)

    # using `remake` to re-create our `prob` with current parameters `p`
    remade_solution = solve(remake(prob_ode, p = p), Tsit5(), saveat = tsteps)

    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

optprob = OptimizationFunction((x, p) -> loss_adjoint(x), Optimization.AutoForwardDiff())

prob = Optimization.OptimizationProblem(optprob, p)

result_ode = Optimization.solve(prob,
                                BFGS(initial_stepnorm = 0.0001),
                                callback = callback)

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x .^ 3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

dudt2 = Chain(x -> x .^ 3,
              Dense(2, 50, tanh),
              Dense(50, 2))

function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

iter = 0
callback = function (p, l, pred)
    global iter
    iter += 1

    display(l)
    return false
end

optprob = OptimizationFunction((p, x) -> loss_neuralode(p), Optimization.AutoForwardDiff())

prob = Optimization.OptimizationProblem(optprob, prob_neuralode.p)

result_neuralode = Optimization.solve(prob,
                                      OptimizationOptimisers.ADAM(), callback = callback,
                                      maxiters = 300)
@test result_neuralode.minimum == loss_neuralode(result_neuralode.u)[1]

prob2 = remake(prob, u0 = result_neuralode.u)
result_neuralode2 = Optimization.solve(prob2,
                                       BFGS(initial_stepnorm = 0.0001),
                                       callback = callback,
                                       maxiters = 100)
@test result_neuralode2.minimum == loss_neuralode(result_neuralode2.u)[1]
@test result_neuralode2.minimum < 10
