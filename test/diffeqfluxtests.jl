using OrdinaryDiffEq, DiffEqFlux, GalacticOptim, Optim, Plots, Flux

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
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
    return Array(solve(prob_ode, Tsit5(), p=p, saveat = tsteps))
end

function loss_adjoint(p)
    prediction = predict_adjoint(p)
    loss = sum(abs2, x-1 for x in prediction)
    return loss, prediction
end

list_plots = []
iter = 0
callback = function (p, l)
    global iter, list_plots

    if iter == 0
        list_plots = []
    end
    iter += 1

    display(l)

    # using `remake` to re-create our `prob` with current parameters `p`
    remade_solution = solve(remake(prob_ode, p = p), Tsit5(), saveat = tsteps)
    plt = plot(remade_solution, ylim = (0, 6))

    push!(list_plots, plt)
    display(plt)

    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

optprob = OptimizationFunction( (p,x) -> loss_adjoint(p)[1], p, GalacticOptim.AutoForwardDiff())

prob = OptimizationProblem(optprob, p)

result_ode = GalacticOptim.solve(prob, 
                                    BFGS(initial_stepnorm = 0.0001),
                                    cb = callback)


u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

dudt2 = Chain(x -> x.^3,
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

list_plots = []
iter = 0
callback = function (p, l)
  global list_plots, iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  display(l)


  return false
end

optprob = OptimizationFunction( (p,x) -> loss_neuralode(p)[1], prob_neuralode.p, GalacticOptim.AutoForwardDiff())

prob = OptimizationProblem(optprob, prob_neuralode.p)

result_neuralode = GalacticOptim.solve(prob, 
                                BFGS(), cb = callback,
                                maxiters = 300)

