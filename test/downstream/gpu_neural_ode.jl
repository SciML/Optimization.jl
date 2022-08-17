using DiffEqFlux, OrdinaryDiffEq, Flux, CUDA
CUDA.allowscalar(false) # Makes sure no slow operations are occuring

# Generate Data
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Make the data into a GPU-based array if the user has a GPU
ode_data = gpu(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x .^ 3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
u0 = Float32[2.0; 0.0] |> gpu
p = initial_params(dudt2) |> gpu
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
    gpu(prob_neuralode(u0, p))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end
# Callback function to observe training
list_plots = []
iter = 0
callback = function (p, l, pred; doplot = false)
    global list_plots, iter
    if iter == 0
        list_plots = []
    end
    iter += 1
    display(l)
    # plot current prediction against data
    plt = scatter(tsteps, Array(ode_data[1, :]), label = "data")
    scatter!(plt, tsteps, Array(pred[1, :]), label = "prediction")
    push!(list_plots, plt)
    if doplot
        display(plot(plt))
    end
    return false
end
result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, p,
                                          ADAM(0.05), callback = callback,
                                          maxiters = 300)
