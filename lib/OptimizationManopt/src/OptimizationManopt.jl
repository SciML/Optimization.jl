module OptimizationManopt

using Optimization, Manopt, ManifoldsBase

"""
    abstract type AbstractManoptOptimizer end

A Manopt solver without things specified by a call to `solve` (stopping criteria) and
internal state.
"""
abstract type AbstractManoptOptimizer end

function stopping_criterion_to_kwarg(stopping_criterion::Nothing)
    return NamedTuple()
end
function stopping_criterion_to_kwarg(stopping_criterion::StoppingCriterion)
    return (; stopping_criterion = stopping_criterion)
end

## gradient descent

struct GradientDescentOptimizer{
                                Teval <: AbstractEvaluationType,
                                TM <: AbstractManifold,
                                TLS <: Linesearch
                                } <: AbstractManoptOptimizer
    M::TM
    stepsize::TLS
end

function GradientDescentOptimizer(M::AbstractManifold;
                                  eval::AbstractEvaluationType = MutatingEvaluation(),
                                  stepsize::Stepsize = ArmijoLinesearch(M))
    GradientDescentOptimizer{typeof(eval), typeof(M), typeof(stepsize)}(M, stepsize)
end

function call_manopt_optimizer(opt::GradientDescentOptimizer{Teval},
                               loss,
                               gradF,
                               x0,
                               stopping_criterion::Union{Nothing, Manopt.StoppingCriterion}) where {
                                                                                                    Teval <:
                                                                                                    AbstractEvaluationType
                                                                                                    }
    sckwarg = stopping_criterion_to_kwarg(stopping_criterion)
    opts = gradient_descent(opt.M,
                            loss,
                            gradF,
                            x0;
                            return_options = true,
                            evaluation = Teval(),
                            stepsize = opt.stepsize,
                            sckwarg...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(opt.M, minimizer), options = opts),
           :who_knows
end

## Nelder-Mead

struct NelderMeadOptimizer{
                           TM <: AbstractManifold,
                           Tpop <: AbstractVector
                           } <: AbstractManoptOptimizer
    M::TM
    initial_population::Tpop
end

function NelderMeadOptimizer(M::AbstractManifold)
    initial_population = [rand(M) for _ in 1:(manifold_dimension(M) + 1)]
    return NelderMeadOptimizer{typeof(M), typeof(initial_population)}(M, initial_population)
end

function call_manopt_optimizer(opt::NelderMeadOptimizer,
                               loss,
                               gradF,
                               x0,
                               stopping_criterion::Union{Nothing, Manopt.StoppingCriterion})
    sckwarg = stopping_criterion_to_kwarg(stopping_criterion)

    opts = NelderMead(opt.M,
                      loss,
                      opt.initial_population;
                      return_options = true,
                      sckwarg...)
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(opt.M, minimizer), options = opts),
           :who_knows
end

## augmented Lagrangian method

## conjugate gradient descent

struct ConjugateGradientDescentOptimizer{Teval <: AbstractEvaluationType,
                                         TM <: AbstractManifold, TLS <: Stepsize} <:
       AbstractManoptOptimizer
    M::TM
    stepsize::TLS
end

function ConjugateGradientDescentOptimizer(M::AbstractManifold;
                                           eval::AbstractEvaluationType = MutatingEvaluation(),
                                           stepsize::Stepsize = ArmijoLinesearch(M))
    ConjugateGradientDescentOptimizer{typeof(eval), typeof(M), typeof(stepsize)}(M,
                                                                                 stepsize)
end

function call_manopt_optimizer(opt::ConjugateGradientDescentOptimizer{Teval},
                               loss,
                               gradF,
                               x0,
                               stopping_criterion::Union{Nothing, Manopt.StoppingCriterion}) where {
                                                                                                    Teval <:
                                                                                                    AbstractEvaluationType
                                                                                                    }
    sckwarg = stopping_criterion_to_kwarg(stopping_criterion)
    opts = conjugate_gradient_descent(opt.M,
                                      loss,
                                      gradF,
                                      x0;
                                      return_options = true,
                                      evaluation = Teval(),
                                      stepsize = opt.stepsize,
                                      sckwarg...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(opt.M, minimizer), options = opts),
           :who_knows
end

## exact penalty method

## particle swarm

## quasi Newton

struct QuasiNewtonOptimizer{Teval <: AbstractEvaluationType,
                            TM <: AbstractManifold, Tretr <: AbstractRetractionMethod,
                            Tvt <: AbstractVectorTransportMethod, TLS <: Stepsize} <:
       AbstractManoptOptimizer
    M::TM
    retraction_method::Tretr
    vector_transport_method::Tvt
    stepsize::TLS
end

function QuasiNewtonOptimizer(M::AbstractManifold;
                              eval::AbstractEvaluationType = MutatingEvaluation(),
                              retraction_method::AbstractRetractionMethod = default_retraction_method(M),
                              vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(M),
                              stepsize = WolfePowellLinesearch(M;
                                                               retraction_method = retraction_method,
                                                               vector_transport_method = vector_transport_method,
                                                               linesearch_stopsize = 1e-12))
    QuasiNewtonOptimizer{typeof(eval), typeof(M), typeof(retraction_method),
                         typeof(vector_transport_method), typeof(stepsize)}(M,
                                                                            retraction_method,
                                                                            vector_transport_method,
                                                                            stepsize)
end

function call_manopt_optimizer(opt::QuasiNewtonOptimizer{Teval},
                               loss,
                               gradF,
                               x0,
                               stopping_criterion::Union{Nothing, Manopt.StoppingCriterion}) where {
                                                                                                    Teval <:
                                                                                                    AbstractEvaluationType
                                                                                                    }
    sckwarg = stopping_criterion_to_kwarg(stopping_criterion)
    opts = conjugate_gradient_descent(opt.M,
                                      loss,
                                      gradF,
                                      x0;
                                      return_options = true,
                                      evaluation = Teval(),
                                      retraction_method = opt.retraction_method,
                                      vector_transport_method = opt.vector_transport_method,
                                      stepsize = opt.stepsize,
                                      sckwarg...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(opt.M, minimizer), options = opts),
           :who_knows
end

## Optimization.jl stuff

function build_loss(f::OptimizationFunction, prob, cur)
    function (::AbstractManifold, θ)
        x = f.f(θ, prob.p, cur...)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end
end

function build_gradF(f::OptimizationFunction{true}, prob, cur)
    function (::AbstractManifold, G, θ)
        X = f.grad(G, θ, cur...)
        if prob.sense === Optimization.MaxSense
            return -X # TODO: check
        else
            return X
        end
    end
end

# TODO:
# 1) convert tolerances and other stopping criteria
# 2) return convergence information
# 3) add callbacks to Manopt.jl

function SciMLBase.__solve(prob::OptimizationProblem,
                           opt::AbstractManoptOptimizer,
                           data = Optimization.DEFAULT_DATA;
                           callback = (args...) -> (false),
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           progress = false,
                           kwargs...)
    local x, cur, state

    if data !== Optimization.DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    stopping_criterion = nothing
    if maxiters !== nothing
        stopping_criterion = StopAfterIteration(maxiters)
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)

    _loss = build_loss(f, prob, cur)

    gradF = build_gradF(f, prob, cur)

    opt_res, opt_ret = call_manopt_optimizer(opt, _loss, gradF, prob.u0, stopping_criterion)

    return SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p),
                                    opt,
                                    opt_res.minimizer,
                                    prob.sense === Optimization.MaxSense ?
                                    -opt_res.minimum : opt_res.minimum;
                                    original = opt_res.options,
                                    retcode = opt_ret)
end

end # module OptimizationManopt
