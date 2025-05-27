module OptimizationODE

using Reexport
@reexport using Optimization, Optimization.SciMLBase
using OrdinaryDiffEq

export ODEOptimizer, ODEGradientDescent, RKChebyshevDescent, RKAccelerated, PRKChebyshevDescent

# ODEOptimizer is a simple wrapper
struct ODEOptimizer{Solver} end

# Define solver aliases using stable methods
const ODEGradientDescent    = ODEOptimizer{Euler}()
const RKChebyshevDescent    = ODEOptimizer{ROCK2}()
const RKAccelerated         = ODEOptimizer{BS3}()
const PRKChebyshevDescent   = ODEOptimizer{ROCK4}()

function SciMLBase.supports_opt_cache_interface(::ODEOptimizer)
    return true
end

function SciMLBase.requiresgradient(::ODEOptimizer)
    return true
end

function SciMLBase.__solve(
    cache::OptimizationCache{F, RC, LB, UB, LC, UC, S, ODEOptimizer{Solver}, D, P, C}
) where {F, RC, LB, UB, LC, UC, S, Solver, D, P, C}

    prob = ODEProblem((du, u, p, t) -> begin
        cache.f.grad(du, u, cache.p)
        du .*= -1
    end, cache.u0, (0.0, cache.solver_args[:maxiters] * cache.solver_args[:dt]), cache.p)

    sol = solve(prob, Solver(); dt = cache.solver_args[:dt])

    final_u = sol.u[end]
    final_f = cache.f(final_u, cache.p)[1]
    stats = Optimization.OptimizationStats(; time = sol.t[end])

    return SciMLBase.build_solution(cache, ODEOptimizer{Solver}(), final_u, final_f;
        original = sol, retcode = ReturnCode.Success, stats)
end

end
