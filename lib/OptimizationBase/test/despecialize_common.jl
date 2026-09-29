# Shared by despecialize_tests.jl (Core) and AD/despecialize_enzyme.jl (AD): a problem with
# a struct `p` and constraints, reference derivatives, and a check of every derivative
# `instantiate_function` builds, each instantiated on its own.
using ForwardDiff, LinearAlgebra, OptimizationBase, SciMLBase, Test
using SciMLBase: AutoDespecialize, AutoSpecialize, DespecializedParameters, FullSpecialize

const DespecializedFunction = OptimizationBase.DespecializedFunction

struct RosenbrockParams
    a::Float64
    b::Float64
end
struct OtherParams
    a::Float64
    b::Float64
    offset::Int
end

rosenbrock(x, p) = (p.a - x[1])^2 + p.b * (x[2] - x[1]^2)^2
shifted_rosenbrock(x, p) = rosenbrock(x, p) + p.offset
function rosenbrock_cons!(res, x, p)
    res[1] = x[1]^2 + x[2]^2
    res[2] = p.a * x[1] * x[2]
    return nothing
end
function other_cons!(res, x, p)
    res[1] = p.b * x[1]^2
    res[2] = x[2] + p.offset
    return nothing
end

const X0 = [0.3, 0.7]
const V = [0.2, -0.5]
const Σ = 0.7
const Μ = [1.3, -0.4]
const P0 = RosenbrockParams(1.0, 100.0)

function reference_derivatives(p = P0; x = X0, v = V, σ = Σ, μ = Μ)
    cons(y) = (res = zeros(eltype(y), 2); rosenbrock_cons!(res, y, p); res)
    g = ForwardDiff.gradient(y -> rosenbrock(y, p), x)
    H = ForwardDiff.hessian(y -> rosenbrock(y, p), x)
    J = ForwardDiff.jacobian(cons, x)
    Hc = [ForwardDiff.hessian(y -> cons(y)[i], x) for i in 1:2]
    return (;
        grad = g, fg = (rosenbrock(x, p), g), hess = H, hv = H * v, cons_j = J,
        cons_jvp = J * v, cons_vjp = J' * v, cons_h = Hc,
        lag_h = σ * H + sum(μ[i] * Hc[i] for i in 1:2),
    )
end

# Each derivative is instantiated on its own, so a failure in one does not hide the others.
# `v` is copied for every call: the products must not rely on it staying unchanged.
const DERIVATIVES = (;
    grad = ((; g = true), fi -> (G = zeros(2); fi.grad(G, X0); G)),
    fg = ((; fg = true), fi -> (G = zeros(2); y = fi.fg(G, X0); (y, G))),
    hess = ((; h = true), fi -> (H = zeros(2, 2); fi.hess(H, X0); H)),
    hv = ((; hv = true), fi -> (Hv = zeros(2); fi.hv(Hv, X0, copy(V)); Hv)),
    cons_j = ((; cons_j = true), fi -> (J = zeros(2, 2); fi.cons_j(J, X0); J)),
    cons_jvp = ((; cons_jvp = true), fi -> (Jv = zeros(2); fi.cons_jvp(Jv, X0, copy(V)); Jv)),
    cons_vjp = ((; cons_vjp = true), fi -> (Jv = zeros(2); fi.cons_vjp(Jv, X0, copy(V)); Jv)),
    cons_h = ((; cons_h = true), fi -> (Hs = [zeros(2, 2) for _ in 1:2]; fi.cons_h(Hs, X0); Hs)),
    lag_h = ((; lag_h = true), fi -> (H = zeros(2, 2); fi.lag_h(H, X0, Σ, Μ); H)),
)

function despecialized_instance(spec, ad, obj = rosenbrock, cons = rosenbrock_cons!, p = P0; flags...)
    of = OptimizationFunction{true, spec}(obj, ad; cons)
    f, pp = OptimizationBase.despecialize(of, p)
    return OptimizationBase.instantiate_function(
        f, OptimizationBase.ReInitCache(copy(X0), pp), ad, 2; flags...
    )
end

approx(a::Tuple, b::Tuple; kw...) = all(approx.(a, b; kw...))
approx(a::AbstractVector{<:AbstractArray}, b; kw...) = all(approx.(a, b; kw...))
approx(a, b; rtol) = isapprox(a, b; rtol, atol = rtol)

"""
    test_derivatives(ad; ops, rtol, broken)

Check the derivatives `ops` built for `ad` against ForwardDiff, for every specialization,
and that the despecialized ones are built from the erased functions and `p`. `broken`
lists the derivatives expected to fail.
"""
function test_derivatives(ad; ops = keys(DERIVATIVES), rtol = 1.0e-8, broken = ())
    ref = reference_derivatives()
    for spec in (FullSpecialize, AutoSpecialize, AutoDespecialize)
        @testset "$spec" begin
            of = OptimizationFunction{true, spec}(rosenbrock, ad; cons = rosenbrock_cons!)
            f, p = OptimizationBase.despecialize(of, P0)
            @test (f.f isa DespecializedFunction) == (spec !== FullSpecialize)
            @test (p isa DespecializedParameters) == (spec === AutoDespecialize)
            for op in ops
                flags, evaluate = DERIVATIVES[op]
                result = try
                    evaluate(despecialized_instance(spec, ad; flags...))
                catch e
                    e
                end
                ok = !(result isa Exception) && approx(result, ref[op]; rtol)
                if op in broken
                    @test_broken ok
                else
                    @test ok
                end
            end
        end
    end
    return
end
