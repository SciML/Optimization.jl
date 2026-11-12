using OptimizationBase, SciMLBase, ADTypes, Test
using ForwardDiff, Enzyme, Zygote
import DifferentiationInterface as DI

# `instantiate_function` builds each derivative in a branch selected by a request flag.
# Only the requested branches may be inferred: with Enzyme, inferring a preparation that
# evaluates a derivative compiles that derivative. So with a user-supplied gradient and only
# `g` requested, no preparation of the objective or the constraints may have a method
# instance, whatever the backend. Each case uses its own functions, so the method instances
# it checks cannot come from another case. A preparation whose callable is already inferred as
# `Any` (e.g. a boxed constraint wrapper) leaves no trace in this check, but inference does
# not compile a derivative for it either.

const PREPARATIONS = (
    DI.prepare_gradient, DI.prepare_hessian, DI.prepare_hvp, DI.prepare_jacobian,
    DI.prepare_pullback, DI.prepare_pushforward,
)

function inferred_preparations(fs...; preparations = PREPARATIONS)
    names = string.(nameof.(fs))
    found = String[]
    for prepare in preparations
        # calls with keyword arguments are inferred as `Core.kwcall` specializations
        ms = [methods(prepare); methods(Core.kwcall, Tuple{Any, typeof(prepare), Vararg{Any}})]
        for m in ms, mi in Base.specializations(m)
            sig = string(mi.specTypes)
            any(n -> occursin(n, sig), names) && push!(found, sig)
        end
    end
    return found
end

enzyme = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
sparse_ad(dense) = AutoSparse(
    dense; sparsity_detector = OptimizationBase.TracerSparsityDetector(),
    coloring_algorithm = OptimizationBase.GreedyColoringAlgorithm()
)
const BACKENDS = [
    ("ForwardDiff", AutoForwardDiff()),
    ("sparse ForwardDiff", sparse_ad(AutoForwardDiff())),
    ("Zygote", AutoZygote()),
    ("sparse Zygote", sparse_ad(AutoZygote())),
    ("sparse Enzyme", sparse_ad(enzyme)),
    ("SecondOrder with Enzyme", DI.SecondOrder(AutoForwardDiff(), enzyme)),
    ("sparse MixedMode with Enzyme", sparse_ad(DI.MixedMode(AutoForwardDiff(), enzyme))),
]

# one objective and constraint function per case, defined at top level
for k in 0:length(BACKENDS)
    @eval begin
        $(Symbol(:up_obj_, k, :_iip))(x, p) = sum(abs2, x)
        $(Symbol(:up_cons_, k, :_iip))(res, x, p) = (res .= [x[1] * x[2]]; nothing)
        $(Symbol(:up_obj_, k, :_oop))(x, p) = sum(abs2, x)
        $(Symbol(:up_cons_, k, :_oop))(x, p) = [x[1] * x[2]]
    end
end
up_grad!(G, x, p) = (G .= 2 .* x; nothing)
up_grad(x, p) = 2 .* x
up_fun(k, kind, iip) = getfield(@__MODULE__, Symbol(:up_, kind, :_, k, iip ? :_iip : :_oop))

function instantiate(k, adtype, iip; flags...)
    optf = OptimizationFunction{iip}(
        up_fun(k, :obj, iip), adtype; grad = iip ? up_grad! : up_grad,
        cons = up_fun(k, :cons, iip)
    )
    return OptimizationBase.instantiate_function(optf, [0.5, 1.5], adtype, nothing, 1; flags...)
end

@testset "no unrequested preparation is inferred: $label, $(iip ? "in place" : "out of place")" for (k, (label, adtype)) in enumerate(BACKENDS), iip in (true, false)
    inst = instantiate(k, adtype, iip; g = true)
    x0 = [0.5, 1.5]
    G = iip ? (G = similar(x0); inst.grad(G, x0); G) : inst.grad(x0)
    @test G == 2 .* x0
    @test isempty(inferred_preparations(up_fun(k, :obj, iip), up_fun(k, :cons, iip)))
end

# control: a requested derivative is prepared, and the check sees its preparation
@testset "a requested preparation is inferred" begin
    inst = instantiate(0, AutoForwardDiff(), true; h = true)
    H = zeros(2, 2)
    inst.hess(H, [0.5, 1.5])
    @test H ≈ [2.0 0.0; 0.0 2.0]
    @test !isempty(inferred_preparations(up_fun(0, :obj, true); preparations = (DI.prepare_hessian,)))
end
