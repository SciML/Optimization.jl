using OptimizationBase, SciMLBase, Test

# An `AbstractVector{Expr}` that records which elements were read.
struct ReadTrackingExprs <: AbstractVector{Expr}
    exprs::Vector{Expr}
    reads::Vector{Int}
end
Base.size(r::ReadTrackingExprs) = size(r.exprs)
Base.getindex(r::ReadTrackingExprs, i::Int) = (push!(r.reads, i); r.exprs[i])

@testset "NoAD instantiation does not read a lazy `cons_expr`" begin
    cons = (res, x, p) -> (res .= [x[1] + x[2], x[1] * x[2]])
    x0 = [1.0, 2.0]
    for T in (OptimizationFunction, MultiObjectiveOptimizationFunction)
        # `symbolify` turns the function objects `+` and `getindex` into symbols
        row = Expr(:call, +, Expr(:call, getindex, :x, 1), :(x[2]))
        tracked = ReadTrackingExprs([row, :(x[1] * x[2])], Int[])
        obj = T === OptimizationFunction ? (x, p) -> sum(abs2, x) : (x, p) -> [sum(abs2, x)]
        f = T(obj, SciMLBase.NoAD(); cons, cons_expr = tracked)
        f2 = OptimizationBase.instantiate_function(f, x0, SciMLBase.NoAD(), nothing, 2)
        @test isempty(tracked.reads)
        @test length(f2.cons_expr) == 2
        @test isempty(tracked.reads)
        e = f2.cons_expr[1]
        @test e == :(getindex(x, 1) + x[2])
        @test e.args[1] === :+ && e.args[2].args[1] === :getindex
        @test tracked.reads == [1]
    end

    # a plain `Vector{Expr}` is still symbolified eagerly into a `Vector{Expr}`
    f = OptimizationFunction(
        (x, p) -> sum(abs2, x), SciMLBase.NoAD(); cons,
        cons_expr = [Expr(:call, getindex, :x, 1)]
    )
    f2 = OptimizationBase.instantiate_function(f, x0, SciMLBase.NoAD(), nothing, 1)
    @test f2.cons_expr isa Vector{Expr}
    @test f2.cons_expr[1].args[1] === :getindex
end
