function barrier_calls(ex)
    ex isa Expr || return ex
    ex.head == :function && return ex
    if ex.head == :call && ex.args[1] === :check_case
        return Expr(:call, GlobalRef(Base, :invokelatest), ex.args...)
    end
    return Expr(ex.head, map(barrier_calls, ex.args)...)
end

include(barrier_calls, joinpath(@__DIR__, "standalone.jl"))
