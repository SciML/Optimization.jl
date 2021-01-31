# [OptimizationFunction](@id optfunction)

The `OptimizationFunction` type is a function type that holds all of
the extra differentiation data required to do fast and accurate
optimization. The signature for the constructor is:

```julia
OptimizationFunction{iip}(f,adtype=NoAD();
                          grad=nothing,
                          hess=nothing,
                          hv=nothing,
                          cons=nothing,
                          cons_j=nothing,
                          cons_h=nothing)
```

The keyword arguments are as follows:

- `grad`: Gradient
- `hess`: Hessian
- `hv`: Hessian vector products `hv(du,u,p,t,v)` = H*v
- `cons`: Constraint function
- `cons_j`
- `cons_h`

### Defining Optimization Functions Via AD

While using the keyword arguments gives the user control over defining
all of the possible functions, the simplest way to handle the generation
of an `OptimizationFunction` is by specifying an AD type. By doing so,
this will automatically fill in all of the extra functions. For example,

```julia
OptimizationFunction(f,AutoZygote())
```

will use [Zygote.jl](https://github.com/FluxML/Zygote.jl) to define
all of the necessary functions. Note that if any functions are defined
directly, the auto-AD definition does not overwrite the user's choice.

The choices for the auto-AD fill-ins with quick descriptions are:

- `AutoForwardDiff()`: The fastest choice for small optimizations
- `AutoReverseDiff(compile=false)`: A fast choice for large scalar optimizations
- `AutoTracker()`: Like ReverseDiff but GPU-compatible
- `AutoZygote()`: The fastest choice
- `AutoFiniteDiff()`: Finite differencing, not optimal but always applicable
- `AutoModelingToolkit()`: The fastest choice for large scalar optimizations

The following sections describe the Auto-AD choices in detail.

### AutoForwardDiff

This uses the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
package. It is the fastest choice for small systems, especially with
heavy scalar interactions. It is easy to use and compatible with most
pure is Julia functions which have loose type restrictions. However,
because it's forward-mode, it scales poorly in comparison to other AD
choices. Hessian construction is suboptimal as it uses the forward-over-forward
approach.

- Compatible with GPUs
- Compatible with Hessian-based optimization
- Compatible with Hv-based optimization
- Compatible with constraints

### AutoReverseDiff

This uses the [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
package. `AutoReverseDiff` has a default argument, `compile`, which
denotes whether the reverse pass should be compiled. **`compile` should only
be set to `true` if `f` contains no branches (if statements, while loops)
otherwise it can produce incorrect derivatives!**.

`AutoReverseDiff` is generally applicable to many pure Julia codes,
and with `compile=true` it is one of the fastest options on code with
heavy scalar interactions. Hessian calculations are fast by mixing
ForwardDiff with ReverseDiff for forward-over-reverse. However, its
performance can falter when `compile=false`.

- Not compatible with GPUs
- Compatible with Hessian-based optimization by mixing with ForwardDiff
- Compatible with Hv-based optimization by mixing with ForwardDiff
- Not compatible with constraint functions

### AutoTracker

This uses the [Tracker.jl](https://github.com/FluxML/Tracker.jl) package.
Generally slower than ReverseDiff, it is generally applicable to many
pure Julia codes.

- Compatible with GPUs
- Not compatible with Hessian-based optimization
- Not compatible with Hv-based optimization
- Not compatible with constraint functions

### AutoZygote

This uses the [Zygote.jl](https://github.com/FluxML/Zygote.jl) package.
This is the staple reverse-mode AD that handles a large portion of
Julia with good efficiency. Hessian construction is fast via
forward-over-reverse mixing ForwardDiff.jl with Zygote.jl

- Compatible with GPUs
- Compatible with Hessian-based optimization via ForwardDiff
- Compatible with Hv-based optimization via ForwardDiff
- Not compatible with constraint functions

### AutoFiniteDiff

This uses [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).
While to necessarily the most efficient in any case, this is the only
choice that doesn't require the `f` function to be automatically
differentiable, which means it applies to any choice. However, because
it's using finite differencing, one needs to be careful as this procedure
introduces numerical error into the derivative estimates.

- Compatible with GPUs
- Compatible with Hessian-based optimization
- Compatible with Hv-based optimization
- Not compatible with constraint functions

### AutoModelingToolkit

This uses the [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
symbolic system for automatically converting the `f` function into
a symbolic equation and uses symbolic differentiation in order to generate
a fast derivative code. Note that this will also compile a new version
of your `f` function that is automatically optimized. In this choice,
it defaults to `grad=false` and `hess=false`, and one must change these
to `true` in order to enable the symbolic derivation. Future updates
will enable automatic parallelization and sparsity in the derived
functions. This can be the fastest for many systems, especially when
parallelization and sparsity are required, but can take the longest
to generate.

- Not compatible with GPUs
- Compatible with Hessian-based optimization
- Not compatible with Hv-based optimization
- Not compatible with constraint functions
