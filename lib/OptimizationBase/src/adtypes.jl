"""
    AutoEnzyme <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:
```julia
OptimizationFunction(f, AutoEnzyme(); kwargs...)
```
This uses the [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) package. Enzyme performs automatic differentiation on the LLVM IR code generated from julia.
It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.
  - Compatible with GPUs
  - Compatible with Hessian-based optimization
  - Compatible with Hv-based optimization
  - Compatible with constraints
Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the Hessian
is not defined via Enzyme.
"""
AutoEnzyme

"""
    AutoFiniteDiff{T1,T2,T3} <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoFiniteDiff(); kwargs...)
```

This uses [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).
While not necessarily the most efficient, this is the only
choice that doesn't require the `f` function to be automatically
differentiable, which means it applies to any choice. However, because
it's using finite differencing, one needs to be careful as this procedure
introduces numerical error into the derivative estimates.

  - Compatible with GPUs
  - Compatible with Hessian-based optimization
  - Compatible with Hv-based optimization
  - Compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via FiniteDiff.

## Constructor

```julia
AutoFiniteDiff(; fdtype = Val(:forward)fdjtype = fdtype, fdhtype = Val(:hcentral))
```

  - `fdtype`: the method used for defining the gradient
  - `fdjtype`: the method used for defining the Jacobian of constraints.
  - `fdhtype`: the method used for defining the Hessian

For more information on the derivative type specifiers, see the
[FiniteDiff.jl documentation](https://github.com/JuliaDiff/FiniteDiff.jl).
"""
AutoFiniteDiff

"""
    AutoForwardDiff{chunksize} <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoForwardDiff(); kwargs...)
```

This uses the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
package. It is the fastest choice for small systems, especially with
heavy scalar interactions. It is easy to use and compatible with most
Julia functions which have loose type restrictions. However,
because it's forward-mode, it scales poorly in comparison to other AD
choices. Hessian construction is suboptimal as it uses the forward-over-forward
approach.

  - Compatible with GPUs
  - Compatible with Hessian-based optimization
  - Compatible with Hv-based optimization
  - Compatible with constraints

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via ForwardDiff.
"""
AutoForwardDiff

"""
    AutoModelingToolkit <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoModelingToolkit(); kwargs...)
```

This uses the [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
package's `modelingtookitize` functionality to generate the derivatives and other fields of an `OptimizationFunction`.
This backend creates the symbolic expressions for the objective and its derivatives as well as
the constraints and their derivatives. Through `structural_simplify`, it enforces simplifications
that can reduce the number of operations needed to compute the derivatives of the constraints. This automatically
generates the expression graphs that some solver interfaces through OptimizationMOI like
[AmplNLWriter.jl](https://github.com/jump-dev/AmplNLWriter.jl) require.

  - Compatible with GPUs
  - Compatible with Hessian-based optimization
  - Compatible with Hv-based optimization
  - Compatible with constraints

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not generated via ModelingToolkit.

## Constructor

```julia
AutoModelingToolkit(false, false)
```

  - `obj_sparse`: to indicate whether the objective hessian is sparse.
  - `cons_sparse`: to indicate whether the constraints' jacobian and hessian are sparse.

"""
AutoModelingToolkit

"""
    AutoReverseDiff <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoReverseDiff(); kwargs...)
```

This uses the [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
package. `AutoReverseDiff` has a default argument, `compile`, which
denotes whether the reverse pass should be compiled. **`compile` should only
be set to `true` if `f` contains no branches (if statements, while loops)
otherwise it can produce incorrect derivatives!**

`AutoReverseDiff` is generally applicable to many pure Julia codes,
and with `compile=true` it is one of the fastest options on code with
heavy scalar interactions. Hessian calculations are fast by mixing
ForwardDiff with ReverseDiff for forward-over-reverse. However, its
performance can falter when `compile=false`.

  - Not compatible with GPUs
  - Compatible with Hessian-based optimization by mixing with ForwardDiff
  - Compatible with Hv-based optimization by mixing with ForwardDiff
  - Not compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via ReverseDiff.

## Constructor

```julia
AutoReverseDiff(; compile = false)
```

#### Note: currently, compilation is not defined/used!
"""
AutoReverseDiff

"""
    AutoTracker <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoTracker(); kwargs...)
```

This uses the [Tracker.jl](https://github.com/FluxML/Tracker.jl) package.
Generally slower than ReverseDiff, it is generally applicable to many
pure Julia codes.

  - Compatible with GPUs
  - Not compatible with Hessian-based optimization
  - Not compatible with Hv-based optimization
  - Not compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via Tracker.
"""
AutoTracker

"""
    AutoZygote <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoZygote(); kwargs...)
```

This uses the [Zygote.jl](https://github.com/FluxML/Zygote.jl) package.
This is the staple reverse-mode AD that handles a large portion of
Julia with good efficiency. Hessian construction is fast via
forward-over-reverse mixing ForwardDiff.jl with Zygote.jl

  - Compatible with GPUs
  - Compatible with Hessian-based optimization via ForwardDiff
  - Compatible with Hv-based optimization via ForwardDiff
  - Not compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via Zygote.
"""
AutoZygote

function generate_adtype(adtype)
    if !(adtype isa SciMLBase.NoAD || adtype isa DifferentiationInterface.SecondOrder ||
         adtype isa AutoZygote)
        soadtype = DifferentiationInterface.SecondOrder(adtype, adtype)
    elseif adtype isa AutoZygote
        soadtype = DifferentiationInterface.SecondOrder(AutoForwardDiff(), adtype)
    elseif adtype isa DifferentiationInterface.SecondOrder
        soadtype = adtype
        adtype = adtype.inner
    elseif adtype isa SciMLBase.NoAD
        soadtype = adtype
        adtype = adtype
    end
    return adtype, soadtype
end

function spadtype_to_spsoadtype(adtype)
    if !(adtype.dense_ad isa SciMLBase.NoAD ||
         adtype.dense_ad isa DifferentiationInterface.SecondOrder ||
         adtype.dense_ad isa AutoZygote)
        soadtype = AutoSparse(
            DifferentiationInterface.SecondOrder(adtype.dense_ad, adtype.dense_ad),
            sparsity_detector = adtype.sparsity_detector,
            coloring_algorithm = adtype.coloring_algorithm)
    elseif adtype.dense_ad isa AutoZygote
        soadtype = AutoSparse(
            DifferentiationInterface.SecondOrder(AutoForwardDiff(), adtype.dense_ad),
            sparsity_detector = adtype.sparsity_detector,
            coloring_algorithm = adtype.coloring_algorithm)
    else
        soadtype = adtype
    end
    return soadtype
end

function filled_spad(adtype)
    if adtype.sparsity_detector isa ADTypes.NoSparsityDetector &&
       adtype.coloring_algorithm isa ADTypes.NoColoringAlgorithm
        adtype = AutoSparse(adtype.dense_ad; sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm())
    elseif adtype.sparsity_detector isa ADTypes.NoSparsityDetector &&
           !(adtype.coloring_algorithm isa ADTypes.NoColoringAlgorithm)
        adtype = AutoSparse(adtype.dense_ad; sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = adtype.coloring_algorithm)
    elseif !(adtype.sparsity_detector isa ADTypes.NoSparsityDetector) &&
           adtype.coloring_algorithm isa ADTypes.NoColoringAlgorithm
        adtype = AutoSparse(adtype.dense_ad; sparsity_detector = adtype.sparsity_detector,
            coloring_algorithm = GreedyColoringAlgorithm())
    end
end

function generate_sparse_adtype(adtype)
    if !(adtype.dense_ad isa DifferentiationInterface.SecondOrder)
        adtype = filled_spad(adtype)
        soadtype = spadtype_to_spsoadtype(adtype)
    else
        soadtype = adtype
        adtype = AutoSparse(
            adtype.dense_ad.inner,
            sparsity_detector = soadtype.sparsity_detector,
            coloring_algorithm = soadtype.coloring_algorithm)
        adtype = filled_spad(adtype)
        soadtype = filled_spad(soadtype)
    end

    return adtype, soadtype
end
