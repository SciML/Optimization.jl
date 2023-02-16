# Optimization.jl: A Unified Optimization Package

Optimization.jl is a package with a scope that is beyond your normal global optimization
package. Optimization.jl seeks to bring together all the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package, and you learn them all! Optimization.jl adds a few high-level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all the options in a single
unified interface.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
Optimization.jl in the standard way:

```julia
import Pkg
Pkg.add("Optimization")
```

The packages relevant to the core functionality of Optimization.jl will be imported
accordingly and, in most cases, you do not have to worry about the manual
installation of dependencies. However, you will need to add the specific optimizer
packages.

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Overview of the Optimizers

| Package                 | Local Gradient-Based | Local Hessian-Based | Local Derivative-Free | Box Constraints | Local Constrained | Global Unconstrained | Global Constrained   |
|:----------------------- |:--------------------:|:-------------------:|:---------------------:|:---------------:|:-----------------:|:--------------------:|:--------------------:|
| BlackBoxOptim           | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | ❌                  ✅ |
| CMAEvolutionaryStrategy | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | ❌                    |
| Evolutionary            | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | 🟡                    |
| Flux                    | ✅                    | ❌                   | ❌                     | ❌               | ❌                 | ❌                    | ❌                    |
| GCMAES                  | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | ❌                    |
| MathOptInterface        | ✅                    | ✅                   | ✅                     | ✅               | ✅                 | ✅                    | 🟡                    |
| MultistartOptimization  | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | ❌                    |
| Metaheuristics          | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | 🟡                    |
| NOMAD                   | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | 🟡                    |
| NLopt                   | ✅                    | ❌                   | ✅                     | ✅               | 🟡                 | ✅                    | 🟡                    |
| Nonconvex               | ✅                    | ✅                   | ✅                     | ✅               | 🟡                 | ✅                    | 🟡                    |
| Optim                   | ✅                    | ✅                   | ✅                     | ✅               | ✅                 | ✅                    | ✅                    |
| QuadDIRECT              | ❌                    | ❌                   | ❌                     | ✅               | ❌                 | ✅                    | ❌                    |

✅ = supported

🟡 = supported in downstream library but not yet implemented in `Optimization`; PR to add this functionality are welcome

❌ = not supported

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@raw html
You can also download the 
<a href="
```

```@eval
using TOML
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
       "/assets/Manifest.toml"
```

```@raw html
">manifest</a> file and the
<a href="
```

```@eval
using TOML
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
       "/assets/Project.toml"
```

```@raw html
">project</a> file.
```
