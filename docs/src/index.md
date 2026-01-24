# Optimization.jl: A Unified Optimization Package

Optimization.jl provides the easiest way to create an optimization problem and solve it.
It enables rapid prototyping and experimentation with minimal syntax overhead by providing
a uniform interface to >25 optimization libraries, hence 100+ optimization solvers
encompassing almost all classes of optimization algorithms such as global, mixed-integer,
non-convex, second-order local, constrained, etc. It allows you to choose an
Automatic Differentiation (AD) backend by simply passing an argument to indicate
the package to use and automatically generates the efficient derivatives of the
objective and constraints while giving you the flexibility to switch between
different AD engines as per your problem. Additionally, Optimization.jl takes
care of passing problem specific information to solvers that can leverage it
such as the sparsity pattern of the hessian or constraint jacobian and the expression graph.

It extends the common SciML interface making it very easy to use for anyone
familiar with the SciML ecosystem. It is also very easy to extend to new
solvers and new problem types. The package is actively maintained and new
features are added regularly.

## Installation

In most instances you'll want to use some solver directly. For example, to use
the Optim set of solvers, you'd do:

```julia
Pkg.add("OptimizationOptimJL")
```

See the solver lists for more details. In many scenarios it's recommended to
have some automatic differentiation (AD) package installed, most tutorials will 
use some form AD and thus require installing the associated AD package.
AD choices are made with ADTypes, and thus it's recommended you also add the
`ADTypes.jl` package for most use cases.

Optimization.jl is simply a bundle/interface over many of these dependencies.
It may add some optional higher level behavior in the future but at this time
the top level package does not add any extra behavior.

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

## Overview of the solver packages in alphabetical order

```@raw html
<details>
  <summary><strong>BlackBoxOptim</strong></summary>
  - <strong>Global Methods</strong>
    - Zeroth order
    - Unconstrained
    - Box Constraints
</details>
<details>
  <summary><strong>CMAEvolutionaryStrategy</strong></summary>
  - <strong>Global Methods</strong>
    - Zeroth order
    - Unconstrained
    - Box Constraints
</details>
<details>
  <summary><strong>Evolutionary</strong></summary>
  - <strong>Global Methods</strong>
    - Zeroth order
    - Unconstrained
    - Box Constraints
    - Non-linear Constraints
</details>
<details>
  <summary><strong>GCMAES</strong></summary>
  - <strong>Global Methods</strong>
    - First order
    - Box Constraints
    - Unconstrained
</details>
<details>
  <summary><strong>Manopt</strong></summary>
  - <strong>Local Methods</strong>
    - First order
    - Second order
    - Zeroth order
    - Box Constraints
    - Constrained 🟡
  - <strong>Global Methods</strong>
    - Zeroth order
    - Unconstrained
</details>
<details>
  <summary><strong>MathOptInterface</strong></summary>
  - <strong>Local Methods</strong>
    - First order
    - Second order
    - Box Constraints
    - Constrained
  - <strong>Global Methods</strong>
    - First order
    - Second order
    - Constrained
</details>
<details>
  <summary><strong>MultistartOptimization</strong></summary>
  - <strong>Global Methods</strong>
    - Zeroth order
    - First order
    - Second order
    - Box Constraints
</details>
<details>
  <summary><strong>Metaheuristics</strong></summary>
  - <strong>Global Methods</strong>
    - Zeroth order
    - Unconstrained
    - Box Constraints
</details>
<details>
  <summary><strong>NOMAD</strong></summary>
  - <strong>Global Methods</strong>
    - Zeroth order
    - Unconstrained
    - Box Constraints
    - Constrained 🟡
</details>
<details>
  <summary><strong>NLopt</strong></summary>
  - <strong>Local Methods</strong>
    - First order
    - Zeroth order
    - Second order 🟡
    - Box Constraints
    - Local Constrained 🟡
  - <strong>Global Methods</strong>
    - Zeroth order
    - First order
    - Unconstrained
    - Constrained 🟡
</details>
<details>
  <summary><strong>Optim</strong></summary>
  - <strong>Local Methods</strong>
    - Zeroth order
    - First order
    - Second order
    - Box Constraints
    - Constrained
  - <strong>Global Methods</strong>
    - Zeroth order
    - Unconstrained
    - Box Constraints
</details>
<details>
  <summary><strong>PRIMA</strong></summary>
  - <strong>Local Methods</strong>
    - Derivative-Free: ✅
  - **Constraints**
    - Box Constraints: ✅
    - Local Constrained: ✅
</details>
<details>
  <summary><strong>QuadDIRECT</strong></summary>
  - **Constraints**
    - Box Constraints: ✅
  - <strong>Global Methods</strong>
    - Unconstrained: ✅
</details>
```

🟡 = supported in downstream library but not yet implemented in `Optimization.jl`; PR to add this functionality are welcome

## Citation

```
@software{vaibhav_kumar_dixit_2023_7738525,
	author = {Vaibhav Kumar Dixit and Christopher Rackauckas},
	month = mar,
	publisher = {Zenodo},
	title = {Optimization.jl: A Unified Optimization Package},
	version = {v3.12.1},
	doi = {10.5281/zenodo.7738525},
  	url = {https://doi.org/10.5281/zenodo.7738525},
	year = 2023}
```

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

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
