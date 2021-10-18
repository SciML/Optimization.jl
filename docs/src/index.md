# GalacticOptim.jl

GalacticOptim.jl is a package with a scope that is beyond your normal global optimization
package. GalacticOptim.jl seeks to bring together all of the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package and you learn them all! GalacticOptim.jl adds a few high-level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all of the options in a single
unified interface.

##### Note: The package is still in active development.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
GalacticOptim.jl in the standard way:

```julia
import Pkg; Pkg.add("GalacticOptim")
```
The packages relevant to the core functionality of GalacticOptim.jl will be imported
accordingly and, in most cases, you do not have to worry about the manual
installation of dependencies. However, you will need to add the specific optimizer
packages.

## Overview of the Optimizers

| Package                  | Local Gradient-Based     | Local Hessian-Based      | Local Derivative-Free    | Local Constrained        | Global Unconstrained     | Global Constrained       |
|--------------------------|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|
| BlackBoxOptim            | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :heavy_multiplication_x: |
| CMEAEvolutionaryStrategy | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :heavy_multiplication_x: |
| Evolutionary             | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :o:                      |
| Flux                     | :heavy_check_mark:       | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: |
| GCMAES                   | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :heavy_multiplication_x: |
| MathOptInterface         | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_multiplication_x: |
| MultistartOptimization   | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :heavy_multiplication_x: |
| Metaheuristics           | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :o:                      |
| NOMAD                    | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :o:                      |
| NLopt                    | :heavy_check_mark:       | :heavy_multiplication_x: | :heavy_check_mark:       | :o:                      | :heavy_check_mark:       | :o:                      |
| Nonconvex                | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       | :o:                      | :heavy_check_mark:       | :o:                      |
| Optim                    | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       | :heavy_check_mark:       |
| QuadDIRECT               | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark:       | :heavy_multiplication_x: |

:heavy_check_mark: = supported

:o: = supported in downstream library but not yet implemented in `GalacticOptim`; PR to add this functionality are welcome

:heavy_multiplication_x: = not supported
