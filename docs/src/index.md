# GalacticOptim.jl

GalacticOptim.jl is a package with a scope that is beyond your normal global optimization
package. GalacticOptim.jl seeks to bring together all of the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package and you learn them all! GalacticOptim.jl adds a few high-level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all of the options in a single
unified interface.

#### Note: This package is still in development. This guide is currently both an active documentation and a development roadmap.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
GalacticOptim.jl in the standard way:

```julia
import Pkg; Pkg.add("GalacticOptim")
```
The packages relevant to the core functionality of GalacticOptim.jl will be imported
accordingly and, in most cases, you do not have to worry about the manual
installation of dependencies. Below is the list of packages that need to be
installed explicitly if you intend to use the specific optimization algorithms
offered by them:

- [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl) (solver: `BBO()`)
- [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) (usage via the NLopt API;
see also the available [algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/))
- [MultistartOptimization.jl](https://github.com/tpapp/MultistartOptimization.jl)
(see also [this documentation](https://juliahub.com/docs/MultistartOptimization/cVZvi/0.1.0/))
- [QuadDIRECT.jl](https://github.com/timholy/QuadDIRECT.jl)
- [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) (see also [this documentation](https://wildart.github.io/Evolutionary.jl/dev/))
- [CMAEvolutionStrategy.jl](https://github.com/jbrea/CMAEvolutionStrategy.jl)
