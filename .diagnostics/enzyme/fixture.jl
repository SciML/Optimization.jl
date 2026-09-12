using InteractiveUtils, Enzyme

versioninfo()
println("Enzyme ", pkgversion(Enzyme), "; CPU ", Sys.CPU_NAME)
flush(stdout)
include(joinpath(ENV["OPTIMIZATION_SUBJECT"], "lib/OptimizationBase/test/AD/enzyme_lagrangian_hessian.jl"))
