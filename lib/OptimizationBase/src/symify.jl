function symify_cache(f::OptimizationFunction, prob, num_cons, manifold)
    throw("Structural analysis requires SymbolicAnalysis.jl to be loaded, either add `using SymbolicAnalysis` to your script or set `structural_analysis = false`.")
end
