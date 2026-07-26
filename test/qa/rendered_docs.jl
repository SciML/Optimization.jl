using SciMLTesting: public_api_names

const OPTIMIZATION_DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))

const OPTIMIZATION_OWNED_MODULES = Set(
    (
        :Optimization,
        :OptimizationAuglag,
        :OptimizationBBO,
        :OptimizationBase,
        :OptimizationCMAEvolutionStrategy,
        :OptimizationEvolutionary,
        :OptimizationGCMAES,
        :OptimizationIpopt,
        :OptimizationLBFGSB,
        :OptimizationMOI,
        :OptimizationMadNLP,
        :OptimizationManopt,
        :OptimizationMetaheuristics,
        :OptimizationMultistartOptimization,
        :OptimizationNLPModels,
        :OptimizationNLopt,
        :OptimizationNOMAD,
        :OptimizationODE,
        :OptimizationOptimJL,
        :OptimizationOptimisers,
        :OptimizationPRIMA,
        :OptimizationPolyalgorithms,
        :OptimizationPyCMA,
        :OptimizationQuadDIRECT,
        :OptimizationSciPy,
        :OptimizationSophia,
        :OptimizationSpeedMapping,
        :SimpleOptimization,
    )
)

function optimization_rendered_doc_names(docs_src::AbstractString)
    rendered = Set{Symbol}()
    isdir(docs_src) || return rendered
    for (root, _, files) in walkdir(docs_src)
        for file in files
            endswith(file, ".md") || continue
            in_docs = false
            for raw in eachline(joinpath(root, file))
                line = strip(raw)
                if startswith(line, "```@docs")
                    in_docs = true
                    continue
                elseif startswith(line, "```")
                    in_docs = false
                    continue
                end
                in_docs || continue
                isempty(line) && continue
                token = first(split(line))
                token = first(split(token, '('))
                dot = findlast(==('.'), token)
                dot === nothing || (token = token[nextind(token, dot):end])
                push!(rendered, Symbol(token))
            end
        end
    end
    return rendered
end

function optimization_dependency_rendered_ignore(pkg::Module)
    rendered = optimization_rendered_doc_names(OPTIMIZATION_DOCS_SRC)
    ignored = Symbol[]
    for name in public_api_names(pkg)
        name in rendered && continue
        isdefined(pkg, name) || continue
        owner = try
            value = getproperty(pkg, name)
            nameof(parentmodule(value))
        catch
            try
                nameof(parentmodule(typeof(getproperty(pkg, name))))
            catch
                nameof(pkg)
            end
        end
        owner in OPTIMIZATION_OWNED_MODULES || push!(ignored, name)
    end
    return Tuple(sort!(unique(ignored)))
end

# `run_qa`'s reexport audit flags every public name a package does not own. The glue
# packages deliberately surface their backend's solver names — that is what they are for
# — so allow the public API of the modules each one still `@reexport`s. Everything else
# must be owned; the umbrella packages instead list their curated re-surface explicitly.
function optimization_reexports_allow(mods::Module...)
    allowed = Symbol[]
    for m in mods
        push!(allowed, nameof(m))
        append!(allowed, public_api_names(m))
    end
    return Tuple(sort!(unique!(allowed)))
end

# The names Optimization/OptimizationBase intentionally re-surface from SciMLBase and
# ADTypes so that `using Optimization` is enough to state and solve a problem. Adding to
# this list is a deliberate widening of the public API, not an accident of `@reexport`.
const OPTIMIZATION_CURATED_REEXPORTS = (
    :AutoEnzyme, :AutoFiniteDiff, :AutoForwardDiff, :AutoMooncake, :AutoReverseDiff,
    :AutoSparse, :AutoSymbolics, :AutoTracker, :AutoZygote,
    :MaxSense, :MinSense, :ObjSense,
    :MultiObjectiveOptimizationFunction, :OptimizationFunction, :OptimizationProblem,
    :OptimizationSolution, :OptimizationStats,
    :ReturnCode, :allowsbounds, :allowscallback, :allowsconstraints,
    :init, :remake, :requiresbounds, :requiresconshess, :requiresconsjac,
    :requiresconstraints, :requiresgradient, :requireshessian, :solve, :solve!,
)
