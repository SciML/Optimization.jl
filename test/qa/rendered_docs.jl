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
