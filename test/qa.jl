using Optimization, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(Optimization)
    Aqua.test_ambiguities(Optimization, recursive = false)
    Aqua.test_deps_compat(Optimization)
    Aqua.test_piracies(Optimization,
        treat_as_own = [OptimizationProblem,
            Optimization.SciMLBase.AbstractOptimizationCache])
    Aqua.test_project_extras(Optimization)
    if VERSION > v"1.10"
        # in CI we need to dev packages to run the tests
        # which adds stale deps
        # on later versions [sources] is used instead
        Aqua.test_stale_deps(Optimization)
    end
    Aqua.test_unbound_args(Optimization)
    Aqua.test_undefined_exports(Optimization)
end
