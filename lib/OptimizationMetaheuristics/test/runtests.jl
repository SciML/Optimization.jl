using OptimizationMetaheuristics, Optimization
using Test

@testset "OptimizationMetaheuristics.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0],
        ub = [1.5, 1.5])
    sol = solve(prob, ECA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.DE())
    @test 10 * sol.objective < l1

    sol = solve(prob, PSO())
    @test 10 * sol.objective < l1

    sol = solve(prob, ABC())
    @test 10 * sol.objective < l1

    sol = solve(prob, CGSA(N = 100))
    @test 10 * sol.objective < l1

    sol = solve(prob, SA())
    @test 10 * sol.objective < l1

    sol = solve(prob, WOA())
    @test 10 * sol.objective < l1

    sol = solve(prob, ECA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.DE(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, PSO(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, ABC(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, CGSA(N = 100), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, SA(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, WOA(), use_initial = true)
    @test 10 * sol.objective < l1

    # Define the benchmark functions as multi-objective problems
function sphere(x)
    f1 = sum(x .^ 2)
    f2 = sum((x .- 2.0) .^ 2)
    gx = [0.0]
    hx = [0.0]
    return [f1, f2], gx, hx
end

function rastrigin(x)
    f1 = sum(x .^ 2 .- 10 .* cos.(2 .* π .* x) .+ 10)
    f2 = sum((x .- 2.0) .^ 2 .- 10 .* cos.(2 .* π .* (x .- 2.0)) .+ 10)
    gx = [0.0]
    hx = [0.0]
    return [f1, f2], gx, hx
end

function rosenbrock(x)
    f1 = sum(100 .* (x[2:end] .- x[1:end-1] .^ 2) .^ 2 .+ (x[1:end-1] .- 1) .^ 2)
    f2 = sum(100 .* ((x[2:end] .- 2.0) .- (x[1:end-1] .^ 2)) .^ 2 .+ ((x[1:end-1] .- 1.0) .^ 2))
    gx = [0.0]
    hx = [0.0]
    return [f1, f2], gx, hx
end

function ackley(x)
    f1 = -20 * exp(-0.2 * sqrt(sum(x .^ 2) / length(x))) - exp(sum(cos.(2 * π .* x)) / length(x)) + 20 + ℯ
    f2 = -20 * exp(-0.2 * sqrt(sum((x .- 2.0) .^ 2) / length(x))) - exp(sum(cos.(2 * π .* (x .- 2.0))) / length(x)) + 20 + ℯ
    gx = [0.0]
    hx = [0.0]
    return [f1, f2], gx, hx
end


function dtlz2(x)
    g = sum((x[3:end] .- 0.5) .^ 2)
    f1 = (1 + g) * cos(x[1] * π / 2) * cos(x[2] * π / 2)
    f2 = (1 + g) * cos(x[1] * π / 2) * sin(x[2] * π / 2)
    gx = [0.0]
    hx = [0.0]
    return [f1, f2], gx, hx
end

function schaffer_n2(x)
    f1 = x[1]^2
    f2 = (x[1] - 2.0)^2
    gx = [0.0]
    hx = [0.0]
    return [f1, f2], gx, hx
end
OBJECTIVES = Dict(
            "Metaheuristics.Algorithm{NSGA2} for sphere"=> [0.4761274648673104, 7.888859360956367],
            "Metaheuristics.Algorithm{NSGA3} for sphere"=> [1.1245011962315388, 5.9084439601220105],
            "Metaheuristics.Algorithm{SPEA2} for sphere"=> [0.45500157273715425, 8.060476156495577],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for sphere"=> [0.8537159192703154, 6.721186217733861],
            "Metaheuristics.Algorithm{MOEAD_DE} for sphere"=> [1.7135443166012259, 4.818225194026158],
            "Metaheuristics.Algorithm{SMS_EMOA} for sphere"=> [1.1376191314229631, 5.935092118744685],
            "Metaheuristics.Algorithm{NSGA2} for rastrigin"=> [3.914962881168682, 11.552205533592897],
            "Metaheuristics.Algorithm{NSGA3} for rastrigin"=> [4.842031386209626, 5.542348181529025],
            "Metaheuristics.Algorithm{SPEA2} for rastrigin"=> [2.9692594618763835, 10.596356482458171],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for rastrigin"=> [0.4152393951206974, 7.953188854042798],
            "Metaheuristics.Algorithm{MOEAD_DE} for rastrigin"=> [0.0, 12.0],
            "Metaheuristics.Algorithm{SMS_EMOA} for rastrigin"=> [10.668382998122013, 11.672554721420616],
            "Metaheuristics.Algorithm{NSGA2} for rosenbrock"=> [13.564144823755003, 608.7768632268896],
            "Metaheuristics.Algorithm{NSGA3} for rosenbrock"=> [41.32512246661068, 479.9472092328193],
            "Metaheuristics.Algorithm{SPEA2} for rosenbrock"=> [20.921291737001457, 566.887198567844],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for rosenbrock"=> [0.4152393951206974, 7.953188854042798],
            "Metaheuristics.Algorithm{MOEAD_DE} for rosenbrock"=> [2.215363988408552, 723.1454508385998],
            "Metaheuristics.Algorithm{SMS_EMOA} for rosenbrock"=> [20.27041333432111, 575.7366151959259],
            "Metaheuristics.Algorithm{NSGA2} for ackley"=> [3.4438643047130992, 5.9371415671384895],
            "Metaheuristics.Algorithm{NSGA3} for ackley"=> [3.4659156540969573, 5.287995047899489],
            "Metaheuristics.Algorithm{SPEA2} for ackley"=> [2.3209460118197716, 5.918573168574383],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for ackley"=> [0.4152393951206974, 7.953188854042798],
            "Metaheuristics.Algorithm{MOEAD_DE} for ackley"=> [4.440892098500626e-16, 6.593599079287213],
            "Metaheuristics.Algorithm{SMS_EMOA} for ackley"=> [2.4079028491253074, 6.085847745455787],
            "Metaheuristics.Algorithm{NSGA2} for dtlz2"=> [0.0008621981163705847, 0.016776532222616037],
            "Metaheuristics.Algorithm{NSGA3} for dtlz2"=> [0.00530717096691627, 0.006810762449448562],
            "Metaheuristics.Algorithm{SPEA2} for dtlz2"=> [0.0022573638805422967, 0.0012875185095928014],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for dtlz2"=> [2.9276186095638996, 3.0744092709040185],
            "Metaheuristics.Algorithm{MOEAD_DE} for dtlz2"=> [0.0009460864848779976, 0.015153151632789923],
            "Metaheuristics.Algorithm{SMS_EMOA} for dtlz2"=> [0.006063356611750317, 0.014614126585905095],
            "Metaheuristics.Algorithm{NSGA2} for schaffer_n2"=> [1.0978202866371685, 0.9067435054036517],
            "Metaheuristics.Algorithm{NSGA3} for schaffer_n2"=> [2.755035084049435, 0.11571574056316469],
            "Metaheuristics.Algorithm{SPEA2} for schaffer_n2"=> [2.2990190172651723, 0.23401248171694122],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for schaffer_n2"=> [0.0, 800.0],
            "Metaheuristics.Algorithm{MOEAD_DE} for schaffer_n2"=> [0.0017365039124724727, 3.8350509838468123],
            "Metaheuristics.Algorithm{SMS_EMOA} for schaffer_n2"=> [0.7559493982502018, 1.278135376195079],
    )
    # Define the testset
@testset "Multi-Objective Optimization with Various Functions and Metaheuristics" begin
    # Define the problems and their bounds
    problems = [
        (sphere, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        (rastrigin, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        (rosenbrock, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        (ackley, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        (dtlz2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        (schaffer_n2, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    ]

    nobjectives = 2
    npartitions = 100

    # Define the different algorithms
    algs = [
        NSGA2(),
        NSGA3(),
        SPEA2(),
        CCMO(NSGA2(N=100, p_m=0.001)),
        MOEAD_DE(gen_ref_dirs(nobjectives, npartitions), options=Options(debug=false, iterations = 250)),
        SMS_EMOA()
    ]

    # Run tests for each problem and algorithm
    for (prob_func, lb, ub) in problems
        prob_name = string(prob_func)
        for alg in algs
            alg_name = string(typeof(alg))
            @testset "$alg_name on $prob_name" begin
                multi_obj_fun = MultiObjectiveOptimizationFunction((x, p) -> prob_func(x))
                prob = OptimizationProblem(multi_obj_fun, lb; lb = lb, ub = ub)
                if (alg_name=="Metaheuristics.Algorithm{CCMO{NSGA2}}")
                    sol = solve(prob, alg)
                else
                    sol = solve(prob, alg; maxiters = 100, use_initial = true)
                end

                # Tests
                @test !isempty(sol.minimizer)  # Check that a solution was found
                
                # Use sol.objective to get the objective values
                key = "$alg_name for $prob_name"
                value = OBJECTIVES[key]
                objectives = sol.objective
                @test value==objectives
                end
            end
        end
    end
end
