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
            "Metaheuristics.Algorithm{NSGA2} for sphere"=> [0.9179326495423358, 6.500479761845751],
            "Metaheuristics.Algorithm{NSGA3} for sphere"=>  [1.459119296264428, 5.316958184343253],
            "Metaheuristics.Algorithm{SPEA2} for sphere"=> [0.6866588142724173, 7.18284015333389],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for sphere"=> [0.8829537455812262, 6.5718236788783555],
            "Metaheuristics.Algorithm{MOEAD_DE} for sphere"=> [0.3976541196796912, 8.117594342458556],
            "Metaheuristics.Algorithm{SMS_EMOA} for sphere"=> [0.5003293369817386, 7.837151299208113],
            "Metaheuristics.Algorithm{NSGA2} for rastrigin"=> [3.2085851798990426, 11.008095021493556],
            "Metaheuristics.Algorithm{NSGA3} for rastrigin"=> [7.651119804304736, 8.720012335441673],
            "Metaheuristics.Algorithm{SPEA2} for rastrigin"=> [6.584283198538056, 11.285298996109109],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for rastrigin"=> [1.5700038876682554, 4.889538439797789],
            "Metaheuristics.Algorithm{MOEAD_DE} for rastrigin"=> [3.630324386882652, 11.255596145071413],
            "Metaheuristics.Algorithm{SMS_EMOA} for rastrigin"=> [2.429354909458187, 10.665738548398641],
            "Metaheuristics.Algorithm{NSGA2} for rosenbrock"=> [8.289098963014444, 690.3540872929669],
            "Metaheuristics.Algorithm{NSGA3} for rosenbrock"=> [5.809098608164398, 677.7633682236716] ,
            "Metaheuristics.Algorithm{SPEA2} for rosenbrock"=> [78.91822306586232, 426.30261147533],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for rosenbrock"=> [1.5700038876682554, 4.889538439797789],
            "Metaheuristics.Algorithm{MOEAD_DE} for rosenbrock"=> [14.107118783363653, 615.3036153255869],
            "Metaheuristics.Algorithm{SMS_EMOA} for rosenbrock"=> [30.756427683784672, 542.6831635776932],
            "Metaheuristics.Algorithm{NSGA2} for ackley"=> [2.240787163704834, 5.990002878952371],
            "Metaheuristics.Algorithm{NSGA3} for ackley"=> [4.440892098500626e-16, 6.593599079287213],
            "Metaheuristics.Algorithm{SPEA2} for ackley"=> [2.6329825257456503, 6.060788875236632],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for ackley"=> [1.5700038876682554, 4.889538439797789],
            "Metaheuristics.Algorithm{MOEAD_DE} for ackley"=> [4.440892098500626e-16, 6.593599079287213],
            "Metaheuristics.Algorithm{SMS_EMOA} for ackley"=> [2.1417573971815256, 6.015324307249184],
            "Metaheuristics.Algorithm{NSGA2} for dtlz2"=> [0.013283104966270814, 0.010808186786590583],
            "Metaheuristics.Algorithm{NSGA3} for dtlz2"=> [0.013428265441897881, 0.03589930489326534],
            "Metaheuristics.Algorithm{SPEA2} for dtlz2"=> [0.019006068021099495, 0.0009905093731377751],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for dtlz2"=> [1.5700038876682554, 4.889538439797789],
            "Metaheuristics.Algorithm{MOEAD_DE} for dtlz2"=> [0.027075258566241527, 0.00973958317460759],
            "Metaheuristics.Algorithm{SMS_EMOA} for dtlz2"=> [0.056304481489060705, 0.026075248436234502],
            "Metaheuristics.Algorithm{NSGA2} for schaffer_n2"=> [1.4034569322987955, 0.6647534264038837],
            "Metaheuristics.Algorithm{NSGA3} for schaffer_n2"=> [0.9679143048226242, 1.0326088684985277],
            "Metaheuristics.Algorithm{SPEA2} for schaffer_n2"=> [0.24369709729308642, 2.269069649724968],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for schaffer_n2"=> [3.632401400816196e-17, 4.9294679997494206e-17],
            "Metaheuristics.Algorithm{MOEAD_DE} for schaffer_n2"=> [0.059772358446876686, 3.0818369131591896],
            "Metaheuristics.Algorithm{SMS_EMOA} for schaffer_n2"=> [0.4978888767998813, 1.67543922644328],
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
                @test value ≈ objectives atol=1e-1
                end
            end
        end
    end
end
