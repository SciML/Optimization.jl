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
            "Metaheuristics.Algorithm{NSGA2} for sphere"=> [1.2825775684488987, 5.542698673334447],
            "Metaheuristics.Algorithm{NSGA3} for sphere"=> [1.4037392987204247, 5.27960233564319],
            "Metaheuristics.Algorithm{SPEA2} for sphere"=> [0.7489097792697987, 6.810116922148639],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for sphere"=> [0.705560866088165, 7.099072840195393],
            "Metaheuristics.Algorithm{MOEAD_DE} for sphere"=> [0.07806395081944045, 10.569578390542956],
            "Metaheuristics.Algorithm{SMS_EMOA} for sphere"=> [0.1786522135251722, 9.510830918969237],
            "Metaheuristics.Algorithm{NSGA2} for rastrigin"=> [0.0, 12.0],
            "Metaheuristics.Algorithm{NSGA3} for rastrigin"=> [3.6211843795326253, 8.094700083634313],
            "Metaheuristics.Algorithm{SPEA2} for rastrigin"=> [4.621746036074858, 8.865705515641517],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for rastrigin"=> [1.1057418927099407, 5.822357866887147],
            "Metaheuristics.Algorithm{MOEAD_DE} for rastrigin"=> [4.066568903563153, 8.511756354624936],
            "Metaheuristics.Algorithm{SMS_EMOA} for rastrigin"=> [3.713936191157112, 11.089405465875496],
            "Metaheuristics.Algorithm{NSGA2} for rosenbrock"=> [13.066572378560455, 631.9299839626948],
            "Metaheuristics.Algorithm{NSGA3} for rosenbrock"=> [10.050422620361184, 638.7582963114556],
            "Metaheuristics.Algorithm{SPEA2} for rosenbrock"=> [32.65994969150141, 531.6251481922213],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for rosenbrock"=> [1.1057418927099407, 5.822357866887147],
            "Metaheuristics.Algorithm{MOEAD_DE} for rosenbrock"=> [57.11598017501275, 436.3573486523958],
            "Metaheuristics.Algorithm{SMS_EMOA} for rosenbrock"=> [69.55872489888084, 425.9708273845619],
            "Metaheuristics.Algorithm{NSGA2} for ackley"=> [3.5426389168269847, 6.192082150329497],
            "Metaheuristics.Algorithm{NSGA3} for ackley"=> [2.600939673226922, 6.418843253009765],
            "Metaheuristics.Algorithm{SPEA2} for ackley"=> [3.969736656218448, 4.704118905245824],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for ackley"=> [1.1057418927099407, 5.822357866887147],
            "Metaheuristics.Algorithm{MOEAD_DE} for ackley"=> [3.014117404434334, 5.189544505776976],
            "Metaheuristics.Algorithm{SMS_EMOA} for ackley"=> [2.4105461048882373, 5.915339334024861],
            "Metaheuristics.Algorithm{NSGA2} for dtlz2"=> [0.02392674091616439, 0.009783549539645912],
            "Metaheuristics.Algorithm{NSGA3} for dtlz2"=> [0.003629941363507031, 0.011059594788256341],
            "Metaheuristics.Algorithm{SPEA2} for dtlz2"=> [0.00023267753741465326, 0.08058561661765253],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for dtlz2"=> [1.1057418927099407, 5.822357866887147],
            "Metaheuristics.Algorithm{MOEAD_DE} for dtlz2"=> [0.03199207429508088, 0.0012836221622943944],
            "Metaheuristics.Algorithm{SMS_EMOA} for dtlz2"=> [0.13316508540906347, 0.0008924220223277544],
            "Metaheuristics.Algorithm{NSGA2} for schaffer_n2"=> [2.171298522190922, 0.27716785218921663],
            "Metaheuristics.Algorithm{NSGA3} for schaffer_n2"=> [0.0403717729174977, 3.2366626422738527],
            "Metaheuristics.Algorithm{SPEA2} for schaffer_n2"=> [2.5115325783832896, 0.17240635672366453],
            "Metaheuristics.Algorithm{CCMO{NSGA2}} for schaffer_n2"=> [0.0, 800.0],
            "Metaheuristics.Algorithm{MOEAD_DE} for schaffer_n2"=> [0.019151313090508694, 3.4655982357364583],
            "Metaheuristics.Algorithm{SMS_EMOA} for schaffer_n2"=> [0.06964400287148177, 3.0140379940103594],
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
