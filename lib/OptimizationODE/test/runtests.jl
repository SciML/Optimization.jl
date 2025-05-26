using Test
using Optimization
using Optimization.SciMLBase  # for NoAD
using OptimizationODE

# Helpers
make_zeros(u) = fill(zero(eltype(u)), length(u))
make_ones(u)  = fill(one(eltype(u)),  length(u))

# Quadratic and Rosenbrock + gradients
quad(u,p) = u[1]^2 + p[1]*u[2]^2
quad_grad!(g,u,p,data) = (g[1]=2u[1]; g[2]=2p[1]*u[2]; g)

rosen(u,p) = (p[1]-u[1])^2 + p[2]*(u[2]-u[1]^2)^2
rosen_grad!(g,u,p,data) = (
  g[1] = -2*(p[1]-u[1]) - 4*p[2]*u[1]*(u[2]-u[1]^2);
  g[2] = 2*p[2]*(u[2]-u[1]^2);
  g
)


@testset "OptimizationODE Solvers" begin

  # Setup Quadratic
  u0q, pq = [2.0,-3.0], [5.0]
  fq = OptimizationFunction(quad, SciMLBase.NoAD(); grad=quad_grad!)
  probQ = OptimizationProblem(fq, u0q, pq)

  @testset "ODEGradientDescent on Quadratic" begin
    sol = solve(probQ, ODEGradientDescent(); η=0.1, tmax=200.0, dt=0.05)
    @test isapprox(sol.u, make_zeros(sol.u); atol=1e-2)
    @test sol.retcode == ReturnCode.Success
  end

  @testset "ODEGradientDescent on Rosenbrock" begin
    u0r, pr = [-1.2,1.0], [1.0,100.0]
    fr = OptimizationFunction(rosen, SciMLBase.NoAD(); grad=rosen_grad!)
    probR = OptimizationProblem(fr, u0r, pr)
    solR = solve(probR, ODEGradientDescent(); η=5e-3, tmax=5000.0, dt=5e-3)
    @test isapprox(solR.u, make_ones(solR.u); atol=2e-2)
    @test solR.retcode == ReturnCode.Success
  end
    
  @testset "ODEGradientDescent on Rosenbrock (AutoForwardDiff)" begin
      using Optimization.ADTypes
      u0 = [0.0, 0.0]
      p = [1.0, 100.0]

      function rosen(u, p)
          return (p[1] - u[1])^2 + p[2]*(u[2] - u[1]^2)^2
      end
      rosen(u, p, data) = rosen(u, p)

      f_ad = OptimizationFunction(rosen, ADTypes.AutoForwardDiff())
      prob_ad = OptimizationProblem(f_ad, u0, p)

      sol = solve(prob_ad, ODEGradientDescent(); η = 0.005, tmax = 5000.0, dt = 0.01)

      @test isapprox(sol.u[1], 1.0; atol = 0.01)
      @test isapprox(sol.u[2], 1.0; atol = 0.01)
      @test sol.retcode == ReturnCode.Success
  end

  @testset "RKChebyshevDescent on Quadratic" begin
    solC = solve(probQ, RKChebyshevDescent();
                 use_hessian=true, η=0.1, s=8, maxiters=200)
    @test isapprox(solC.u, make_zeros(solC.u); atol=1e-2)
  end

  @testset "RKAccelerated on Quadratic" begin
    solA = solve(probQ, RKAccelerated(); η=0.5, p=2.0, s=4, maxiters=200)
    @test isapprox(solA.u, make_zeros(solA.u); atol=1e-2)
  end

  @testset "PRKChebyshevDescent on Quadratic" begin
    sol2 = solve(probQ, PRKChebyshevDescent();
                 use_hessian=true, η=0.5, s=6, maxiters=100, reestimate=20)
    @test isapprox(sol2.u, make_zeros(sol2.u); atol=1e-2)
  end

  @testset "PRKChebyshevDescent on Rosenbrock" begin
    u0R = [0.0, 0.0]
    pR  = [1.0, 100.0]
    fR  = OptimizationFunction(rosen, SciMLBase.NoAD(); grad=rosen_grad!)
    probR = OptimizationProblem(fR, u0R, pR)
    obj0 = rosen(u0R, pR)
    solR2 = solve(probR, PRKChebyshevDescent();
                  use_hessian=true, η=0.1, s=10, maxiters=500, reestimate=50)
    @test solR2.objective < 0.01 * obj0
  end
    
end
