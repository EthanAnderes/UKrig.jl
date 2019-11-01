using PyPlot
using LinearAlgebra
using Statistics
using Random
using NLopt
using UKrig

##
# data grid and interpolation grid
n₀ = 100 # grid evals
n₁ = 500  # obs points
x₁ = rand(n₁)
y₁ = rand(n₁)

sidex₀ = range(-0.1, 1.1, length=n₀)
sidey₀ = range(-0.1, 1.1, length=n₀)
x₀ = sidex₀ .+ 0 .* sidey₀'
y₀ = 0 .* sidex₀ .+ sidey₀'

##
# set the simulation truth parameters
σεₒ = 0.1
σzₒ = 1.0
ρₒ  = 0.2
νₒ  = 1.25
mₒ  = 2
βₒ  = 2 * (rand(mₒ) .- 1/2) 

##
# simulate data
dmat = UKrig.distmat((x₁,y₁), (x₁,y₁))
Σ₁ = (σzₒ^2) .* Mnu.(dmat ./ ρₒ, νₒ)
L₁ = cholesky(Σ₁).L
Z₁ = L₁ * randn(n₁)
Y₁ = Z₁ # .+ fpx.(1:mₒ,x₁')' * βₒ
data = Y₁ + σεₒ .* randn(n₁)





## ===============================
#

musr = 2
ν    = 1.25

loglike_inv_nu1, loglike_chol_nu1, wgrad_loglike_inv_nu1 = generate_Gnu_loglike(
	data, x₁, y₁;
	musr=musr,  
	ν=ν, # ≈ differentiability
)

loglike_inv_nu1(0.1, 2.0)
loglike_chol_nu1(0.1, 2.0)
wgrad_loglike_inv_nu1(0.1, 2.0)

ϵ = 1e-7
(loglike_inv_nu1(0.1+ϵ, 2.0) - loglike_inv_nu1(0.1-ϵ, 2.0))/(2ϵ)
(loglike_inv_nu1(0.1, 2.0+ϵ) - loglike_inv_nu1(0.1, 2.0-ϵ))/(2ϵ)

#LN_algm = [:LN_BOBYQA, :LN_COBYLA, :LN_PRAXIS, :LN_NELDERMEAD, :LN_SBPLX]
#LD_algm = [:LD_MMA, :LD_SLSQP, :LD_LBFGS, :LD_TNEWTON]
opt1 = Opt(:LN_BOBYQA, 2) # second arg is the number of variables to optimize
opt2 = Opt(:LN_BOBYQA, 2) # second arg is the number of variables to optimize
opt3 = Opt(:LD_LBFGS, 2) # second arg is the number of variables to optimize
opt1.max_objective = (x,grad) -> loglike_inv_nu1(x[1],x[2])
opt2.max_objective = (x,grad) -> loglike_chol_nu1(x[1],x[2])
opt3.max_objective = function (x,grad) 
	ll, grad1, grad2 = wgrad_loglike_inv_nu1(x[1],x[2])
	if length(grad)>0
		grad[1]=grad1
		grad[2]=grad2
	end
	return ll
end
opt1.lower_bounds = [0.0, 0.0]
opt2.lower_bounds = [0.0, 0.0]
opt3.lower_bounds = [0.0, 0.0]
opt1.maxtime = 10 # in seconds
opt2.maxtime = 10
opt3.maxtime = 10
# opt1.maxeval = 50
# opt2.maxeval = 50
# opt3.maxeval = 50
# opt1.ftol_abs = 0.1 * sqrt(length(data)) / sqrt(2)
# opt2.ftol_abs = 0.1 * sqrt(length(data)) / sqrt(2)
# opt3.ftol_abs = 0.1 * sqrt(length(data)) / sqrt(2)
optf1, optx1, ret1 = optimize(opt1, Float64[.1,2])
optf2, optx2, ret2 = optimize(opt2, Float64[.1,2])
optf3, optx3, ret3 = optimize(opt3, Float64[.1,2])

σe = optx1[1]
σg = optx1[2]

rng_σe = σe |> x->range(x - .05 * x, x + .05 * x, length=50)
rng_σg = σg |> x->range(x - .15 * x, x + .15 * x, length=50)
llmat = loglike_chol_nu1.(rng_σe,rng_σg')
matshow(llmat)


krig1 = generate_Gnu_krig(
	data, x₁, y₁;
	musr=musr,  
	ν=ν, 
	σe=σe, 
	σg=σg, 
)


figure()
scatter(x₁, y₁ ,c=data) 

figure()
pcolor(x₀, y₀, krig1.(x₀, y₀))