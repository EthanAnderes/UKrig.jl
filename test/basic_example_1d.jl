using PyPlot
using LinearAlgebra
using Statistics
using Random
using NLopt
using UKrig

##
# data grid and interpolation grid
n₀ = 500 # grid evals
n₁ = 200  # obs points
x₁ = sort(rand(n₁)) 
x₀ = range(-0.1, 1.1, length=n₀) 
x₀₀ = range(0, 1, length=n₀) 

##
# set the simulation truth parameters
σεₒ = 0.05
σzₒ = 1.0
ρₒ  = 0.1
νₒ  = 1.25
mₒ  = 2
βₒ  = 2 * (rand(mₒ) .- 1/2) 

##
# simulate data
Σ₁ = (σzₒ^2) .* Mnu.(UKrig.ℓ2_dist.(x₁,permutedims(x₁)) ./ ρₒ, νₒ)
L₁ = cholesky(Σ₁).L
Z₁ = L₁ * randn(n₁)
Y₁ = Z₁ .+ βₒ[1] .+ x₁ .* βₒ[2]
data = Y₁ + σεₒ .* randn(n₁)



## ===============================
#

musr = 2
ν    = 1.25

loglike_inv_nu1, loglike_chol_nu1, wgrad_loglike_inv_nu1 = loglike_Gnu(
	data, x₁;
	musr=musr,  
	ν=ν, # ≈ differentiability
)

#=
loglike_inv_nu1(0.1, 2.0)
loglike_chol_nu1(0.1, 2.0)
wgrad_loglike_inv_nu1(0.1, 2.0)
ϵ = 1e-7
(loglike_inv_nu1(0.1+ϵ, 2.0) - loglike_inv_nu1(0.1-ϵ, 2.0))/(2ϵ)
(loglike_inv_nu1(0.1, 2.0+ϵ) - loglike_inv_nu1(0.1, 2.0-ϵ))/(2ϵ)
=#

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
xinit1 = [0.1, 2.0]
xinit2 = [0.1, 2.0]
xinit3 = [0.1, 2.0]
optf1, optx1, ret1 = optimize!(opt1, xinit1)
optf2, optx2, ret2 = optimize!(opt2, xinit2)
optf3, optx3, ret3 = optimize!(opt3, xinit3)
@show sqrt(length(data))/√2



σe = optx1[1]
σg = optx1[2]

rng_σe = σe |> x->range(x - .05 * x, x + .05 * x, length=50)
rng_σg = σg |> x->range(x - .15 * x, x + .15 * x, length=50)
llmat = loglike_chol_nu1.(rng_σe,rng_σg')
pcolor(rng_σe, rng_σg, llmat)

krig1 = krig_Gnu(
	data, x₁;
	musr=musr,  
	ν=ν, 
	σe=σe, 
	σg=σg, 
)

figure(figsize=(9,6)) 
plot(x₁, data, ".", label="data")
plot(x₀, krig1.(x₀))
legend()


## ===============================
# do the interpolation, vary signal/noise
figure(figsize=(9,6)) 
plot(x₁, data, ".", label="data")
for σg ∈ [1,5,10]
	krig1 = krig_Gnu(
		data, x₁;
		musr=2,  
		ν=1.0,    # ≈ differentiability
		σe=σe, # ≈ noise/signal ratio
		σg=σg,    # ≈ signal/noise ratio
	)
	plot(x₀, krig1.(x₀), label = "signal/noise = $σg")
end
legend()



# do the interpolation, vary smoothness
figure(figsize=(9,6)) 
plot(x₁, data, ".", label="data")
for ν ∈ [0.25, .5, 1.0, 1.25, 2.0]
	krig1 = krig_Gnu(
		data, x₁;
		musr=2, 
		ν=ν,   # ≈ differentiability
		σe=σe, # ≈ noise/signal ratio
		σg=σg, # ≈ signal/noise ratio
	)
	plot(x₀, krig1.(x₀), label = "nu = $ν")
end
legend()



# do the interpolation, interpolate
figure(figsize=(9,6)) 
plot(x₁, data, ".", label="data")
for ν ∈ [0.25, .5, 1.0]
	krig1 = krig_Gnu(
		data, x₁;
		musr=2,  
		ν=ν,    # ≈ differentiability
		σe=1e-7, # interpolate
		σg=σg, # ≈ signal/noise ratio
	)
	plot(x₀₀, krig1.(x₀₀), label = "nu = $ν")
end
legend()


