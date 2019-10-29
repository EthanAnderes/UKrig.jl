using PyPlot
using LinearAlgebra
using Statistics
using Random
using UKrig

##
# data grid and interpolation grid
n₀ = 500 # grid evals
n₁ = 100  # obs points
x₀ = [[x] for x in range(-1.0, 2.0, length=n₀)]
x₁ = [[x] for x in sort(rand(n₁))]

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
Σ₁ = (σzₒ^2) .* Mnu.(UKrig.ℓ2_dist.(x₁,permutedims(x₁)) ./ ρₒ, νₒ)
L₁ = cholesky(Σ₁).L
Z₁ = L₁ * randn(n₁)
Y₁ = Z₁ # .+ fpx.(1:mₒ,x₁')' * βₒ
data = Y₁ + σεₒ .* randn(n₁)


##
# do the interpolation
krig1 = generate_Gnu_krig(
	data, x₁, 
	ν=νₒ+2, 
	σg=σzₒ/ρₒ^νₒ, 
	σε= σεₒ,
)


colx₀ = reduce(vcat, x₀)
colx₁ = reduce(vcat, x₁)

##
# plot the krig prediction
figure(figsize=(9,6)) 
plot(colx₁, data, ".", label="data")
plot(colx₀, krig1(x₀), label = "GkrigY2")
legend()
