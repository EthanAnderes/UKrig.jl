using PyPlot
using LinearAlgebra
using Statistics
using Random
using UKrig

##
# data grid and interpolation grid
n₀ = 100 # grid evals
n₁ = 100  # obs points
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


##
# do the interpolation
krig1 = generate_Gnu_krig(
	data, x₁, y₁, 
	ν=νₒ, 
	# σg= σzₒ/ρₒ^νₒ, 
	σe= 0.01,
)

interp = krig1.(x₀, y₀)
interp |> matshow

