module UKrig

using SpecialFunctions: besselk, gamma
import DynamicPolynomials: @polyvar, monomials
using FixedPolynomials
using LinearAlgebra
using Statistics
using Random

export Mnu, Gnu, generate_Gnu_krig, generate_Mnu_krigY

tν𝒦t(t,ν) = t^ν * besselk(ν, t)

function Mnu(t, ν)::Float64
	pt, pν, p0, p1 = promote(t, ν, Float64(0), Float64(1)) # note besselk always returns a Float64 apparently
	return (pt==p0) ? p1 : tν𝒦t(√(2pν)*pt,pν) * 2^(1-pν) / gamma(pν)
end

# the const on the principle irregular term
scν(ν)       = - (2ν)^ν * gamma(ν + 1//2) * gamma(1-ν) / gamma(2ν+1) / sqrt(π)
scν(ν::Int)  = - 2 * (-2ν)^ν * gamma(ν + 1//2) / gamma(ν) / gamma(2ν+3) / sqrt(π)

function Gnu(t::T, ν::Int) where T<:Real
	if t==0
		return T(0)
	end
	return scν(ν) * t^(2ν) * log(t)
end

# Gnu(t::T, ν) where T<:Real = scν(ν) * t^(2ν)
function Gnu(t::T, ν) where T<:Real 
	return scν(ν) * t^(2ν)
end

ℓ2_dist(x,y) = norm(x - y) 

"""
`generate_Gnu_krig(;fdata, xdata, ν, σg, σε) -> (x::Array->krigY(x), x::Array->fp(x'), b, c)`
"""
function generate_Gnu_krig(fdata::Vector{T}, xdata::Vector{P}; ν, σg, σε) where {Q<:Real,T<:Real,P<:AbstractArray{Q}}
	m   = floor(Int, ν)
	n   = length(fdata)
	d   = length(xdata[1])
	@assert n == length(xdata)

	@polyvar x[1:d]
	monos = monomials(x,0:m)
	fp    = Polynomial{T}.(monos) |> System
	cnf   = JacobianConfig(fp, xdata[1])
	mp = length(fp)

	# f(x)  = mapslices(x;dims=[1]) do xᵢ
	# 	evaluate(f,xᵢ,cfg)
	# end
	function f(x::P)
		evaluate(fp,x,cnf)
	end

	n₁  = length(fdata)
	G₁₁ = (σg^2) .* Gnu.(ℓ2_dist.(xdata, permutedims(xdata)), ν)
	F₁₁ = reduce(hcat,f.(permutedims(xdata)))
	Ξ   = [
		G₁₁ .+ σε^2*I(n₁)  F₁₁'
		F₁₁                zeros(mp, mp)
	]
	cb = Ξ \ vcat(fdata, zeros(mp))
	c  = cb[1:length(fdata)]
	b  = cb[length(fdata)+1:end]

	function krig(x::P)
		K  = (σg^2) .* Gnu.(ℓ2_dist.(Ref(x), permutedims(xdata)), ν)
		Fᵀ = f(x)'
		return (K*c .+ Fᵀ*b)[1] 
	end
	function krig(x::Vector{P})
		K  = (σg^2) .* Gnu.(ℓ2_dist.(x, permutedims(xdata)), ν)
		Fᵀ = reduce(hcat,f.(permutedims(x)))'
		return K*c .+ Fᵀ*b 
	end

	return krig
end

"""
`generate_Mnu_krig(;fdata, xdata, ν, σg, σε) -> (x::Array->krigY(x), x::Array->fp(x'), b, c)`
"""
function generate_Mnu_krig(;fdata, xdata, ν, σ, ρ, σε)
	
	fpx_local(p,x) = x^(p-1)
	
	m   = floor(Int, ν)
	n₁  = length(fdata)

	M₁₁ = (σ^2) .* Mnu.(abs.(xdata .- xdata') ./ ρ, ν)
	Ξ   = [
		M₁₁ .+ σε^2*I(n₁)  fpx_local.(1:m, xdata')'
		fpx_local.(1:m, xdata')      zeros(m,m)
	]
	cb = Ξ \ vcat(fdata, zeros(m))
	c  = cb[1:length(fdata)]
	b  = cb[length(fdata)+1:end]

	function krigY(x)
		K  = (σ^2) .* Mnu.(abs.(x .- xdata') ./ ρ, ν)
		Fᵀ = fpx_local.(1:m, x')'
		return K*c .+ Fᵀ*b 
	end

	return krigY, x->fpx_local.(1:m,x'), b, c
end

# TODO: add ability to include other spatial covariate functions

end # module
