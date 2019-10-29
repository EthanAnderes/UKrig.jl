module UKrig

using SpecialFunctions: besselk, gamma
import DynamicPolynomials: @polyvar, monomials, MonomialVector
using FixedPolynomials
using LinearAlgebra
using Statistics
using Random
using StaticArrays

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


function distmat(xcols::NTuple{d,Vx}, ycols::NTuple{d,Vy}) where {d,T<:Real,Vx<:AbstractVector{T}, Vy<:AbstractVector{T}} 
	nx = length(xcols[1])
	ny = length(ycols[1])
	D = fill(T(0), nx, ny)
	for (x,y) in zip(xcols,ycols)
		D .+= (x .- y').^2
	end
	return sqrt.(D)
end


################################################
#
# monomial closures
#
##############################################


function _construct_monos_Fp(m::Int, ::Val{d}) where {d}
    @polyvar x[1:d]
    monos   = monomials(x,0:m)
    poly_fp = System(Polynomial{Float64}.(monos))
    cnf_sv  = JacobianConfig(poly_fp, SVector{d}(Vector{Float64}(undef,d)))
    Fp(x::SVector{d,Q}) where {Q<:Real} = evaluate(poly_fp, x, cnf_sv)
    Fp(x::Real...) = Fp(SVector(x))
    Fp(x::NTuple{d,Q}) where {Q<:Real} = Fp(SVector(x))
    return monos, Fp
end


function _generate_fpb(monos::MonomialVector{true}, b::Vector, ::Val{d}) where {d}
    poly_fpb = Polynomial{Float64}.(dot(monos,b))
    cnf_sv = config(poly_fpb, SVector{d}(Vector{Float64}(undef,d)))
    fpb(x::SVector{d,Q}) where {Q<:Real} = evaluate(poly_fpb, x, cnf_sv)
    fpb(x::Real...) = fpb(SVector(x))
    fpb(x::NTuple{d,Q}) where {Q<:Real} = fpb(SVector(x))
    return fpb
end


#######################################################
#
# Kriging closures
#
######################################################

"""

```
krig = generate_Gnu_krig(fdata::Vector{T}, xdata::Vector{T}...; ν=3/2, σg=1.0, σε=0.0)
```

Here is a short summary how `krig` can be called and vectorized where `n` denotes the number of spatial 
interpolation points and `d` denotes the spatial dimension.

```
xvar = Tuple(rand(n) for i=1:d)
xs_in_rows = hcat(xvar...)
xzip   = zip(xvar...) 
xvtupe = xzip |> collect 
xsv    = [SVector(x) for x in zip(xvar...)] 
xad    = [reduce(vcat,x) for x in zip(xvar...)] 

xvar isa NTuple{d,Vector{Float64}}
xs_in_rows isa Matrix{Float64}
xzip isa Base.Iterators.Zip{NTuple{d,Vector{Float64}}}
xvtupe isa Vector{NTuple{d,Float64}}
xsv isa Vector{SVector{d,Float64}}
xad isa Vector{Vector{Float64}}
```

Evaluation at a single spatial point
``` 
krig(xsv[1])
krig(xad[1]...)
```

Evaluation at a collection of spatial points
```
krig.(xsv)
krig.(SVector{d}.(xad))
krig.(xvar...)
krig.(xzip)
krig.(xvtupe)
krig.(eachcol(xs_in_rows)...)
```

"""
function generate_Gnu_krig(fdata::Vector{T}, xdata::Vararg{Vector{T},d}; ν=3/2, σg=1.0, σε=0.0) where {d,T<:Real}
	m   = floor(Int, ν)
	n   = length(fdata)
	monos, Fp = _construct_monos_Fp(m, Val(d))
	mp = length(monos)
	
	dmat = distmat(xdata, xdata)
	G₁₁  = (σg^2) .* Gnu.(dmat, ν)
	
	FpVec = Fp.(xdata...)
	F₁₁   = reduce(hcat,permutedims(FpVec))

	Ξ   = [
		G₁₁ .+ σε^2*I(n)  F₁₁'
		F₁₁               zeros(mp, mp)
	]
	cb = Ξ \ vcat(fdata, zeros(mp))
	c  = cb[1:length(fdata)]
	b  = cb[length(fdata)+1:end]

	fpb = _generate_fpb(monos, b, Val(d))

	function krig(x::SVector{d,Q}) where Q<:Real
		sqdist = fill(Q(0),n)
		for i = 1:d
			sqdist .+= (x[i] .- xdata[i]).^2
		end
		Kvec = (σg^2) .* Gnu.(sqrt.(sqdist), ν)
		return dot(Kvec,c) + fpb(x)
	end
    krig(x::Real...) = krig(SVector(x))
    krig(x::NTuple{d,Q}) where {Q<:Real} = krig(SVector(x))

	return krig
end


function generate_Mnu_krig(fdata::Vector{T}, xdata::Vararg{Vector{T},d}; m=0, ν=3/2, σ=1.0, ρ=1.0, σε=0.0) where {d,T<:Real}
	n   = length(fdata)
	monos, Fp = _construct_monos_Fp(m, Val(d))
	mp = length(monos)
	
	dmat = distmat(xdata, xdata)
	M₁₁  = (σ^2) .* Mnu.(dmat ./ ρ, ν)
	
	FpVec = Fp.(xdata...)
	F₁₁   = reduce(hcat,permutedims(FpVec))

	Ξ   = [
		M₁₁ .+ σε^2*I(n)  F₁₁'
		F₁₁               zeros(mp, mp)
	]
	cb = Ξ \ vcat(fdata, zeros(mp))
	c  = cb[1:length(fdata)]
	b  = cb[length(fdata)+1:end]

	fpb = _generate_fpb(monos, b, Val(d))

	function krig(x::SVector{d,Q}) where Q<:Real
		sqdist = fill(Q(0),n)
		for i = 1:d
			sqdist .+= (x[i] .- xdata[i]).^2
		end
		Kvec = (σ^2) .* Mnu.(sqrt.(sqdist) ./ ρ, ν)
		return dot(Kvec,c) + fpb(x)
	end
    krig(x::Real...) = krig(SVector(x))
    krig(x::NTuple{d,Q}) where {Q<:Real} = krig(SVector(x))

	return krig
end

# TODO: add ability to include other spatial covariate functions

end # module
