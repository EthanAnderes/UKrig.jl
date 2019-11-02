module UKrig

using SpecialFunctions: besselk, gamma
import DynamicPolynomials: @polyvar, monomials, MonomialVector
using FixedPolynomials
using LinearAlgebra
using Statistics
using Random
using StaticArrays

export Mnu, krig_Mnu, krig_Mnu_plus, loglike_Mnu, loglike_Mnu_plus 
export Gnu, krig_Gnu, krig_Gnu_plus, loglike_Gnu, loglike_Gnu_plus 



## ===================================================
# Matérn auto-cov 

tν𝒦t(t,ν) = t^ν * besselk(ν, t)

function Mnu(t, ν)::Float64
	pt, pν, p0, p1 = promote(t, ν, Float64(0), Float64(1)) # note besselk always returns a Float64 apparently
	return (pt==p0) ? p1 : tν𝒦t(√(2pν)*pt,pν) * 2^(1-pν) / gamma(pν)
end

# the const on Gnu which matches Matern's principle irregular term
function scν_Mnu(ν) 
	if floor(ν)==ν
		return (2 * (-1)^ν)   * (- (ν/2)^ν / gamma(ν) / gamma(ν+1))
	else
		return (π / sin(ν*π)) * (- (ν/2)^ν / gamma(ν) / gamma(ν+1))
	end
end


## ===================================================
# Generalized auto-cov 

scν(ν) = (-1)^(floor(Int,ν)+1)

function Gnu(t::A, ν::B) where {A<:Real, B<:Real}
	C = promote_type(A,B)
	if t==A(0)
		return C(0)
	end
	if floor(ν)==ν
		return C(scν(ν) * t^(2ν) * log(t))
	else 
		return C(scν(ν) * t^(2ν))
	end
end



## ===================================================
# Util

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


## ===================================================
# monomial closures

function _generate_fpb(monos::MonomialVector{true}, b::Vector, ::Val{d}) where {d}
    poly_fpb = Polynomial{Float64}.(dot(monos,b))
    cnf_sv = config(poly_fpb, SVector{d}(Vector{Float64}(undef,d)))
    fpb(x::SVector{d,Q}) where {Q<:Real} = evaluate(poly_fpb, x, cnf_sv)
    fpb(x::Real...) = fpb(SVector(x))
    fpb(x::NTuple{d,Q}) where {Q<:Real} = fpb(SVector(x))
    return fpb
end

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

function _construct_Fmat_monos(m::Int, xdata::Vararg{Vector{T},d}) where {d,T<:Real}
	n   = length(xdata[1])	
	monos, Fp = _construct_monos_Fp(m, Val(d))
	FpVec  = Fp.(xdata...)
	Fmat   = reduce(hcat,permutedims(FpVec))
	return Fmat, monos
end

function construct_constrasts(xdata::NTuple{d,Vector{T}}, mxdata::NTuple{ma,Vector{T}}; musr::Int=0, ν=0.5) where {d,ma,T<:Real}
	m = max(musr, floor(Int, ν))	
	
	## monomial covariates
	Fmat_pre, monos = _construct_Fmat_monos(m, xdata...)
	mp = length(monos)
	
	## additional user supplied covariates
	if ma > 0
		Fmat = vcat(Fmat_pre,transpose.(mxdata)...)
	else
		Fmat = Fmat_pre
	end
	
	return Fmat, monos
end



## ===================================================
# Kriging closures




# """

# ```
# krig = krig_Gnu(fdata::Vector{T}, xdata::Vector{T}...; musr=0, nu=0.5, σg=1.0, σe=1.0)
# ```

# Here is a short summary how `krig` can be called and vectorized where `n` denotes the number of spatial 
# interpolation points and `d` denotes the spatial dimension.

# ```
# xvar = Tuple(rand(n) for i=1:d)
# xs_in_rows = hcat(xvar...)
# xzip   = zip(xvar...) 
# xvtupe = xzip |> collect 
# xsv    = [SVector(x) for x in zip(xvar...)] 
# xad    = [reduce(vcat,x) for x in zip(xvar...)] 

# xvar isa NTuple{d,Vector{Float64}}
# xs_in_rows isa Matrix{Float64}
# xzip isa Base.Iterators.Zip{NTuple{d,Vector{Float64}}}
# xvtupe isa Vector{NTuple{d,Float64}}
# xsv isa Vector{SVector{d,Float64}}
# xad isa Vector{Vector{Float64}}
# ```

# Evaluation at a single spatial point
# ``` 
# krig(xsv[1])
# krig(xad[1]...)
# ```

# Evaluation at a collection of spatial points
# ```
# krig.(xsv)
# krig.(SVector{d}.(xad))
# krig.(xvar...)
# krig.(xzip)
# krig.(xvtupe)
# krig.(eachcol(xs_in_rows)...)
# ```

# """



function krig_Gnu(fdata::Vector{T}, xdata::Vararg{Vector{T},d}; musr::Int=0, σg=1.0, σe=1.0, ν=0.5) where {d,T<:Real}
	krig_pre, bma = krig_Gnu_plus(fdata, xdata, (); musr=musr, σg=σg, σe=σe, ν=ν)
	return krig_pre
end

function krig_Mnu(fdata::Vector{T}, xdata::Vararg{Vector{T},d}; musr::Int=0, σs=1.0, σe=1.0, ρ=1.0, ν=0.5) where {d,T<:Real}
	krig_pre, bma = krig_Mnu_plus(fdata, xdata, (); musr=musr, σg=σg, σe=σe, ρ=ρ, ν=ν)
	return krig_pre
end





function krig_Gnu_plus(fdata::Vector{T}, xdata::NTuple{d,Vector{T}}, mxdata::NTuple{ma,Vector{T}}; musr::Int=0, σg=1.0, σe=1.0, ν=0.5) where {d,ma,T<:Real}
		
	## spatial monomial covariates 
	Fmat, monos = construct_constrasts(xdata, mxdata; musr=musr, ν=ν)

	## generalized auto-cov (without the var mult σg²)
	dmat = distmat(xdata, xdata)
	Gmat  = Gnu.(dmat, ν)
	
	## Solve for Krigin coeffs
	n     = length(fdata)
	mtot  = size(Fmat,1)
	mmono = length(monos)
	Ξ   = [
		(σg^2).*Gmat .+ (σe^2).*I(n)  Fmat'
		Fmat               zeros(mtot, mtot)
	]
	cb = Ξ \ vcat(fdata, zeros(mtot))
	c  = cb[1:n]
	b  = cb[n+1:end]
	bmono = b[1:mmono]
	bma   = b[(mmono+ma):end]
	fpb = _generate_fpb(monos, bmono, Val(d))

	## Generate Krigin closure
	function krig_pre(x::SVector{d,Q}) where Q<:Real
		sqdist = fill(Q(0),n)
		for i = 1:d
			sqdist .+= (x[i] .- xdata[i]).^2
		end
		Kvec = (σg^2).*Gnu.(sqrt.(sqdist), ν)
		return dot(Kvec,c) + fpb(x)
	end
    krig_pre(x::Real...) = krig_pre(SVector(x))
    krig_pre(x::NTuple{d,Q}) where {Q<:Real} = krig_pre(SVector(x))

	return krig_pre, bma # bma is used 
end


# TODO: needs testing
function krig_Mnu_plus(fdata::Vector{T}, xdata::NTuple{d,Vector{T}}, mxdata::NTuple{ma,Vector{T}}; musr::Int=0, σg=1.0, σe=1.0, ρ=1.0, ν=0.5) where {d,ma,T<:Real}
		
	## spatial monomial covariates 
	Fmat, monos = construct_constrasts(xdata, mxdata; musr=musr, ν=ν)

	## generalized auto-cov (without the var mult σg²)
	dmat = distmat(xdata, xdata)
	Mmat  = Mnu.(dmat ./ ρ, ν)
	
	## Solve for Krigin coeffs
	n     = length(fdata)
	mtot  = size(Fmat,1)
	mmono = length(monos)
	Ξ   = [
		(σs^2).*Mmat .+ (σe^2).*I(n)  Fmat'
		Fmat               zeros(mtot, mtot)
	]
	cb = Ξ \ vcat(fdata, zeros(mtot))
	c  = cb[1:n]
	b  = cb[n+1:end]
	bmono = b[1:mmono]
	bma   = b[(mmono+ma):end]
	fpb = _generate_fpb(monos, bmono, Val(d))

	## Generate Krigin closure
	function krig_pre(x::SVector{d,Q}) where Q<:Real
		sqdist = fill(Q(0),n)
		for i = 1:d
			sqdist .+= (x[i] .- xdata[i]).^2
		end
		Kvec = (σs^2).*Mnu.(sqrt.(sqdist), ν)
		return dot(Kvec,c) + fpb(x)
	end
    krig_pre(x::Real...) = krig_pre(SVector(x))
    krig_pre(x::NTuple{d,Q}) where {Q<:Real} = krig_pre(SVector(x))

	return krig_pre, bma # bma is used 
end



## ===================================================
# REML closures for each fixed ν


function loglike_Gnu(fdata::Vector{T}, xdata::Vararg{Vector{T},d}; musr::Int=0, ν=0.5) where {d,T<:Real}
	return loglike_Gnu_plus(fdata, xdata, (); musr=musr, ν=ν)
end


function loglike_Mnu(fdata::Vector{T}, xdata::Vararg{Vector{T},d}; musr::Int=0, ν=0.5) where {d,T<:Real}
	return loglike_Mnu_plus(fdata, xdata, (); musr=musr, ν=ν)
end



function loglike_Gnu_plus(fdata::Vector{T}, xdata::NTuple{d,Vector{T}}, mxdata::NTuple{ma,Vector{T}}; musr::Int=0, ν=0.5) where {d,ma,T<:Real}

	Fmat, monos = construct_constrasts(xdata, mxdata; musr=musr, ν=ν)

	## generalized auto-cov (without the var mult σg²)
	dmat = distmat(xdata, xdata)
	Gmat  = Gnu.(dmat, ν)
    n   = length(fdata)
    Mᵀ = nullspace(Fmat)
    M  = transpose(Mᵀ)
	Mfdata = M * fdata

	function loglike_inv(σe, σg)
		Σ = M * ((σg^2).*Gmat .+ (σe^2).*I(n)) * Mᵀ  |> Symmetric
		Σ⁻¹Mfdata = Σ \ Mfdata
		ll = - (Mfdata⋅Σ⁻¹Mfdata) / 2 - logdet(Σ) / 2
		return ll
	end 

	function loglike_chol(σe, σg)
		Σ = M * ((σg^2).*Gmat .+ (σe^2).*I(n)) * Mᵀ  |> Symmetric
		cholΣ = cholesky(Σ, check=false)
		if !LinearAlgebra.issuccess(cholΣ)
			return -Inf
		end
		L = cholΣ.L
		L⁻¹Mfdata = L \ Mfdata
		ll = - (L⁻¹Mfdata⋅L⁻¹Mfdata) / 2 - sum(log,diag(L))
		return ll
	end 

	function wgrad_loglike_inv(σe, σg)
		uΣe  = M * Mᵀ  |> Symmetric
		uΣg  = M * Gmat * Mᵀ  |> Symmetric
		Σ    = (σg^2) * uΣg + (σe^2) * uΣe
		∂σeΣ = (2σe) * uΣe
		∂σgΣ = (2σg) * uΣg
		L = cholesky(Σ).L
		L⁻¹Mfdata = L \ Mfdata
		Σ⁻¹Mfdata = transpose(L) \ L⁻¹Mfdata
		ll    = - dot(L⁻¹Mfdata, L⁻¹Mfdata) / 2 - sum(log,diag(L))
		∂σell =  dot(Σ⁻¹Mfdata, ∂σeΣ * Σ⁻¹Mfdata) / 2 - tr(Σ \ ∂σeΣ) / 2
		∂σgll =  dot(Σ⁻¹Mfdata, ∂σgΣ * Σ⁻¹Mfdata) / 2 - tr(Σ \ ∂σgΣ) / 2
		return ll, ∂σell, ∂σgll
	end 

	return loglike_inv, loglike_chol, wgrad_loglike_inv
end


# TODO: needs testing
function loglike_Mnu_plus(fdata::Vector{T}, xdata::NTuple{d,Vector{T}}, mxdata::NTuple{ma,Vector{T}}; musr::Int=0, ν=0.5) where {d,ma,T<:Real}

	Fmat, monos = construct_constrasts(xdata, mxdata; musr=musr, ν=ν)

	dmat  = distmat(xdata, xdata)
    n  = length(fdata)
    Mᵀ = nullspace(Fmat)
    M  = transpose(Mᵀ)
	Mfdata = M * fdata

	function loglike_inv(σe, σs, ρ)
		Mmat  = Mnu.(dmat ./ ρ, ν)
		Σ = M * ((σs^2).*Mmat .+ (σe^2).*I(n)) * Mᵀ  |> Symmetric
		Σ⁻¹Mfdata = Σ \ Mfdata
		ll = - (Mfdata⋅Σ⁻¹Mfdata) / 2 - logdet(Σ) / 2
		return ll
	end 

	function loglike_chol(σe, σs, ρ)
		Mmat  = Mnu.(dmat ./ ρ, ν)
		Σ = M * ((σs^2).*Mmat .+ (σe^2).*I(n)) * Mᵀ  |> Symmetric
		cholΣ = cholesky(Σ, check=false)
		if !LinearAlgebra.issuccess(cholΣ)
			return -Inf
		end
		L = cholΣ.L
		L⁻¹Mfdata = L \ Mfdata
		ll = - (L⁻¹Mfdata⋅L⁻¹Mfdata) / 2 - sum(log,diag(L))
		return ll
	end 

	return loglike_inv, loglike_chol
end





end # module
