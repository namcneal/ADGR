import Einsum
include("tests_and_checks.jl")

function metric_derivative(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    dim = length(point)

    ∂g  = ForwardDiff.jacobian(x->metric(x), point)
    ∂g  = reshape(∂g, (dim,dim,dim))

    if check_symmetry && !contentsAreDuals(∂g) 
        test_metric_derivative_symmetry(∂g, point)
    end
    
    return ∂g
end

function christoffel(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    dim = length(point)
    
    # Inverse of the metric here
    gin   = LinearAlgebra.inv(metric(point))

    # Derivative of the (non-inverse) metric
    ∂g = metric_derivative(metric, point)

    # Need to reshape on the forward due to how ForwardDiff computes the jacobian
    ∂g = reshape(∂g, (dim,dim,dim))

    Einsum.@einsum Γ[σ,μ,ν] := (gin[σ, ρ]/2 * (∂g[ν,ρ,μ] + ∂g[ρ,μ,ν] - ∂g[μ,ν,ρ]))

    if check_symmetry && !contentsAreDuals(Γ) 
        test_christoffel_symmetry(Γ, point)
    end

    return Γ
end

function christoffel_derivative(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    f = x->christoffel(metric, x)
    
    # Need to reshape due to how ForwardDiff computes the jacobian
    dim = length(point)
    ∂Γ = reshape(ForwardDiff.jacobian(f, point), (dim,dim,dim,dim))

    if check_symmetry && !contentsAreDuals(∂Γ) 
        test_christoffel_jacobian_symmetry(∂Γ, point)
    end

    return ∂Γ
end

function riemannian(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    Γ  = christoffel(metric, point)
    ∂Γ = christoffel_derivative(metric, point)

    # I'm assuming I'll have to reshape due to something about 
    # how the ForwardDiff tape 
    dim  = length(point)
    Γ  = reshape(Γ, (dim,dim,dim))
    ∂Γ = reshape(∂Γ, (dim,dim,dim,dim))        


    Einsum.@einsum Riem[ρ,σ,μ,ν] := (∂Γ[ρ,σ,ν,μ] - ∂Γ[ρ,σ,μ,ν] + Γ[ρ,μ,λ]*Γ[λ,ν,σ] - Γ[ρ,ν,λ]*Γ[λ,σ,μ])

    if check_symmetry && !(contentsAreDuals(Riem))
        test_riemannian_symmetry(Riem, point)
    end

    return Riem
end

function ricci(metric::Function, point::AbstractArray{T}) where T<:Real
    dim   = length(point)
    g     = metric(point)
    g_inv = LinearAlgebra.inv(g)

    # Riemannian curvature tensor with first index lowered
    Riem  = riemannian(metric, point)

    Einsum.@einsum lowered_Riem[μ,ν,α,β] := g[μ,λ] * Riem[λ,ν,α,β]

    Riem = lowered_Riem

    Einsum.@einsum Ric[μ,ν] := g_inv[λ,σ] * Riem[σ,μ,λ,ν]

    if !contentsAreDuals(Ric)
        test_ricci_symmetry(Ric, point)
    end

    return Ric
end

function scalar(metric::Function, point::AbstractArray{T}) where T<:Real
    dim   = length(point)
    g_inv = LinearAlgebra.inv(metric(point))
    Ric   = ricci(metric, point)

    S = 0
    for μ=1:dim, ν=1:dim
        S += g_inv[μ,ν] * Ric[μ,ν]
    end

    return S
end

function EFE_LHS(metric::Function, point::AbstractArray{T}) where T<:Real
    g = metric(point)
    R = ricci(metric,  point)
    S = scalar(metric, point)

    return @. R - S/2 * g
end

check_against_schwartzschild()
