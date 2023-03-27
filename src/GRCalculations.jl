import Einsum
include("tests_and_checks.jl")

function check(arr::AbstractArray{T}, test_function::Function, point::AbstractArray{TT}) where {T,TT<:Real}
    test = deepcopy(arr)
        
    if contentsAreDuals(test) 
        test = getproperty.(test, Ref(:value))
    end

    test_function(test, point)
end

function metric_derivative(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    dim = length(point)

    ∂g  = ForwardDiff.jacobian(x->metric(x), point)
    ∂g  = reshape(∂g, (dim,dim,dim))

    if check_symmetry 
        check(∂g, test_metric_derivative_symmetry, point)
    end
    
    return ∂g
end

function christoffel(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    dim = length(point)
    
    # Inverse of the metric here
    g   = LinearAlgebra.inv(metric(point))

    # Derivative of the (non-inverse) metric
    ∂g = metric_derivative(metric, point)

    # Need to reshape on the forward due to how ForwardDiff computes the jacobian
    ∂g = reshape(∂g, (dim,dim,dim))

    Γ = zeros(T, size(∂g));
    for ρ=1:dim, μ=1:dim, ν=1:dim
        for σ=1:dim
            Γ[σ,μ,ν] += g[σ, ρ]/2 * (∂g[ν,ρ,μ] + ∂g[ρ,μ,ν] - ∂g[μ,ν,ρ])
        end
    end

    if check_symmetry 
        check(Γ, test_christoffel_symmetry, point)
    end

    return Γ
end

function christoffel_derivative(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    f = x->christoffel(metric, x)
    
    # Need to reshape due to how ForwardDiff computes the jacobian
    dim = length(point)
    ∂Γ = reshape(ForwardDiff.jacobian(f, point), (dim,dim,dim,dim))

    if check_symmetry 
        check(∂Γ, test_christoffel_jacobian_symmetry, point)
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

    Riem = zeros(T, size(∂Γ))
    for μ in 1:d for ν in 1:d for α in 1:d for β in 1:d
        Riem[ρ,σ,μ,ν] = ∂Γ[ρ,ν,σ,μ] - ∂Γ[ρ,μ,σ,ν]
        Riem[μ,α,ν,β]= ∂Γ[μ,β,α,ν] - ∂Γ[μ,α,ν,β]

        for λ=1:dim
            Riem[μ,α,ν,β] += Γ[μ,ν,λ]*Γ[λ,β,α] - Γ[μ,β,λ]*Γ[λ,ν,α]
        end 
    end end end end 

    g = metric(point)
    lowered_riemann = zeros(T, size(Riem))
    for μ in 1:d for ν in 1:d for α in 1:d for β in 1:d
        for λ in 1:d
            lowered_riem[μ,α,ν,β] := g[μ,λ] * Riem[λ,α,ν,β]
        end
    end end end end

    if check_symmetry 
        check(lowered_riem, test_riemannian_symmetry, point)
    end

    return lowered_riem
end

function ricci(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    dim   = length(point)
    g     = metric(point)
    g_inv = LinearAlgebra.inv(g)

    # Riemannian curvature tensor with first index lowered
    Riem  = riemannian(metric, point)

    # What will be the Ricci tensor
    Ric = zeros(Float64, (dim,dim))

    # ForwardDiff changes the type of the tensor to something like
    # ForwardDiff.Dual{ForwardDiff.Tag{var"#130#131", Float32}, Float64, 12}
    # if !(Riem[1,1,1,1] isa Float64)
    #     Ric = 0 .* Riem[:,:,1,1]
    # end

    for μ=1:dim, ν=1:dim
        for λ=1:dim, σ=1:dim
            Ric[μ,ν] += g_inv[λ,σ] * Riem[σ,μ,λ,ν]
        end
    end

      if check_symmetry 
        check(Ric, test_ricci_symmetry, point)
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