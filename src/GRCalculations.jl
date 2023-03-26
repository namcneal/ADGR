import TensorOperations
include("tests_and_checks.jl")

function check(arr::AbstractArray{T}, test_function::Function, point::AbstractArray{T}) where {T<:Real}
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
    d = length(point)
    
    # Inverse of the metric here
    gin   = LinearAlgebra.inv(metric(point))

    # Derivative of the (non-inverse) metric
    ∂g = metric_derivative(metric, point)

    # Need to reshape on the forward due to how ForwardDiff computes the jacobian
    ∂g = reshape(∂g, (d,d,d))

    Γ = zeros(T, size(∂g))
    for (up,a,b,_sum) in Iterators.product(1:d, 1:d, 1:d, 1:d)
        Γ[up,a,b] += (1/2) * gin[up, _sum] * (∂g[_sum,a,b] + ∂g[_sum,b,a] - ∂g[a,b,_sum])
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
    d  = length(point)
    Γ  = reshape(Γ, (d,d,d))
    ∂Γ = reshape(∂Γ, (d,d,d,d))   
    
    TensorOperations.@tensor begin
        Riem[u,b,c,d] := ∂Γ[u,a,c,b] - ∂Γ[u,a,b,c] + Γ[s,a,c] * Γ[d,b,s] - Γ[t,a,b] * Γ[d,c,t]
    end

    # Riem = zeros(T, size(∂Γ))
    # for (up,a,b,c) in Iterators.product(1:d,1:d,1:d,1:d)
    #     Riem[up,a,b,c] = ∂Γ[up,a,c,b] - ∂Γ[up,a,b,c]

    #     for sum1 in 1:d
    #         Riem[up,a,b,c] += Γ[sum1,a,c] * Γ[d,b,sum1]
    #     end

    #     for sum2 in 1:d
    #         Riem[up,a,b,c] -= Γ[sum2,a,b] * Γ[d,c,sum2]
    #     end
    # end

    if check_symmetry 
        check(Riem, test_riemannian_symmetry, point)
    end

    return Riem
end

function ricci(metric::Function, point::AbstractArray{T}; check_symmetry::Bool=false) where T<:Real
    d   = length(point)
    g     = metric(point)
    g_inv = LinearAlgebra.inv(g)

    Riem  = riemannian(metric, point)
    
    TensorOperations.@tensor begin
        lower[a,b,c,d] := g[a,i] * Riem[i,b,c,d]
    end

    TensorOperations.@tensor begin
        Ric[u,v] := g_inv[a,b] * lower[a,u,b,v]
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
