
import LinearAlgebra

include("symmetry_checks.jl")

"""
Test the calculation of the partial derivatives of the metric

""" 

function contentsAreDuals(arr::Array{T,N}) where {T, N} 
    return fieldnames(T) == (:value, :partials)
end


function test_metric_derivative_symmetry(∂g::Array{Float64}, point::Vector)
    if contentsAreDuals(∂g) return nothing end
    d = length(point)

    # Checking to make sure each component on the derivative index is symmetric
    for i in 1:d
        try 
            @assert issymmetric(∂g[:,:,i])
        catch AssertionError
            @show ∂g[:,:,i]
            error(".")
        end
    end
end

function test_christoffel_symmetry(Γ::Array{Float64}, point::Vector)
    if contentsAreDuals(Γ) return nothing end

    # Check the symmetry of the lower two indices for each upper
    d = length(point)
    for up in 1:d
        @assert issymmetric(Γ[up,:,:])
    end
end

function test_christoffel_jacobian_symmetry(∂Γ::Array{Float64}, point::Vector)
    if contentsAreDuals(∂Γ) return nothing end

    d = length(point)

    # Check the symmetry of the lower indices for each upper and each derivative
    for up in 1:d
        for der in 1:d
            @assert issymmetric(∂Γ[up,:,:,der])
        end
    end
end

function test_riemannian_symmetry(Riem::Array{Float64}, point::Vector)
    if contentsAreDuals(Riem) return nothing end

    d = length(point)

    for a in 1:d
        for b in 1:d
            @assert isasymmetric(Riem[a,b,:,:]) "Failed asymmetry on indices [$(a), $(b), :,:]"
        end
    end

    for a in 1:d
        for b in 1:d
            @assert isasymmetric(Riem[:,:,a,b]) "Failed asymmetry on indices [:,:,$(a), $(b)]"
        end
    end
end

function test_ricci_symmetry(Ric::Array{Float64}, point::Vector)
    if contentsAreDuals(Ric) return nothing end

    d = length(point)
#     @assert issymmetric(Ric) "Failed asymmetry on indices [$(a), $(b), :,:]"
end
##

""" 
Verify against analytic Schwartzschild solution

"""
function schwarzschild(point::Vector;
    c::Float64=1.0, G::Float64=1.0, M::Float64=0.5)

    rs = 2 * G * M / c^2

    return LinearAlgebra.diagm([-(1 - rs / point[2]), 
    1/(1 - rs / point[2]),
    point[2]^2,
    point[2]^2 * sin(point[3])^2
    ])
end

function analytic_metric_jacobian(point::Vector;
               c::Float64=1.0, 
               G::Float64=1.0, 
               M::Float64=0.5)
    rs = 2 * G * M / c^2
    d = length(point)
    ∂g = zeros(Float64, (d,d,d))

    r = point[2]
    θ = point[3]
    ∂g[1, 1, 2] = - rs / r^2
    ∂g[2, 2, 2] = - rs / (rs - r)^2
    ∂g[3, 3, 2] = 2r
    ∂g[4, 4, 2] = 2r  * sin(θ)^2

    ∂g[4, 4, 3] = r^2 * 2*sin(θ)*cos(θ)
    return ∂g
end

function analytic_christoffel(point::Vector;
           c::Float64=1.0, 
           G::Float64=1.0, 
           M::Float64=0.5)
    d = length(point)
    Γ = zeros(Float64, (d,d,d))

    # From Carroll p. 206. Going across each row:
    # Carroll sets c = 1, so I won't touch other cases for now
    @assert c ≈ 1

    r = point[2]
    θ = point[3]
    Γ[2,1,1] =  G*M / r^3 * (r - 2G*M)
    Γ[2,2,2] = -G*M / (r  * (r - 2G*M))
    Γ[1,1,2] =  G*M / (r  * (r - 2G*M))
    Γ[1,2,1] =  Γ[1,1,2]    # Symmetry


    Γ[3,2,3] = 1 / r
    Γ[3,3,2] = Γ[3,2,3]     # Symmetry
    Γ[2,3,3] = -(r - 2G*M)
    Γ[4,2,4] = 1 / r
    Γ[4,4,2] = Γ[4,2,4]     # Symmetry

    Γ[2,4,4] = -(r - 2G*M) * sin(θ)^2
    Γ[3,4,4] = -sin(θ)*cos(θ)
    Γ[4,3,4] = cot(θ)
    Γ[4,4,3] = Γ[4,3,4]     # Symmetry
    return Γ
end

function analytic_christoffel_jacobian(point::Vector; 
                    c::Float64=1.0, 
                    G::Float64=1.0, 
                    M::Float64=0.5)                         
    d  = length(point)
    ∂Γ = zeros(Float64, (d,d,d,d))

    # Carroll sets c = 1, and I don't want to fill in the missing c's down below
    # so I won't touch other cases for now
    @assert c ≈ 1

    # From Carroll p. 206. Going across each row, but differentiating
    # each term. Subdivided by row.
    # The last index is the derivative. The first three are the same 
    # as the Christoffel symbol, i.e. [top, left, right, ∂]
    r = point[2]
    θ = point[3]

    ∂Γ[2,1,1,2] = 2G*M * (3G*M - r) / r^4
    ∂Γ[2,2,2,2] = -2G*M * (G*M - r) / (r^2 * (r - 2G*M)^2)
    ∂Γ[1,1,2,2] =  2G*M * (G*M - r) / (r^2 * (r - 2G*M)^2)
    ∂Γ[1,2,1,2] =  ∂Γ[1,1,2,2] # Symmetry

    ∂Γ[3,2,3,2] = -1 / r^2
    ∂Γ[3,3,2,2] = ∂Γ[3,2,3,2]  # Symmetry
    ∂Γ[2,3,3,2] = -1
    ∂Γ[4,2,4,2] = -1 / r^2
    ∂Γ[4,4,2,2] = ∂Γ[4,2,4,2]  # Symmetry

    ∂Γ[2,4,4,2] = -1 * sin(θ)^2
    ∂Γ[2,4,4,3] = -(r - 2G*M) * 2 * sin(θ) * cos(θ)
    ∂Γ[3,4,4,3] = -cos(θ)*cos(θ) + sin(θ)*sin(θ)

    ∂Γ[4,3,4,3] = - csc(θ)^2
    ∂Γ[4,4,3,3] = ∂Γ[4,3,4,3] # Symmetry

    return ∂Γ
end

function test_schwarzschild_metric_derivative(point::Vector)
    analytic = analytic_metric_jacobian(point)
    computed = metric_derivative(schwarzschild, point)
    @assert sum((analytic .- computed).^2) < 1e-16
end

function test_schwarzschild_christoffel(point::Vector)
    analytic = analytic_christoffel(point)
    computed = christoffel(schwarzschild, point)

    @assert sum((analytic .- computed).^2) <  1e-16
end

function test_schwarzschild_christoffel_jacobian(point::Vector)
    analytic = analytic_christoffel_jacobian(point)
    computed = christoffel_derivative(schwarzschild, point)
    mse = (analytic .- computed).^2

    d = length(point)
    for i=1:d, j=1:d, k = 1:d, l=1:d
        @assert mse[i,j,k,l] < 1e-16 "Error at Christoffel index [$(i),$(j),$(k)] at derivative  $(l) ⟹  ∂Γ[$(i),$(j),$(k), $(l)]"
    end
end

function check_against_schwartzschild()
    r = 10.0
    t = 0.0
    θ = π / 2
    ϕ = 0
    point = [t,r,θ,ϕ]

    test_schwarzschild_christoffel(point)
    test_schwarzschild_metric_derivative(point)
    test_schwarzschild_christoffel_jacobian(point)

    tol = 1e-7
    @assert matrix_small_enough(ricci(schwarzschild,  point),  tol)
    @assert scalar(schwarzschild, point)  < tol
end

