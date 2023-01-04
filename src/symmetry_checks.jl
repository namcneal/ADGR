function matrix_small_enough(A::Matrix{Float64}, tol::Float64)
    return LinearAlgebra.norm(A) / length(A) < tol
end

symmetrize(A::Matrix{Float64})  = (A.+A')/2.0
asymmetrize(A::Matrix{Float64}) = (A.-A')/2.0

issymmetric( A::Matrix{Float64}; tol::Float64=1e-6) = matrix_small_enough(asymmetrize(A), tol)
isasymmetric(A::Matrix{Float64}; tol::Float64=1e-6) = matrix_small_enough( symmetrize(A), tol)


function test_symmetry_checks()
    d = 4
    mat  = rand(d,d)
    sym  = rand(d,d); sym  .+= sym'
    asym = rand(d,d); asym .-= asym' 

    ## Testing the symmetric check
    @assert  issymmetric(sym)
    @assert !issymmetric(mat)
    @assert !issymmetric(asym)

    # Testing the asymmetric check
    @assert  isasymmetric(asym)
    @assert !isasymmetric(mat)
    @assert !isasymmetric(sym)
end