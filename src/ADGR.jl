module ADGR

export metric_derivative, 
       christoffel, 
       christoffel_derivative, 
       riemannian,
       ricci,
       scalar

import Einsum
import ForwardDiff
import LinearAlgebra

include("GRCalculations.jl")

end;

