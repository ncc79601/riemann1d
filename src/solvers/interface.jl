"""
    compute_numerical_flux
Compute the numerical flux at the interface between two states using a given Riemann solver (exact or approximate) and equation of state.
"""
function compute_numerical_flux end


"""
    evolve!

Update the solution based on the computed numerical fluxes at its interfaces and the time step size.
"""
function evolve! end
