import scikits.bvp_solver
import numpy as np

T10 = 130
T2A = 100
T2Ahx = 70
Ahx = 5
U = 1.0

def function(a , T):
    q = (T[0] - T[1]) * U           # calculate the heat transfer from stream 1 to stream 2
    return np.array([-q ,        # evaluate dT1/dA
                        q/-2.0])    # evaluate dT2/dA

def boundary_conditions(Ta,Tb):
    return (np.array([Ta[0] - T10]),  #evaluate the difference between the temperature of the hot
                                         #stream on theleft and the required boundary condition
            np.array([Tb[1] - T2A]))  #evaluate the difference between the temperature of the cold
                                         #stream on the right and the required boundary condition

problem = scikits.bvp_solver.ProblemDefinition(num_ODE = 2,
                                      num_parameters = 0,
                                      num_left_boundary_conditions = 1,
                                      boundary_points = (0, Ahx),
                                      function = function,
                                      boundary_conditions = boundary_conditions)


solution = scikits.bvp_solver.solve(problem,
                            solution_guess = np.random.uniform(low=-10, high=10, size=(2,))) #((T10 + T2Ahx)/2.0, (T10 + T2Ahx)/2.0))

A = np.linspace(0,Ahx, 45)
T = solution(A)
print T

import pylab
pylab.plot(A, T[0,:],'-')
pylab.plot(A, T[1,:],'-')
pylab.show()