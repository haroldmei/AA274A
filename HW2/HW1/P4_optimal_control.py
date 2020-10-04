import numpy as np
import math
import scikits.bvp_solver
import matplotlib.pyplot as plt
from utils import *

dt = 0.005

def ode_fun(tau, z):
    """
    This function computes the dz given tau and z. It is used in the bvp solver.
    Inputs:
        tau: the independent variable. This must be the first argument.
        z: the state vector. The first three states are [x, y, th, ...]
    Output:
        dz: the state derivative vector. Returns a numpy array.
    """
    ########## Code starts here ##########
    om = -z[5]/2
    V = (-z[3]*math.cos(z[2]) - z[4]*math.sin(z[2]))/2

    # ODE Equations
    x_dot = V*math.cos(z[2])
    y_dot = V*math.sin(z[2])
    th_dot = om
    dHdth = z[3]*V*math.sin(z[2]) - z[4]*V*math.cos(z[2])
    p_dot = np.hstack((0,0,dHdth))
    r_dot = 0 #dummy state ode

    dz = z[6]*np.hstack((x_dot,y_dot,th_dot,p_dot,r_dot))
    
    ########## Code ends here ##########
    return dz


def bc_fun(za, zb):
    """
    This function computes boundary conditions. It is used in the bvp solver.
    Inputs:
        za: the state vector at the initial time
        zb: the state vector at the final time
    Output:
        bca: tuple of boundary conditions at initial time
        bcb: tuple of boundary conditions at final time
    """
    # final goal pose
    xf = 5
    yf = 5
    thf = -np.pi/2.0
    xf = [xf, yf, thf]
    # initial pose
    x0 = [0, 0, -np.pi/2.0]

    ########## Code starts here ##########
    lambda_factor = 0.3
    w = -zb[5]/2
    V = (-zb[3]*math.cos(zb[2]) - zb[4]*math.sin(zb[2]))/2
    
    bca = np.array([za[0]-x0[0], za[1]-x0[1], za[2]-x0[2]])

    H_f = lambda_factor + V**2 + w**2 + zb[3]*V*math.cos(zb[2]) + zb[4]*V*math.sin(zb[2]) + zb[5]*w 
    bcb = np.array([zb[0]-xf[0], zb[1]-xf[1], zb[2]-xf[2], H_f])
    ########## Code ends here ##########
    return (bca, bcb)

def solve_bvp(problem_inputs, initial_guess):
    """
    This function solves the bvp_problem.
    Inputs:
        problem_inputs: a dictionary of the arguments needs to define the problem
                        num_ODE, num_parameters, num_left_boundary_conditions,
                        boundary_points, function, boundary_conditions
        initial_guess: initial guess of the solution
    Output:
        z: a numpy array of the solution. It is of size [time, state_dim]

    Read this documentation -- https://pythonhosted.org/scikits.bvp_solver/tutorial.html
    """
    problem = scikits.bvp_solver.ProblemDefinition(**problem_inputs)
    soln = scikits.bvp_solver.solve(problem, solution_guess=initial_guess, trace = 0)

    # Test if time is reversed in bvp_solver solution
    flip, tf = check_flip(soln(0))
    t = np.arange(0,tf,dt)
    z = soln(t/tf)
    if flip:
        z[3:7,:] = -z[3:7,:]
    z = z.T # solution arranged so that it is [time, state_dim]
    return z

def compute_controls(z):
    """
    This function computes the controls V, om, given the state z. It is used in main().
    Input:
        z: z is the state vector for multiple time instances. It has size [time, state_dim]
    Outputs:
        V: velocity control input
        om: angular rate control input
    """
    ########## Code starts here ##########
    ''' why did I do this? there is no \dot x.
    V = np.sqrt(np.power(np.diff(z[:,0]), 2) + np.power(np.diff(z[:,1]), 2))
    om = np.diff(z[:,2])
    '''
    V = -0.5*(z[:,3]*np.cos(z[:,2]) + z[:,4]*np.sin(z[:,2]))
    om = -0.5*z[:,5]
    ########## Code ends here ##########

    return V, om

def main():
    """
    This function solves the specified bvp problem and returns the corresponding optimal contol sequence
    Outputs:
        V: optimal V control sequence 
        om: optimal om ccontrol sequence
    You are required to define the problem inputs, initial guess, and compute the controls

    Hint: The total time is between 15-25
    """
    ########## Code starts here ##########
    num_ODE = 7
    num_parameters = 0
    num_left_boundary_conditions = 3
    boundary_points = (0,1)
    function = ode_fun
    boundary_conditions = bc_fun

    #initial_guess = np.random.uniform(low=0, high=5, size=(7,))
    #initial_guess[2] = -np.pi #np.random.uniform(-np.pi, np.pi)
    #initial_guess[6] = 2000
    #print initial_guess

    #solution 1
    #initial_guess = (2.56992092e-01,  9.49860400e-01, -3.14159265e+00,  5.06923249e-01, 2.80611260e+00,  2.59356897e+00,  2.00000000e+03)

    #solution 2
    initial_guess = (3.24207804e+00,  3.33710509e+00, -3.14159265e+00,  1.86043668e+00, 8.46899596e-01,  2.05139181e+00,  2.00000000e+03)
    ########## Code ends here ##########

    problem_inputs = {
                      'num_ODE' : num_ODE,
                      'num_parameters' : num_parameters,
                      'num_left_boundary_conditions' : num_left_boundary_conditions,
                      'boundary_points' : boundary_points,
                      'function' : function,
                      'boundary_conditions' : boundary_conditions
                     }

    z = solve_bvp(problem_inputs, initial_guess)
    V, om = compute_controls(z)
    return z, V, om

if __name__ == '__main__':
    z, V, om = main()
    tf = z[0,-1]
    t = np.arange(0,tf,dt)
    x = z[:,0]
    y = z[:,1]
    th = z[:,2]
    data = {'z': z, 'V': V, 'om': om}
    save_dict(data, 'data/optimal_control.pkl')
    maybe_makedirs('plots')

    # plotting
    # plt.rc('font', weight='bold', size=16)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y,'k-',linewidth=2)
    plt.quiver(x[1:-1:200], y[1:-1:200],np.cos(th[1:-1:200]),np.sin(th[1:-1:200]))
    plt.grid(True)
    plt.plot(0,0,'go',markerfacecolor='green',markersize=15)
    plt.plot(5,5,'ro',markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis([-1, 6, -1, 6])
    plt.title('Optimal Control Trajectory')

    plt.subplot(1, 2, 2)
    plt.plot(t, V,linewidth=2)
    plt.plot(t, om,linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
    plt.title('Optimal control sequence')
    plt.tight_layout()
    plt.savefig('plots/optimal_control.png')
    plt.show()
