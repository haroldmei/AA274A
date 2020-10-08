import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        t_switch = self.traj_controller.traj_times[-1] - self.t_before_switch
        #print self.traj_controller.traj_times[-1]
        if t < t_switch:
            V, om = self.traj_controller.compute_control(x,y,th,t)
        else:
            V, om = self.pose_controller.compute_control(x,y,th,t)
        return V, om
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    from scipy.interpolate import splev, splrep
    sz = len(path)
    extended_ts = [0.0]
    ts = [0.0]
    for i in range(1, sz):
        l = np.linalg.norm(np.array(path[i-1]) - np.array(path[i])) * 1.2
        tm = l/V_des
        extended_ts = extended_ts + list(np.arange(extended_ts[-1], extended_ts[-1] + tm, dt))
        ts.append(ts[-1] + tm)

    sa = splrep(ts, np.array(path)[:,0])
    sb = splrep(ts, np.array(path)[:,1])

    traj_smoothed = np.zeros((len(extended_ts), 7))
    traj_smoothed[:,0] = splev(extended_ts, sa)
    traj_smoothed[:,1] = splev(extended_ts, sb)
    traj_smoothed[:,3] = splev(extended_ts, sa, der=1)
    traj_smoothed[:,4] = splev(extended_ts, sb, der=1)    
    traj_smoothed[:,5] = splev(extended_ts, sa, der=2)
    traj_smoothed[:,6] = splev(extended_ts, sb, der=2)

    traj_smoothed[:,2] = np.arccos(traj_smoothed[:,3]/(traj_smoothed[:,3]**2 + traj_smoothed[:,4]**2)**0.5) 

    t_smoothed = extended_ts
    #print traj_smoothed[:,2], V_des
    ########## Code ends here ##########
    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    
    s_f = State(x=traj[-1][0], y=traj[-1][1], V=0.0, th=0.0)
    t_new, V_scaled, om_scaled, traj_scaled =interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
