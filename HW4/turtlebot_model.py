import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    V, om = u
    x, y, th = xvec
    if abs(om) < EPSILON_OMEGA:
        x_t = x + V * np.cos(th) * dt
        y_t = y + V * np.sin(th) * dt
        th_t = th + om * dt
        g = np.array([x_t, y_t, th_t])

        Gx = np.array([[1, 0, -V * np.sin(th) * dt],
                       [0, 1, V * np.cos(th) * dt],
                       [0, 0, 1]])

        Gu = np.array([[np.cos(th) * dt, -V / 2 * (dt ** 2) * np.sin(th)],
                       [np.sin(th) * dt, V / 2 * (dt ** 2) * np.cos(th)],
                       [0, dt]])
    else:
        x_t = x + V / om * (np.sin(th + om * dt) - np.sin(th))
        y_t = y - V / om * (np.cos(th + om * dt) - np.cos(th))
        th_t = th + om * dt
        g = np.array([x_t, y_t, th_t])

        Gx = np.array([[1., 0., V / om * (np.cos(th + om * dt) - np.cos(th))],
                       [0., 1., V / om * (np.sin(th + om * dt) - np.sin(th))],
                       [0., 0., 1.]])

        dx_dom = V / (om ** 2) * (np.sin(th) - np.sin(th + om * dt)) + V * dt / om * np.cos(th + om * dt)
        dy_dom = V / (om ** 2) * (np.cos(th + om * dt) - np.cos(th)) + V * dt / om * np.sin(th + om * dt)
        Gu = np.array([[1. / om * (np.sin(th + om * dt) - np.sin(th)), dx_dom],
                       [-1. / om * (np.cos(th + om * dt) - np.cos(th)), dy_dom],
                       [0., dt]])

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    
    # apply two rotations
    R_c_b = np.array([[np.cos(tf_base_to_camera[2]), np.sin(tf_base_to_camera[2]), 0], \
                [-np.sin(tf_base_to_camera[2]),np.cos(tf_base_to_camera[2]),0],[0,0,1]])
    R_b_w = np.array([[np.cos(x[2]), np.sin(x[2]), 0],[-np.sin(x[2]),np.cos(x[2]),0],[0,0,1]])
    R_c_w = np.matmul(R_c_b, R_b_w)
    x_cam, y_cam, th_cam = np.dot(np.linalg.inv(R_c_w), tf_base_to_camera) + x

    alpha_in_cam = alpha - th_cam
    norm_cam = np.linalg.norm((x_cam, y_cam))
    r_in_cam = r - norm_cam * np.cos(alpha - np.arctan2(y_cam, x_cam))
    h = (alpha_in_cam, r_in_cam)

    # alpha_b = alpha - th_base  
    # r_b = r - norm(x_base,y_base)*cos(alpha - arctan(y_base,x_base)) ## the hint confuses me.
    drc_xc = -x_cam/norm_cam*np.cos(alpha - np.arctan2(y_cam, x_cam)) + norm_cam * np.sin(alpha-np.arctan2(y_cam, x_cam))*(y_cam/(x_cam*x_cam+y_cam*y_cam))
    drc_yc = -y_cam/norm_cam*np.cos(alpha - np.arctan2(y_cam, x_cam)) - norm_cam * np.sin(alpha-np.arctan2(y_cam, x_cam))*(x_cam/(x_cam*x_cam+y_cam*y_cam))
    dxc_xb = 1
    dxc_yb = 0
    dxc_tb = -np.sin(x[2])*tf_base_to_camera[0] - np.cos(x[2])*tf_base_to_camera[1]
    dyc_xb = 0
    dyc_yb = 1
    dyc_tb = np.cos(x[2])*tf_base_to_camera[0] - np.sin(x[2])*tf_base_to_camera[1]
    drc_xb = drc_xc * dxc_xb + drc_yc * dyc_xb
    drc_yb = drc_xc * dxc_yb + drc_yc * dyc_yb
    drc_tb = drc_xc * dxc_tb + drc_yc * dyc_tb
    Hx = np.array([[0, 0, -1], [drc_xb, drc_yb, drc_tb]])
    #print "tada", Hx
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
