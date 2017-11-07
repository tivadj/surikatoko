import math
import numpy as np
import numpy.linalg as LA
import scipy.linalg

def IntPnt(pnt):
    if pnt is None: return None
    if isinstance(pnt, list):
        return [int(v) for v in pnt]
    if isinstance(pnt, tuple):
        return tuple(int(v) for v in pnt)
    if isinstance(pnt, np.ndarray):
        #return np.array([int(v) for v in pnt])
        # cv2.line require tuple, otherwise {SystemError}new style getargs format but argument is not a tuple
        return tuple([int(v) for v in pnt])
    assert False, "Unknown type: {}".format(type(pnt))

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

# Computes cubic root.
# x**(1./3.) works differently in Python2 and Python3, see http://stackoverflow.com/questions/28014241/how-to-find-cube-root-using-python
def rootCube(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)

def unzip(iterable):
    return zip(*iterable)

# Square of the vector norm.
def NormSqr(v):
    return sum(np.square(v))

# Finds the distance between two matrices as the ratio between their difference and max of absolute values.
# It is similar to relative tolerance ('rtol') in numpy.isclose.
def RelativeMatDistance(a, b):
    abs_a = np.abs(a)
    abs_b = np.abs(b)
    max_ab = np.maximum(abs_a, abs_b)
    diff_ab = np.abs(a-b)
    rel_ratio = np.divide(diff_ab, max_ab)
    result = np.min(rel_ratio)
    return result

# puts R[3x3] and T[3x1] into result[4x4]
def FillRT4x4(R, T, result = None):
    h,w = R.shape[0:2]
    assert w == 3
    assert h == 3
    assert len(T) == 3
    if not result is None:
        h,w = result.shape[0:2]
        assert w == 4
        assert h == 4
    else:
        result = np.zeros((4,4), R.dtype)

    result[:, :] = np.eye(4, 4)
    result[0:3, 0:3] = R[:,:]
    result[0:3, 3] = T[:]
    return result

def SE3Apply(rt_tup, x):
    """ Applies Special Euclidian SE3 transformation to a 3-element point. """
    r,t = rt_tup
    return np.dot(r, x) +t

def SE3Compose(rt1, rt2):
    """ Computes [r1|t1]*[r2|t2] """
    r1,t1 = rt1
    r2,t2 = rt2
    rnew = np.dot(r1, r2)
    tnew = np.dot(r1, t2) + t1
    return (rnew, tnew)

def SE3Inv(rt_tup):
    """ Inverts Special Euclidian SE3 transformation"""
    r, t = rt_tup
    return (r.T, -np.dot(r.T,t))

def RelMotionBFromA(a_from_world, b_from_world):
    return SE3Compose(b_from_world, SE3Inv(a_from_world))

def SE3AFromB(a_from_world, b_from_world):
    return SE3Compose(a_from_world, SE3Inv(b_from_world))

# v=(3x1)
def skewSymmeticMat(v):
    assert len(v) == 3, "Provide three components vector v=[x,y,z]"
    mat = np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))
    return mat

def skewSymmeticMatWithAdapter(v, elem_conv_fun, result):
    assert len(v) == 3, "Provide three components vector v=[x,y,z]"
    zero = elem_conv_fun(0)
    result[0, 0] = zero
    result[0, 1] = elem_conv_fun(-v[2])
    result[0, 2] = elem_conv_fun(v[1])

    result[1, 0] = elem_conv_fun(v[2])
    result[1, 1] = zero
    result[1, 2] = elem_conv_fun(-v[0])

    result[2, 0] = elem_conv_fun(-v[1])
    result[2, 1] = elem_conv_fun(v[0])
    result[2, 2] = zero

# Reverses the call to skewSymmeticMat
# sym_mat=[3x3]
def unskew(sym_mat):
    tol = 0.0001
    assert math.fabs(sym_mat[1, 2] + sym_mat[2, 1]) < tol
    assert math.fabs(sym_mat[0, 2] + sym_mat[2, 0]) < tol
    assert math.fabs(sym_mat[0, 1] + sym_mat[1, 0]) < tol
    return np.array([-sym_mat[1,2], sym_mat[0,2], -sym_mat[0,1]])


# Find skew symmetric matrix which is closest to the given matrix in Frobenius norm sense.
# fundMat [3x3]
def makeSkewSymmetric(fundMat):
    assert (3,3) == fundMat.shape
    # each component is an arithmetic mean of corresponding symmetric components
    a = (-fundMat[1,2] + fundMat[2,1])/2.0
    b = ( fundMat[0,2] - fundMat[2,0])/2.0
    c = (-fundMat[0,1] + fundMat[1,0])/2.0
    res = np.zeros((3,3))
    res[1,2] = -a
    res[2,1] = a
    res[0,2] = b
    res[2,0] = -b
    res[0,1] = -c
    res[1,0] = c
    return res

def IsSpecialOrthogonal(rot_mat, p_msg = None):
    def FromatMat(m): return str(m)
    tol = 1.0e-5
    c2 = np.allclose(np.eye(3,3), np.dot(rot_mat.T, rot_mat), atol=tol)
    if not c2:
        if not p_msg is None: p_msg[0] = "failed Rt.R=I (R={})".format(rot_mat)
        return False
    det_R = scipy.linalg.det(rot_mat)
    c1 = np.isclose(1, det_R, atol=tol)
    if not c1:
        if not p_msg is None: p_msg[0] = "failed det(R)=1 (R={}, detR={})".format(rot_mat, det_R)
        return False
    return True

# Creates the rotation matrix around the vector @n by angle @ang in radians.
# Using the Rodrigues formula.
def RotMatNew(n, ang, check_log_SO3 = True):
    if np.isclose(0, ang):
        return False, None
    assert np.isclose(1, LA.norm(n), atol=1e-3), "direction must be a unity vector"
    s = math.sin(ang)
    c = math.cos(ang)
    skew1 = skewSymmeticMat(n)
    skew2 = np.dot(skew1, skew1)
    R = np.eye(3,3) + s*skew1 + (1-c)*skew2
    p_err = [""]
    assert IsSpecialOrthogonal(R, p_err), p_err[0]
    if check_log_SO3 and False: # TODO: doesn't work for ang=>316deg (300deg around axis=1 or -60deg around axis-1?)
        n_new, ang_new = logSO3(R, check_rot_mat=False)
        # R may be decomposed both in (n,a) and (-n, -a)

        cond = np.isclose(ang,  ang_new) and np.allclose(n, n_new) or \
               np.isclose(ang, -ang_new) and np.allclose(n, -n_new) or \
               np.isclose(0, ang_new) and np.allclose(np.zeros(3), n_new) # ambiguous reconstruction


        n_new, ang_new = logSO3(R, check_rot_mat=False)
        assert cond, "initial angle of rotation doesn't persist the conversion"
    return True,R

def rotMat(n, ang, check_log_SO3 = True):
    suc, R = RotMatNew(n, ang, check_log_SO3)
    if suc:
        return R
    return np.identity(4, dtype=type(n[0]))

def RotMatFromAxisAngle(axis_angle, check_log_SO3 = True):
    ang = LA.norm(axis_angle)
    dir = axis_angle / ang
    suc, R = RotMatNew(dir, ang, check_log_SO3)
    if not suc:
        return False,None
    return True, R

# Fills 4x4 transformation matrix with R,T components.
# rot_mat[3x3]
# t_vec[3]
# result[4x4]
def SE3Mat(r = None, t = None, result = None, dtype=np.float32):
    rot_mat = r
    t_vec = t
    if result is None:
        elem_type = rot_mat.dtype if not rot_mat is None else t_vec.dtype if not t_vec.dtype is None else dtype
        result = np.zeros((4,4), elem_type)
    else:
        h,w = result.shape[0:2]
        assert h == 4 and w == 4, "Provide 4x4 matrix"
        result.fill(0)

    rsrc = rot_mat if not rot_mat is None else np.eye(3,3)
    result[0:3,0:3] = rsrc[0:3,0:3]

    if not t_vec is None:
        result[0:3,3] = t_vec[0:3]
    result[3,3] = 1
    return result

# Logarithm of SO(3): R[3x3]->(n,ang) where n=rotation vector, ang=angle in radians.
def LogSO3New(rot_mat, check_rot_mat = True):
    p_err = [""]
    assert IsSpecialOrthogonal(rot_mat, p_err), p_err[0]

    cos_ang = 0.5*(np.trace(rot_mat)-1)
    cos_ang = clamp(cos_ang, -1, 1) # the cosine may be slightly off due to rounding errors
    sin_ang = math.sqrt(1.0-cos_ang**2)

    # n=[0,0,0] ang=0 -> Identity[3x3]
    tol = 1.0e-3 # should treat as zero values: 2e-4
    if np.isclose(0, sin_ang, atol=tol):
        return False, None, None

    n = np.zeros(3)
    n[0] = rot_mat[2,1]-rot_mat[1,2]
    n[1] = rot_mat[0,2]-rot_mat[2,0]
    n[2] = rot_mat[1,0]-rot_mat[0,1]

    n = (0.5 / sin_ang) * n
    # direction vector is already close to unity, but due to rounding errors it diverges
    # TODO: check where the rounding error appears
    n_len = LA.norm(n)
    n = n / n_len

    if check_rot_mat:
        ang = math.acos(cos_ang)
        rot_mat_new = rotMat(n, ang, check_log_SO3=False)
        # atol=1e-4 to ignore 2.97e-3, 4.47974432e-04, 1.950215e-5 error
        assert np.allclose(rot_mat, rot_mat_new, atol=1e-2), "initial rotation matrix doesn't persist the conversion n={} ang={} rotmat|rotmat_new|delta=\n{}\n{}\n{}"\
            .format(n, ang, rot_mat, rot_mat_new, rot_mat-rot_mat_new)
    return True, n, ang

def logSO3(rot_mat, check_rot_mat = True):
    suc, n, ang = LogSO3New(rot_mat, check_rot_mat)
    if not suc: # TODO: this should not return arbitrary (zero) direction vector
        return (np.array([0, 0, 0], dtype=rot_mat.dtype), 0)
    return n, ang

def AxisAngleFromRotMat(R):
    axis_angle = None
    suc, n, ang = LogSO3New(R)
    if not suc:
        return False, axis_angle
    axis_angle = n * ang
    return True, axis_angle

def QuatFromRotationMat(R, check_back_QtoR=True):
    """ Converts from rotation matrix (SO3) to quaternion.
    :param R: [3x3] rotation matrix
    :return: quaternion corresponding to a given rotation matrix
    """
    p_err = [""]
    assert IsSpecialOrthogonal(R, p_err), p_err[0]

    # source: "A Recipe on the Parameterization of Rotation Matrices", Terzakis, 2012
    # formula 24
    quat = np.zeros(4, dtype=type(R[0,0]))
    if R[1, 1] > -R[2, 2] and R[0, 0] > -R[1, 1] and R[0, 0] > -R[2, 2]:
        sum = 1 + R[0, 0] + R[1, 1] + R[2, 2]
        assert sum >= 0
        root = math.sqrt(sum)
        quat[0] = 0.5 * root
        quat[1] = 0.5 * (R[2, 1] - R[1, 2]) / root
        quat[2] = 0.5 * (R[0, 2] - R[2, 0]) / root
        quat[3] = 0.5 * (R[1, 0] - R[0, 1]) / root
    elif R[1, 1] < -R[2, 2] and R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        sum = 1 + R[0, 0] - R[1, 1] - R[2, 2]
        assert sum >= 0
        root = math.sqrt(sum)
        quat[0] = 0.5 * (R[2, 1] - R[1, 2]) / root
        quat[1] = 0.5 * root
        quat[2] = 0.5 * (R[1, 0] + R[0, 1]) / root
        quat[3] = 0.5 * (R[2, 0] + R[0, 2]) / root
    elif R[1, 1] > R[2, 2] and R[0, 0] < R[1, 1] and R[0, 0] < -R[2, 2]:
        sum = 1 - R[0, 0] + R[1, 1] - R[2, 2]
        assert sum >= 0
        root = math.sqrt(sum)
        quat[0] = 0.5 * (R[0, 2] - R[2, 0]) / root
        quat[1] = 0.5 * (R[1, 0] + R[0, 1]) / root
        quat[2] = 0.5 * root
        quat[3] = 0.5 * (R[2, 1] + R[1, 2]) / root
    elif R[1, 1] < R[2, 2] and R[0, 0] < -R[1, 1] and R[0, 0] < -R[2, 2]:
        sum = 1 - R[0, 0] - R[1, 1] + R[2, 2]
        assert sum >= 0
        root = math.sqrt(sum)
        quat[0] = 0.5 * (R[1, 0] - R[0, 1]) / root
        quat[1] = 0.5 * (R[2, 0] + R[0, 2]) / root
        quat[2] = 0.5 * (R[2, 1] + R[1, 2]) / root
        quat[3] = 0.5 * root
    else: assert False
    if check_back_QtoR:
        R_back = RotMatFromQuat(quat, check_back_RtoQ=False)
        assert np.allclose(R, R_back, atol=1e-2), "failed R-q-R q={} R|R_new|delta=\n{}\n{}\n{}"\
            .format(quat, R, R_back, R_back-R)
    len = NormSqr(quat)
    assert np.isclose(1.0, len, atol=1e-2), "expected unity quaternion from rotation matrix but got len={}".format(len)
    return quat

def QuatFromAxisAngle(axis_ang):
    """ Converts from axis-angle representation of a rotation (SO3) to quaternion.
    :param axis_ang: 3-element vector of angle*rot_axis
    :return: quaternion corresponding to a given axis-angle
    """
    ang = LA.norm(axis_ang)
    quat = np.zeros(4, dtype=type(axis_ang[0]))
    quat[0] = math.cos(ang/2)
    sin_ang2 = math.sin(ang/2)
    quat[1] = sin_ang2 * axis_ang[0] / ang
    quat[2] = sin_ang2 * axis_ang[1] / ang
    quat[3] = sin_ang2 * axis_ang[2] / ang
    return quat

def AxisPlusAngleFromQuat(q):
    """ Converts from quaternion to (axis,angle)"""
    zero_ang = math.isclose(1.0, q[0])
    dir = np.zeros(3, dtype=type(q[0]))
    if zero_ang:
        ang = 0
    else:
        ang = 2*math.acos(q[0])
        sin_ang2 = math.sin(ang / 2)
        dir[0] = q[1] / sin_ang2
        dir[1] = q[2] / sin_ang2
        dir[2] = q[3] / sin_ang2
    return dir, ang

def RotMatFromQuat(q, check_back_RtoQ=True):
    """ Constructs rotation matrix (SO3) corresponding to given quaternion.
    :param q: quaternion, 4-element vector
    :return: rotation matrix, [3x3]
    """
    R = np.zeros((3,3), dtype=type(q[0]))

    # source: "A Recipe on the Parameterization of Rotation Matrices", Terzakis, 2012
    # formula 9
    R[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    R[0, 1] = 2*(q[1] * q[2] - q[0] * q[3])
    R[0, 2] = 2*(q[1] * q[3] + q[0] * q[2])

    R[1, 0] = 2*(q[1] * q[2] + q[0] * q[3])
    R[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
    R[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])

    R[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    R[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    R[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

    if check_back_RtoQ:
        q_back = QuatFromRotationMat(R, check_back_QtoR=False)
        assert np.allclose(q, q_back, atol=1e-2), "failed q-r-q conversion R=\n{}\nq={} q_back={} qdelta={}"\
            .format(R, q, q_back, q-q_back)

    return R