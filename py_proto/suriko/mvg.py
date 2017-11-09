# MVGCV=Multiple View Geometry In Computer Vision, Hartley, 2003
# MASKS=An Invitation to 3-D visioning From Images to Geometric Volumes, Yi Ma, 2004
import math
import random
import time
import threading
import os
import argparse
import functools
import operator

import numpy as np
import numpy.linalg as LA
import cv2
#from math import *
import matplotlib.pyplot as plt

from OpenGL.raw.GLU.annotations import gluProject
from OpenGL.raw.GLX._types import struct___GLXcontextRec
from cv2 import Rodrigues
from numpy.f2py.rules import options
import scipy.linalg
from scipy.spatial import Delaunay
from scipy import optimize
import pylab
#import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
import time
# import pygame, pygame.image
# from pygame.locals import *

import suriko.sampson
import suriko.ess_5point_stewenius
import suriko.la_utils
import suriko.test_data_builder
from suriko.obs_geom import *
from suriko.bundle_adjustment_kanatani_impl import *

try:
    import mpmath # multi-precision math, see http://mpmath.org/
    mpmath.mp.dps = 100
except ImportError: pass

# debug = {0: no debugging, 1: errors, 2: warnings, 3: debug, 4: interactive}

def FixOpenCVCalcOpticalFlowPyrLK(prev_pts, next_pts, status, win_size):
    # cv2.calcOpticalFlowPyrLK returns points outside the window size borders
    # ignore them by setting the status=0
    radx = win_size[0] / 2
    rady = win_size[1] / 2
    for i, stat in enumerate(status):
        if stat == 0: continue
        p1 = prev_pts[i]
        p2 = next_pts[i]
        shift = p2-p1
        if abs(shift[0]) > radx or abs(shift[1]) > rady:
            status[i] = 0
            continue

# get transformation of image points so that the center of mass is in the origin and
# the averge distance from origin is sqrt(2)
# x_norm = T * x
def calcNormalizationTransform(xs, resultAvgDist):
    cent = np.mean(xs, axis=0)
    avgDist = LA.norm(xs - cent, axis=1).mean()
    assert avgDist > 0

    sc = resultAvgDist/avgDist

    # T = Scale(sc) * Translate(-centroid)
    T = np.eye(3)
    T[0,0] = T[1,1] = sc
    T[0,2] = -cent[0] * sc
    T[1,2] = -cent[1] * sc
    return T

def calcNormalizationTransformTest():
    # test centr=(100,300) avgDist=17.4
    ks=np.array([[100,310], [120, 290], [80, 300]])
    Ttest = calcNormalizationTransform(ks, expectAvgDist)
    Y=np.dot(Ttest, np.hstack((ks, np.ones((3,1)))).T).T
    cent=np.mean(Y, axis=0)
    assert np.isclose(0, cent[0])
    assert np.isclose(0, cent[1])
    avgDist = LA.norm(Y - cent, axis=1).mean()
    assert np.isclose(expectAvgDist, avgDist)

# Computes a homography - matrix which maps points between two images.
# manual impl of OpenCV.findHomography
# source: MVGCV page 89
# DLT = Direct Linear Transform
def findHomogDltCore(xs1, xs2):
    pointsCount = len(xs1)

    A = np.zeros((2*pointsCount, 9))
    for i in range(0,pointsCount):
        x1 = (xs1[i][0], xs1[i][1], 1)
        x2 = (xs2[i][0], xs2[i][1], 1)

        # 1st row
        A[i*2 + 0, 3] = -x2[2]*x1[0]
        A[i*2 + 0, 4] = -x2[2]*x1[1]
        A[i*2 + 0, 5] = -x2[2]*x1[2]

        A[i*2 + 0, 6] = x2[1]*x1[0]
        A[i*2 + 0, 7] = x2[1]*x1[1]
        A[i*2 + 0, 8] = x2[1]*x1[2]

        # 2nd row
        A[i*2 + 1, 0] = x2[2]*x1[0]
        A[i*2 + 1, 1] = x2[2]*x1[1]
        A[i*2 + 1, 2] = x2[2]*x1[2]

        A[i*2 + 1, 6] = -x2[0]*x1[0]
        A[i*2 + 1, 7] = -x2[0]*x1[1]
        A[i*2 + 1, 8] = -x2[0]*x1[2]

    # h is the last column of V (corresponding to the smallest singular value of A)
    dVec,u,vt = cv2.SVDecomp(A)
    hs = vt.T[:,-1]
    hs = hs / hs[-1]
    H = hs.reshape((3, 3))
    return H

def findHomogDlt(xs1, xs2):
    if not xs1.any():
        return None
    onex = xs1[0]
    assert len(onex) == 2 or (len(onex) == 3 and onex[2] == 1), "xs must be normalized"

    pointsCount = len(xs1)
    assert pointsCount >= 4

    # perform normalization
    expectAvgDist = math.sqrt(2)

    T1 = calcNormalizationTransform(xs1, expectAvgDist)
    T2 = calcNormalizationTransform(xs2, expectAvgDist)
    # NOTE: without normalization the reconstruction doesn't work; try the line below
    # T1 = T2 = np.eye(3)

    ones = np.ones((len(xs1),1))
    xs1Normed = np.dot(T1, np.hstack((xs1, ones)).T).T
    xs2Normed = np.dot(T2, np.hstack((xs2, ones)).T).T

    #
    HCore = findHomogDltCore(xs1Normed, xs2Normed)

    # denormalize F (source: MVGCV page 109)
    HResult = np.dot(LA.inv(T2), np.dot(HCore, T1))

    # OpenCV normalizes H so that the bottom right element = 1
    HResult = HResult / HResult[-1, -1]
    return HResult

# Finds homography from at least 4 points.
# MASKS page 134
def findHomogDltMasks(xs1, xs2):
    pointsCount = len(xs1)
    assert pointsCount >= 4
    dim = len(xs1[0])
    assert dim == 3, "Provide homogeneous 2D points"

    A = np.zeros((3*pointsCount, 9), dtype=type(xs1[0][0]))
    for i in range(0,pointsCount):
        x1 = xs1[i]
        x2 = xs2[i]

        x2_hat = skewSymmeticMat(x2)
        ai = np.kron(x1.reshape((3,1)), x2_hat) # [9,3]
        A[3*i:3*(i+1)] = ai.T

    # h is the last column of V (corresponding to the smallest singular value of A)
    dVec,u,vt = cv2.SVDecomp(A)
    hs = vt.T[:,-1]
    H = hs.reshape((3, 3), order="F")
    return H

def HomogErrSqrOneWay(H, x1, x2):
    x1_mapped_to2 = H.dot(x1)
    err = NormSqr(x1_mapped_to2 - x2)
    return err

def CalculateHomogError(H, xs1, xs2, point_pair_homog_err_fun):
    points_count = len(xs1)
    err1 = sum([point_pair_homog_err_fun(H, x1, x2) for x1, x2 in zip(xs1, xs2)]) / points_count
    return err1

def FundMatEpipolarError(fund_mat, xs1, xs2):
    """ Error, based on coplanarity (epipolar) constraint x2*F*x1==0 """
    points_count = len(xs1)
    err1 = sum([np.dot(x2, fund_mat).dot(x1) for x1, x2 in zip(xs1, xs2)]) / points_count
    return err1

# Decomposes planar homography into the (R,T/d,N) components. Returns the list of two possible solutions.
# d=distance to the plane from camera center.
# N=normal vector to the plane.
# source: MASKS page 139, algorithm 5.2
def ExtractRotTransFromPlanarHomography(H, xs1, xs2, debug):
    # 1. normalize H scale
    dVec,u,vt = cv2.SVDecomp(H)
    dVec = dVec.ravel()
    sig2 = dVec[1]
    Hnorm = H / sig2

    # 2. normalize H sign
    # H is correct except of sign
    # X2=H*X1 => lam2*x2=H*lam1*x1 and lam1>0 and lam2>0 => invert sign of H if left and right sides have different signs
    # or equivalent condition x2*H*x1>0
    # source: MASKS page 136, para 5.3.2
    if not xs1 is None and not xs2 is None:
        require_reverse = False
        for i in range(0, len(xs1)):
            val = xs2[i].dot(Hnorm).dot(xs1[i])
            if val < 0:
                require_reverse = True
                break
        if require_reverse:
            Hnorm = -Hnorm
        got_error = False
        for i in range(0, len(xs1)):
            val = xs2[i].dot(Hnorm).dot(xs1[i])
            if val < 0: # TODO: this happens, what to do?
                print("error: x2Hx1[{0}]={1}".format(i, val))
                got_error = True
        if got_error: # print entire list of x2*H*x1
            print("---")
            for i in range(0, len(xs1)):
                val = xs2[i].dot(Hnorm).dot(xs1[i])
                print("error: x2Hx1[{0}]={1}".format(i, val))
            assert False, "sign of H must be corrected"

    #
    HH = np.dot(Hnorm.T, Hnorm)
    dVec4,u4,vt4 = cv2.SVDecomp(HH)
    det1 = LA.det(u4)
    det2 = LA.det(vt4)
    if np.isclose(-1, det1) and np.isclose(-1, det2):
        u4 = -u4
        vt4 = -vt4
    assert LA.det(u4) > 0, "det(V)==1"
    assert LA.det(vt4) > 0

    sig_s1, sig_s2, sig_s3 = dVec4.ravel()

    # MASKS page 136, formula 5.44
    v1 = u4[:, 0]
    v2 = u4[:, 1]
    v3 = u4[:, 2]
    part1 = math.sqrt(1 - sig_s3) * v1
    part2 = math.sqrt(sig_s1 - 1) * v3
    den = math.sqrt(sig_s1 - sig_s3)
    u1 = (part1 + part2) / den
    u2 = (part1 - part2) / den

    v2_hat = skewSymmeticMat(v2)
    BigU1 = np.vstack((v2, u1, v2_hat.dot(u1))).T
    BigU2 = np.vstack((v2, u2, v2_hat.dot(u2))).T

    Hv2 = Hnorm.dot(v2)
    Hv2_hat = skewSymmeticMat(Hv2)
    BigW1 = np.vstack((Hv2, Hnorm.dot(u1), Hv2_hat.dot(Hnorm).dot(u1))).T
    BigW2 = np.vstack((Hv2, Hnorm.dot(u2), Hv2_hat.dot(Hnorm).dot(u2))).T

    cands_R = []
    cands_N = []
    cands_T_div_d = []
    if True:
        # solution 1
        candR = BigW1.dot(BigU1.T)
        candN = v2_hat.dot(u1)
        candT_div_d = (Hnorm - candR).dot(candN)
        cands_R.append(candR)
        cands_N.append(candN)
        cands_T_div_d.append(candT_div_d)
        # solution 2
        candR = BigW2.dot(BigU2.T)
        candN = v2_hat.dot(u2)
        candT_div_d = (Hnorm - candR).dot(candN)
        cands_R.append(candR)
        cands_N.append(candN)
        cands_T_div_d.append(candT_div_d)
        # solution 3
        cands_R.append(cands_R[0])
        cands_N.append(-cands_N[0])
        cands_T_div_d.append(-cands_T_div_d[0])
        # solution 4
        cands_R.append(cands_R[1])
        cands_N.append(-cands_N[1])
        cands_T_div_d.append(-cands_T_div_d[1])

    valid_cam_poses = []
    for i in range(0,4):
        candR = cands_R[i]
        candN = cands_N[i]
        candT_div_d = cands_T_div_d[i]
        cand_valid = candN[2] > 0

        # ignore R=I
        if cand_valid and np.allclose(np.eye(3,3), candR):
            cand_valid = False

        if cand_valid:
            valid_cam_poses.append((candR, candT_div_d, candN))
            if debug >= 93:
                n, ang = logSO3(candR)
                print("Valid ind:{0}".format(i))
                print("R n={0} ang={1}deg\n{2}".format(n, math.degrees(ang), candR))
                print("T/d={0}".format(candT_div_d))
                print("N={0}".format(candN))
    return valid_cam_poses


# Computes Sampson distance between projections of 3D points.
# It approximates reprojection error and doesn't require the computation of 3D coordinates.
class SampsonDistanceCalc:
    def __init__(self):
        e3 = [0, 0, 1]
        self.e3_hat = skewSymmeticMat(e3)

    # MASKS page 388, formula 11.12
    def Distance(self, fund_mat, x1, x2):
        assert len(x1) == 3
        assert len(x2) == 3
        h1 = np.dot(np.dot(x2, fund_mat), x1)
        den1 = np.dot(np.dot(self.e3_hat, fund_mat), x1)
        den2 = np.dot(x2, np.dot(fund_mat, self.e3_hat))
        dist = h1 * h1 / (LA.norm(den1) ** 2 + LA.norm(den2) ** 2)
        return dist

    def DistanceMult(self, fund_mat, xs1, xs2):
        result = 0.0
        for x1, x2 in zip(xs1, xs2):
            dis = self.Distance(fund_mat, x1, x2)
            result += dis
        result2 = sum(map(lambda x1,x2: self.Distance(fund_mat, x1, x2), xs1, xs2))
        return result

def IsEssentialMat(ess_mat, perr_msg = None, la_engine="scipy"):
    nrows, ncols = ess_mat.shape
    if not (nrows == 3 and ncols == 3):
        if not perr_msg is None: perr_msg[0] = "size(ess_mat)=3x3"
        return False

    if la_engine == "opencv":
        dVec,u,vt = cv2.SVDecomp(ess_mat)
    elif la_engine == "scipy":
        u,dVec,vt = scipy.linalg.svd(ess_mat)

    dVec = dVec.ravel()
    ok = np.isclose(dVec[0],dVec[1]) and np.isclose(0,dVec[2])
    if not ok:
        if not perr_msg is None: perr_msg[0] = "eigen_values(ess_mat)=[X X 0]"
        return False
    return True

# Projects given noisy essential matrix onto the essential space (such that rank(E)=2)
def ProjectOntoEssentialSpace(ess_mat_noisy, unity_translation = True, la_engine='scipy', conceal_lost_precision=True):
    eltype = ess_mat_noisy.dtype

    # project onto the space of essential matrices
    if la_engine == "opencv":
        dVec2, u2, vt2 = cv2.SVDecomp(ess_mat_noisy)
    elif la_engine == "scipy":
        u2, dVec2, vt2 = scipy.linalg.svd(ess_mat_noisy)
        if conceal_lost_precision:
            u2 = u2.astype(eltype)
            vt2 = vt2.astype(eltype)

    # MASKS page 120, we may normalize E to unity: norm(E)=norm(T)=1
    dVec2 = dVec2.ravel()
    sig = 0.5 * (dVec2[0] + dVec2[1])
    if unity_translation:
        sig = 1

    newSig = np.diag((sig, sig, 0))
    essMat = np.dot(np.dot(u2, newSig), vt2)
    return essMat

def FindEssentialMat_FillMatrixA(xs1, xs2, unit_2Dpoint, A):
    dim = len(xs1[0])
    assert dim == 3, "Must be a homogeneous image point (x,y,w)"

    pointsCount = len(xs1)
    x1 = np.zeros(3)
    x2 = np.zeros(3)
    for i in range(0, pointsCount):
        x1[:] = xs1[i]
        x2[:] = xs2[i]

        # normalize homog point
        if unit_2Dpoint:
            x1 = x1 / LA.norm(x1)
            x2 = x2 / LA.norm(x2)

        A[i, 0] = x1[0] * x2[0]
        A[i, 1] = x1[0] * x2[1]
        A[i, 2] = x1[0] * x2[2]
        A[i, 3] = x1[1] * x2[0]
        A[i, 4] = x1[1] * x2[1]
        A[i, 5] = x1[1] * x2[2]
        A[i, 6] = x1[2] * x2[0]
        A[i, 7] = x1[2] * x2[1]
        A[i, 8] = x1[2] * x2[2]

def FindEssentialMat_FillMatrixA_Way2(xs1, xs2, unit_2Dpoint, A):
    dim = len(xs1[0])
    assert dim == 3, "Must be a homogeneous image point (x,y,w)"

    pointsCount = len(xs1)
    x1 = np.zeros(3)
    x2 = np.zeros(3)
    for i in range(0, pointsCount):
        x1[:] = xs1[i]
        x2[:] = xs2[i]

        # normalize homog point
        if unit_2Dpoint:
            x1 = x1 / LA.norm(x1)
            x2 = x2 / LA.norm(x2)

        A[i, 0] = x1[0] * x2[0]
        A[i, 1] = x1[1] * x2[0]
        A[i, 2] = x1[2] * x2[0]
        A[i, 3] = x1[0] * x2[1]
        A[i, 4] = x1[1] * x2[1]
        A[i, 5] = x1[2] * x2[1]
        A[i, 6] = x1[0] * x2[2]
        A[i, 7] = x1[1] * x2[2]
        A[i, 8] = x1[2] * x2[2]


# Essential matrix can be recovered only up to scale.
# normalize, True to set the translation T=1
# perror={error}
# algorithm is in MASKS page 117
# NOTE: E has 6 degrees of freedom (3 for R + 3 for T), but at least 8 points is required
def FindEssentialMat8Point(xs1, xs2, unity_translation = True, plinear_sys_err = None, debug = 0, perr_msg = None):
    ok = len(xs1) >= 8
    if not ok:
        perr_msg[0] = "At least 8 points are required"
        return False, None

    dim = len(xs1[0])
    assert dim == 3,  "Must be a homogeneous image point (x,y,w)"

    pointsCount = len(xs1)
    A = np.zeros((pointsCount, 9), dtype=np.float)
    FindEssentialMat_FillMatrixA(xs1, xs2, True, A)

    # solve A*EssMat=0
    dVec, u, vt = cv2.SVDecomp(A)
    ess_stacked = vt.T[:, -1]

    essMat3 = ess_stacked.reshape((3, 3), order='F') # reshape columnwise
    assert essMat3.size == 9

    essMat = ProjectOntoEssentialSpace(essMat3, unity_translation)

    if not plinear_sys_err is None:
        plinear_sys_err[0] = LA.norm(np.dot(A, essMat.ravel(order='F')))

    return True, essMat

# Essential matrix can be recovered only up to scale.
# normalize, True to set the translation T=1
# perror={error}
# algorithm is in MASKS page 117
# NOTE: E has 6 degrees of freedom (3 for R + 3 for T), but at least 8 points is required
def FindEssentialMat7Point(xs1, xs2, unity_translation = True, debug = 0, perr_msg = None):
    ess_mat_list = None
    real_roots = None
    ok = len(xs1) >= 7
    if not ok:
        if not perr_msg is None: perr_msg[0] = "At least 7 points are required"
        return False,ess_mat_list,real_roots

    dim = len(xs1[0])
    assert dim == 3,  "Must be a homogeneous image point (x,y,w)"

    pointsCount = len(xs1)
    A = np.zeros((pointsCount, 9), dtype=np.float)
    FindEssentialMat_FillMatrixA(xs1, xs2, True, A)

    # solve A*EssMat=0
    dVec, u, vt = cv2.SVDecomp(A)

    # for 7 points, the noisy E is evaluated as E=E1+alpha*E2

    # find alpha, such that det(E1+alpha*E2)=0
    # then E = E1+alpha*E2, where E1,E2 = is 2-dimensional nullspace basis when 7 points are given
    # MASKS page 122
    ess1_stacked = vt.T[:, -1]
    ess2_stacked = vt.T[:, -2]
    e11, e21, e31, e12, e22, e32, e13, e23, e33 = ess1_stacked
    f11, f21, f31, f12, f22, f32, f13, f23, f33 = ess2_stacked

    # the polynom of degree 0 to 3, total of 4 terms
    # see Mathematica file ch5_det_E1_alpha_E2.nb
    poly = np.zeros(4) # from higher to lower degree, deg(poly[0])==3
    poly[0] = -(f13*f22*f31) + f12*f23*f31 + f13*f21*f32 - f11*f23*f32 - f12*f21*f33 + f11*f22*f33
    poly[1] = \
        -(e33*f12*f21) + e32*f13*f21 + e33*f11*f22 - e31*f13*f22 - e32*f11*f23 + \
        e31*f12*f23 + e23*f12*f31 - e22*f13*f31 - e13*f22*f31 + e12*f23*f31 - \
        e23*f11*f32 + e21*f13*f32 + e13*f21*f32 - e11*f23*f32 + e22*f11*f33 - \
        e21*f12*f33 - e12*f21*f33 + e11*f22*f33
    poly[2] = \
        -(e23*e32*f11) + e22*e33*f11 + e23*e31*f12 - e21*e33*f12 - e22*e31*f13 + \
        e21*e32*f13 + e13*e32*f21 - e12*e33*f21 - e13*e31*f22 + e11*e33*f22 + \
        e12*e31*f23 - e11*e32*f23 - e13*e22*f31 + e12*e23*f31 + e13*e21*f32 - \
        e11*e23*f32 - e12*e21*f33 + e11*e22*f33
    poly[3] = -(e13*e22*e31) + e12*e23*e31 + e13*e21*e32 - e11*e23*e32 - e12*e21*e33 + e11*e22*e33
    rr = np.roots(poly)

    real_roots = [c.real for c in rr if np.isclose(0, c.imag)]

    # there is typically one root close to zero and two roots with larger absolute value (like (-5,0,5)),
    # which lead to construction of EssMat with large Sampson error
    # so filter out roots with 'large' abs value
    # if len(real_roots) > 1:
    #     real_roots = [r for r in real_roots if abs(r) < 1.3]

    if len(real_roots) == 0:
        if not perr_msg is None: perr_msg[0] = "Provided points are in critical configuration (eg: all on the same plane)"
        return False, ess_mat_list, real_roots

    ess_mat_list = []
    for i1 in range(0,len(real_roots)):
        alpha = real_roots[i1]
        ess_stacked = ess1_stacked + alpha * ess2_stacked
        essMat3 = ess_stacked.reshape((3, 3), order='F')  # reshape columnwise
        ess_mat = ProjectOntoEssentialSpace(essMat3, unity_translation)
        ess_mat_list.append(ess_mat)

    return True, ess_mat_list, real_roots

def FindEssentialMat5PointStewenius(xs1, xs2, proj_ess_space, unity_translation = True, check_constr = True, debug = 0, expected_ess_mat = None, perr_msg = None, close_tol = 1e-5, la_engine='scipy', conceal_lost_precision=True):
    """
    Computes essential matrix from at least five 2D homogeneous (3 components) point correspondences.
    source: "Recent developments on direct relative orientation", Stewenius 2006
    param: direct refers to close form algorithm (it won't be trapped in local minimum).
    param: la_engine underlying lingear algebra routines to use ['scipy','opencv']
    param: conceal_lost_precision True to hide the fact of loss in float precision
    :return: the list of essential matrix candidates
    """
    ess_mat_list = None
    points_count = len(xs1)
    ok = points_count >= 5
    if not ok:
        if not perr_msg is None: perr_msg[0] = "Provide the minimal number of point correspondences"
        return False, ess_mat_list
    assert la_engine in ["scipy", "opencv"]

    eltype = type(xs1[0][0])

    A = np.zeros((points_count, 9), dtype=eltype)
    # TODO: do unit_2d_point?
    FindEssentialMat_FillMatrixA_Way2(xs1, xs2, False, A)

    # solve A*EssMat=0
    # NOTE: SVD must return full Vt [9x9] matrix
    # NOTE: for some reason cv2.SVDecomp doesn't produce correct last four columns of Vt.T (first five columns are ok)
    #dVec, u, vt = cv2.SVDecomp(A, flags=4) # 4=cv2.FULL_UV
    try:
        u, dVec, vt = scipy.linalg.svd(A, full_matrices=True)
    except LA.LinAlgError:
        return False, ess_mat_list # svd didn't converge
    if conceal_lost_precision:
        vt = vt.astype(eltype)

    assert vt.shape[0] == 9 and vt.shape[1] == 9, "Matrix of right singular vectors Vt must be full (9x9)"

    # E=x*E1+y*E2+z*E3+w*E4; w==1 so E4 should be built from eigenvector, corresponding to smallest eigenvalue (latest column)
    e1_stacked = vt.T[:,-4].reshape((3,3)).ravel(order='F')
    e2_stacked = vt.T[:,-3].reshape((3,3)).ravel(order='F')
    e3_stacked = vt.T[:,-2].reshape((3,3)).ravel(order='F')
    e4_stacked = vt.T[:,-1].reshape((3,3)).ravel(order='F') # w==1

    M = np.zeros((10, 20), dtype=eltype)
    suriko.ess_5point_stewenius.EssentialMat_Stewenius_FillM(e1_stacked, e2_stacked, e3_stacked, e4_stacked, M)

    # MX=[M1|M2]X=0 => [I|B]X=0
    # B is the Grobner basis
    grobner_basis_comput = 0 # 0=svd based,1=jordan-gauss elimination (maybe quicker, TODO: check)
    if grobner_basis_comput == 0:
        B = np.zeros((10,10), dtype=eltype)
        M1 = M[:,0:10]
        M2 = M[:,10:20]
        if la_engine == "opencv":
            suc,B = cv2.solve(M1, M2, dst=B, flags=cv2.DECOMP_SVD) # supports only f32 and f64
            assert suc, "Should succeed, because SVD is used and the solution in least squares sense"
        elif la_engine == "scipy":
            B = scipy.linalg.solve(M1, M2)
            if conceal_lost_precision:
                B = B.astype(eltype)
    elif grobner_basis_comput == 1:
        Mold = M.copy()
        suc = suriko.la_utils.GaussJordanElimination(M) # modify inplace
        B = M[:,10:20]
        # TODO: check why it may not succeed
        if not suc:
            assert suc

    fix_accord_to_matlab_impl = False
    if fix_accord_to_matlab_impl:
        Bold=B.copy()
        #new_ord = np.hstack((0, np.reshape(range(0,9), (3,3), order='F').ravel(order='C')+1))
        new_ord = np.array([1,2,4,7,3,5,8,6,9,10])-1
        B = B[new_ord,:]

    # action matrix
    At = np.zeros((10,10), dtype=eltype)
    At[0:6,:] = -B[[0,1,2,4,5,7],:]
    At[6,0] = 1 # x
    At[7,1] = 1 # y
    At[8,3] = 1 # z
    At[9,6] = 1 # free term

    if la_engine == "opencv":
        # NOTE: OpenCV's cv2.eigen works only with symmetric matrices
        # for C++ impl see the Eigen lib, https://stackoverflow.com/questions/30211091/calculating-the-eigenvector-from-a-complex-eigenvalue-in-opencv
        # for real matrices
        # use numpy.linalg because the processing of non-symmetric matrices is required
        w_At, v_At = LA.eig(At)
    elif la_engine == "scipy":
        w_At, v_At = scipy.linalg.eig(At)
        if conceal_lost_precision:
            v_At = v_At.astype(eltype)

    ess_mat_list = []
    # process left eigenvectors
    #sols = np.multiply(vt_At[6:9,:], 1/vt_At[9,:])
    #sols = np.multiply(u_At[:,6:9].T, 1/u_At[:,9].T) # left, hack
    #sols = np.multiply(u_At[6:9,:], 1/u_At[9,:]) # left
    for i_sol in range(0,10):
        w = v_At[9, i_sol]
        # complex scale is zero
        if np.isclose(0, w):
            continue

        # normalize x,y,z
        x, y, z = v_At[6:9,i_sol] / w
        if not np.isclose(0, x.imag) or not np.isclose(0, y.imag) or not np.isclose(0, z.imag):
            if debug >= 3: print("isol:{} skipped xyz:{}".format(i_sol, (x,y,z)))
            continue

        if not np.isclose(0, w.imag):
            assert False, "w is comlex, but x,y,z are reals; how is that can be"
        x,y,z = x.real, y.real, z.real

        ess_mat_stacked3 = x*e1_stacked+y*e2_stacked+z*e3_stacked+e4_stacked
        ess_mat3 = ess_mat_stacked3.reshape((3,3),order='F')

        remove_duplicates = True
        if remove_duplicates:
            is_dup = False
            for e in ess_mat_list:
                if np.allclose(e, ess_mat3, atol=close_tol): is_dup = True
                break
            if is_dup: continue

        # validate essential matrix
        if check_constr or debug >= 3:
            determ = scipy.linalg.det(ess_mat3)
            if not np.isclose(0, determ, atol=close_tol):
                if check_constr:
                    # assert False
                    if debug >= 3:
                        print("isol:{} skipping E:\n{}".format(i_sol, ess_mat3))
                        print("projectedE:\n{}".format(ProjectOntoEssentialSpace(ess_mat3, unity_translation, la_engine=la_engine)))
                else:
                    if debug >= 3: print("Failed det(E)==0, det={}".format(determ))

            zero_mat = 2*ess_mat3.dot(ess_mat3.T).dot(ess_mat3) - np.trace(ess_mat3.dot(ess_mat3.T))*ess_mat3
            zero_mat_norm = LA.norm(zero_mat)
            if not np.isclose(0, zero_mat_norm):
                if check_constr:
                    #assert False
                    if debug >= 3:
                        print("isol:{} skipping E:\n{}".format(i_sol, ess_mat3))
                        print("projectedE:\n{}".format(ProjectOntoEssentialSpace(ess_mat3, unity_translation)))
                    #continue
                else:
                    if debug >= 3: print("Failed 2E*Et*E-trace(E*Et)E==0 mat_norm={}".format(zero_mat_norm))

        # if we ignore an essential mat E which is outside the space of essentail matrices then we may end up without any hypothesis of E
        # so the best we can do is to project E on the space of essentail matrices
        # TODO: no projection should be made, because det(E)==0 and 2E*Et*E-trace(E*Et)E==0 must be satisfied
        #proj_ess_space = True
        if proj_ess_space:
            ess_mat = ProjectOntoEssentialSpace(ess_mat3, unity_translation, la_engine=la_engine, conceal_lost_precision=conceal_lost_precision)
        else:
            ess_mat = ess_mat3

        ess_mat = ess_mat / LA.norm(ess_mat)

        ess_mat_list.append(ess_mat)
        if debug >= 3:
            err_expect = None
            correct_ess_mat = expected_ess_mat
            if not correct_ess_mat is None:
                err2a=LA.norm(+ess_mat/LA.norm(ess_mat)-correct_ess_mat/LA.norm(correct_ess_mat))
                err2b=LA.norm(-ess_mat/LA.norm(ess_mat)-correct_ess_mat/LA.norm(correct_ess_mat))
                err_expect=min(err2a,err2b)

            err = FundMatEpipolarError(ess_mat, xs1, xs2)
            if proj_ess_space:
                ess_disp=ess_mat.ravel(order='F')
            else:
                ess_disp=ess_mat_stacked3/LA.norm(ess_mat_stacked3)*math.sqrt(2)

            # eigen values of unprojected ess mat
            if la_engine == "opencv":
                eig_vals = cv2.SVDecomp(ess_mat3)[0].ravel()
            elif la_engine == "scipy":
                eig_vals = scipy.linalg.svd(ess_mat3)[1]

            print("i={} E={} xyz={} err_epipol={} err_expect={} eig={} IsEss={}".format(i_sol, ess_disp,(x,y,z), err, err_expect, eig_vals, IsEssentialMat(ess_mat, la_engine=la_engine)))

    suc = len(ess_mat_list) > 0
    return suc, ess_mat_list

def ExtractRotTransFromEssentialMat(ess_mat, xs1_meter, xs2_meter, validate_ess_mat=True, svd_ess_mat=None, almost_zero = 1e-10, la_engine="scipy", debug=3):
    """ E->R,T
    param:xs1_meter is used to select the valid [R,T] camera's position - such that all points viewed in this camera have positive depth.
    param:almost_zero used to check if a value is almost zero
    """
    assert len(xs1_meter[0])==3 and len(xs2_meter[0])==3, "Provide homogeneous 2D points [x,y,w]"

    perr_msg = [""]
    if validate_ess_mat:
        assert IsEssentialMat(ess_mat, perr_msg), perr_msg[0]
    is_ess = IsEssentialMat(ess_mat, perr_msg), perr_msg[0]
    if debug >= 3: print("IsEssentialMat={} {}".format(is_ess, perr_msg[0]))

    eltype = ess_mat.dtype

    ess_mat_old = ess_mat
    xs1_meter_old = xs1_meter
    xs2_meter_old = xs2_meter

    # perform normalization
    do_norming = False
    if do_norming:
        expectAvgDist = math.sqrt(2)
        T1 = None
        T2 = None
        do_norming = True
        if do_norming:
            T1 = calcNormalizationTransform(xs1_meter, expectAvgDist)
            T2 = calcNormalizationTransform(xs2_meter, expectAvgDist)

            xs1_meter = np.dot(xs1_meter, T1.T)
            xs2_meter = np.dot(xs2_meter, T2.T)
        else:
            xs1_meter = xs1_meter
            xs2_meter = xs2_meter
        ess_mat = np.dot(np.dot(LA.inv(T2).T, ess_mat), LA.inv(T1))

    # do svd(ess_mat)
    p_err = [""]
    if svd_ess_mat is None:
        # NOTE: we need u and vt to be SO(3) (special orthogonal, so that det=+1)
        # TODO: why either +E or -E will always have u and vt in SO3? (see MASKS page 120)
        # only U and Vt components are required
        if la_engine == "opencv":
            dVec1, u1, vt1 = cv2.SVDecomp(ess_mat)
        elif la_engine == "scipy":
            u1, dVec1, vt1 = scipy.linalg.svd(ess_mat)
        #print("dvec1={0} u1={1} vt1={2}".format(dVec1, u1, vt1))
        #print("dvec2={0} u2={1} vt2={2}".format(dVec2, u2, vt2))

        det1 = LA.det(u1)
        det2 = LA.det(vt1)
        #if IsSpecialOrthogonal(u1, p_err):
        if det1 > 0 and det2 > 0:
            u = u1
            vt = vt1
        elif det1 < 0 and det2 < 0:
            u = -u1
            vt = -vt1
        elif det1 < 0:
            # dVec2, u2, vt2 = cv2.SVDecomp(-ess_mat)  # minus
            # assert IsSpecialOrthogonal(u2, p_err), p_err[0]
            # assert IsSpecialOrthogonal(vt2, p_err), p_err[0]
            u = -u1 # NOTE: (-E)=(-u)*Sig*vt
            vt = vt1
        else:
            assert det2 < 0
            # NOTE: (-E) = u*Sig*(-vt)
            u = u1
            vt = -vt1
            #assert False, "Can't choose sign of +E and -E"
    else:
        dVec1, u, vt = svd_ess_mat

    assert IsSpecialOrthogonal(u, p_err), p_err[0]
    assert IsSpecialOrthogonal(vt, p_err), p_err[0]

    bigSig = np.diag((1, 1, 0))
    #bigSig = np.diag(dVec1.ravel())

    # recover (R,T)
    cands_T = []
    cands_Tvec = []
    cands_R = []
    rz90p = rotMat([0, 0, 1],  math.pi / 2)
    rz90m = rotMat([0, 0, 1], -math.pi / 2)

    #
    calc_r0_fromU = True
    if calc_r0_fromU:
        r0_fromU = u.dot(rz90p.T).T
        if debug >= 3: print("r0_fromU={}".format(r0_fromU))

    # way4 Nister
    ra = np.dot(np.dot(u, rz90m), vt)
    rb = np.dot(np.dot(u, rz90m.T), vt)
    tu = u[:,2]
    pZ = np.hstack((np.eye(3,3),np.zeros((3,1))))
    pA = np.hstack((ra, tu.reshape((3,1))))
    hr = np.diag([1, 1, 1, -1])
    cands_per_ind_wayNister = np.array([0,0,0,0])
    cands_per_ind_wayNister_singular = np.array([0,0,0,0])

    for pnt_ind in range(0, len(xs1_meter)):
        x1 = xs1_meter[pnt_ind]
        x2 = xs2_meter[pnt_ind]
        q3D = triangulateDlt(pZ, pA, x1, x2, normalize=False)
        almost_zero_nister = 1e-4 # slightly bigger value to catch 1.059e-5
        if abs(q3D[-1]) < almost_zero_nister:
            # algorithm failed to determine the valid [R,T] candidate
            cands_per_ind_wayNister_singular += 1
            continue

        c1 = q3D[2]*q3D[3]
        q2D_A = np.dot(pA, q3D)
        c2 = q2D_A[2]*q3D[3]
        case = -1
        if c1 > 0 and c2 > 0:
            # pA and q3D are true configuration
            nister_sol_R = ra
            nister_sol_T = tu
            case = "A"
            cands_per_ind_wayNister[0] += 1
        elif c1 < 0 and c2 < 0:
            #pB = pA.dot(hr)
            #pB = hr.dot(pA) # seems to be correct
            q3D_new = np.dot(hr, q3D)
            nister_sol_R = ra # NOTE: should be -ra
            nister_sol_T = -tu
            case = "B"
            cands_per_ind_wayNister[1] += 1
        else:
            # different signs of c1 and c2
            ht = np.eye(4, dtype=eltype)
            ht[-1,-1] = -1
            ht[3,0:3] = -2*vt.T[0:3,2]
            pC2= np.dot(pA,ht)
            pC = np.hstack((rb, tu.reshape((3,1))))
            q3D_new = np.dot(ht, q3D)
            c_tmp = q3D[2]*q3D_new[3]
            if c_tmp > 0:
                # ok
                nister_sol_R = rb
                nister_sol_T = tu
                case = "C"
                cands_per_ind_wayNister[2] += 1
            else:
                pD = pC.dot(hr)
                q3D_new = np.dot(hr, q3D_new)
                nister_sol_R = rb
                nister_sol_T = -tu
                case = "D"
                cands_per_ind_wayNister[3] += 1

        n23a, ang23a = logSO3(nister_sol_R)
        if debug >= 3:  print("nister{}{} ang={}deg T={} Rf={}".format(case, pnt_ind, math.degrees(ang23a), nister_sol_T, nister_sol_R.ravel('F')))

        # MASKS page 120, Remark 5.10: construct all four combinations with Rz(+-90)
    combineSigns = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for sign_trans, sign_rot in combineSigns:
        rz_trans = rz90p if sign_trans == 1 else rz90m
        rz_rot   = rz90p if sign_rot   == 1 else rz90m

        candT = np.dot(np.dot(np.dot(u, rz_trans), bigSig), u.T)
        candTvec = unskew(candT)
        candR = np.dot(np.dot(u, rz_rot.T), vt)

        assert IsSpecialOrthogonal(candR, p_err), p_err[0]

        cands_T.append(candT)
        cands_Tvec.append(candTvec)
        cands_R.append(candR)

    # only one solution from four available has positive depth of points
    # find it
    sol_ind = None
    cands_per_ind_way1 = np.zeros(4, np.int32)
    cands_per_ind_way1_singular = np.zeros(4, np.int32)
    cands_per_ind_way2 = np.zeros(4, np.int32)
    cands_per_ind_way2_singular = np.zeros(4, np.int32)
    cands_per_ind_bundle = np.zeros(4, np.int32)
    for cand_ind in range(0, len(cands_T)):
        if debug >= 3: print("cand_ind={0}".format(cand_ind))
        candT = cands_T[cand_ind]
        candTvec = unskew(candT)
        candR = cands_R[cand_ind]

        # validate chosen sign of R,T
        # Way3
        # Compose aggregate matrix of all depths constraints for left and right frames
        # lams=(lam1, lam2, gamma) see MASKS p125, formula 5.20
        points_count = len(xs1_meter)
        M1 = np.zeros((points_count * 3, points_count + 1), np.float32)
        for pnt_ind in range(0, points_count):
            x1 = xs1_meter[pnt_ind]
            x2 = xs2_meter[pnt_ind]
            x2_hat = skewSymmeticMat(x2)

            col1 = x2_hat.dot(candR).dot(x1)
            M1[pnt_ind*3:(pnt_ind+1)*3, pnt_ind] = col1

            col2 = x2_hat.dot(candTvec)
            M1[pnt_ind * 3:(pnt_ind + 1) * 3, -1] = col2

        dVecM1, uM1, vtM1 = cv2.SVDecomp(M1)

        # depths in frame1 (lam1, lam2, ..., lamN, gamma), where gamma is a translation scale
        lams_frame_left = vtM1.T[:, -1]

        # it is possible for all lamdas to be negative - negate all
        # due to noise, the we may get the depthes with eg. universal_scale=0.004 instead of 0
        universal_scale = lams_frame_left[-1]

        almost_zero_bundle = almost_zero
        #almost_zero_bundle = 1e-2  # slightly bigger value to catch 4.28e-3
        if abs(universal_scale) < almost_zero_bundle:
            # algorithm fails to reconstruct depthes of all points
            # -1 may indicate failure better, but to keep results format similar to other methods, use 0
            pass_count1 = 0 # or -1
        else:
            # keep positive universal scale (due to SVD)
            if universal_scale < 0:
                lams_frame_left[:] = -lams_frame_left[:]

            pass_count1 = np.sum([1 for i in range(0, points_count) if lams_frame_left[i] > almost_zero])

        cands_per_ind_bundle[cand_ind] = pass_count1

        # Way1 validate chosen sign of R,T
        # Evaluate depth constraint for each pair of points from two images separately.
        pass_counter_way1 = 0
        pass_counter_way2 = 0
        singularA_counter = 0
        singularB_counter = 0
        for pnt_ind in range(0, len(xs1_meter)):
            x1 = xs1_meter[pnt_ind]
            x2 = xs2_meter[pnt_ind]
            #x1 = np.array([x1[0], x1[1], 1])
            #x2 = np.array([x2[0], x2[1], 1])

            # solve for lamdas: lam2 * x2 = lam1 * R*x1 + T
            # MASKS page 161, exercise 5.11
            err1 = -1
            lams2 = np.empty(0)
            col1 = np.dot(candR, x1)
            if la_engine == "opencv":
                A = np.hstack((np.reshape(col1, (3, 1)), -np.reshape(x2, (3, 1))))
                if LA.matrix_rank(A) < 2:
                    singularA_counter += 1
                else:
                    suc, lams2 = cv2.solve(A, -candTvec, flags=cv2.DECOMP_SVD)
                    assert suc

                    aaaleft = lams2[1] * x2
                    aaaright = lams2[0] * np.dot(candR, x1) + candTvec
                    err1 = LA.norm(aaaleft-aaaright)

                    positive_lams = lams2[0] > 0 and lams2[1] > 0
                    if positive_lams:
                        pass_counter_way1 += 1
            elif la_engine == "scipy":
                pass_counter_way1 = None
                cands_per_ind_way1 = None
                cands_per_ind_way1_singular = None

            # way2 put R*lam1*x1-lam2*x2+T=0 into rows of B and find B*[lam1,lam2,lam3]=0, lam3==1
            B = np.zeros((3,3), dtype=eltype)
            B[:,0] = col1
            B[:,1] = -x2
            B[:,2] = candTvec
            if la_engine == "opencv":
                dVec4, u4, vt4 = cv2.SVDecomp(B)
            elif la_engine == "scipy":
                u4, dVec4, vt4 = scipy.linalg.svd(B)
            lams3 = vt4.T[:,-1]

            # det(B) is always =0 hence the nullspace always has one solution
            # the solution is at infinity if lam3=0
            if abs(lams3[-1]) < almost_zero:
                singularB_counter += 1
            else:
                lams3 = lams3 / lams3[-1]

                positive_lams = lams3[0] > 0 and lams3[1] > 0
                if positive_lams:
                    pass_counter_way2 += 1

            if debug >= 3: print("{0}: lams={1} lams3={2} e={3:.4f} x1={4} x2={5}".format(pnt_ind, lams2.ravel(), lams3.ravel(), err1, x1, x2))

        if not cands_per_ind_way1 is None: # not implemented for float128
            cands_per_ind_way1[cand_ind] = pass_counter_way1
            cands_per_ind_way1_singular[cand_ind] = singularA_counter
        cands_per_ind_way2[cand_ind] = pass_counter_way2
        cands_per_ind_way2_singular[cand_ind] = singularB_counter

        if debug >= 3: print("cand_ind={0} pass_counter={1} pass_counter_way2={2}".format(cand_ind, pass_counter_way1, pass_counter_way2))

        pass_counter_way1or2 = pass_counter_way1
        if pass_counter_way1or2 is None:
            pass_counter_way1or2 = pass_counter_way2

        # assert pass_counter == 1, "Exactly one solution from 4 cases may have both positive lamdas (depths)"
        validRT_method1 = pass_counter_way1or2 == len(xs1_meter)
        if validRT_method1:
            assert sol_ind is None, "Exactly one solution from 4 cases may have both positive lamdas (depths)"
            sol_ind = cand_ind

    if debug >= 3:
        print("number of depth tests points={0}".format(len(xs1_meter)))
        print("cands_per_ind1={0} (3x2-3x1)".format(cands_per_ind_way1))
        print("cands_per_ind1_singular={0}".format(cands_per_ind_way1_singular))
        print("cands_per_ind2={0} (3x3)".format(cands_per_ind_way2))
        print("cands_per_ind2_singular={0}".format(cands_per_ind_way2_singular))
        print("cands_per_ind3={0} (bundle)".format(cands_per_ind_bundle))
        print("cands_per_ind4={0} (Nister)".format(cands_per_ind_wayNister))
        print("cands_per_ind4_singular={0}".format(cands_per_ind_wayNister_singular))

    if sol_ind is None:
        # May be, it is because:
        # points are in degenerate configuration
        # or essential matrix is computed on less than 8 points
        if debug >= 3: print("cands_R={0}".format(cands_R))
        if debug >= 3: print("cands_Tvec={0}".format(cands_Tvec))
        #assert False, "Exactly one solution from 4 cases may have both positive lamdas (depths)"
        return False, None, None

    sol_T = cands_T[sol_ind]
    sol_R = cands_R[sol_ind]
    sol_Tvec = unskew(sol_T)
    assert IsSpecialOrthogonal(sol_R, p_err), p_err[0]

    #
    specimen = cands_per_ind_way1
    if specimen is None: # not implemented for float128
        specimen = cands_per_ind_way2
    match_1 = np.allclose(specimen, cands_per_ind_way2)
    match_2 = np.allclose(specimen, cands_per_ind_bundle)
    match_3 = np.allclose(specimen, cands_per_ind_wayNister)
    if not match_1 or not match_2 or not match_3:
        if debug >= 3: print("can't match way3 cheirality")

    return True, sol_R, sol_Tvec

def FindRelativeMotion(pnt_life, xs1_meter, xs2_meter, debug, la_engine='scipy', conceal_lost_precision=True):
    # find transformation [R,T] of camera (frame2) in world (frame1)
    suc, ess_mat_list = FindEssentialMat5PointStewenius(xs1_meter, xs2_meter, proj_ess_space=True, check_constr=True, debug=debug, la_engine=la_engine, conceal_lost_precision=conceal_lost_precision)
    assert suc, "Essential matrix on consensus set must be calculated"

    # choose essential matrix with minimal sampson error
    sampson_dist_calculator = SampsonDistanceCalc()
    samps_errs = [sampson_dist_calculator.DistanceMult(e, xs1_meter, xs2_meter) for e in ess_mat_list]
    ess_mat_best_item = min(zip(ess_mat_list, samps_errs), key=lambda item: item[1])
    ess_mat = ess_mat_best_item[0]

    if debug >= 3: print("essMat=\n{}".format(ess_mat))

    refine_ess = False # refinement is not implemented
    if refine_ess:
        suc, ess_mat_refined = RefineFundMat(ess_mat, xs1_meter, xs2_meter, debug=debug)
        if suc:
            if debug >= 3: print("calcCameras refined_ess_mat=\n{0}".format(ess_mat_refined))
    if not (refine_ess and suc):
        ess_mat_refined = ess_mat

    #
    # TODO: what if R,T can't be extracted from ess_mat
    suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat_refined, xs1_meter, xs2_meter, debug=debug)
    if not suc:
        return False, (None, None)
    ess_T = skewSymmeticMat(ess_Tvec)
    ess_wvec, ess_wang = logSO3(ess_R)
    if debug >= 3: print("R|T: w={0} ang={1}\n{2}\n{3}".format(ess_wvec, math.degrees(ess_wang), ess_R, ess_Tvec))

    return suc, (ess_R, ess_Tvec)

def FindDistancesTo3DPoints(pnts_life, pnt_ids, frames_R, frames_T, frame_inds, block_base_frame_ind):
    alphas = np.zeros(pnt_ids)

    # find distances to all 3D points in frame1 given position [Ri,Ti] of each frame
    # (MASKS formula 8.44)
    for pnt_id in pnt_ids:
        pnt_ind = pnt_id
        x1 = pnts_life[pnt_ind][block_base_frame_ind]

        alpha_num = 0
        alpha_den = 0
        for frame_ind in frame_inds:
            # other cameras, except world frame1
            if frame_ind == block_base_frame_ind: continue

            x2 = pnts_life[pnt_ind][frame_ind]

            x2_skew = skewSymmeticMat(x2)
            frame_R = frames_R[frame_ind, :, :]
            frame_T = frames_T[frame_ind, :]
            h1 = np.dot(x2_skew, frame_T)
            h2 = np.dot(np.dot(x2_skew, frame_R), x1)
            alpha_num += np.dot(h1, h2)
            alpha_den += LA.norm(h1) ** 2
            if not math.isfinite(alpha_num) or not math.isfinite(alpha_den):
                print("error: nan")

        alpha = -alpha_num / alpha_den
        alphas[pnt_ind] = alpha
        dist = 1 / alpha

        # assert dist > 0, "distances are positive"
    if sum(1 for a in alphas if not math.isfinite(a)) > 0:
        print("error: nan")
    return 1/alphas

def FindDistanceToOne3DPoint(block_base_frame_ind, frame_inds, x_per_frame, frames_R, frames_T):
    # find distances to all 3D points in frame1 given position [Ri,Ti] of each frame
    # (MASKS formula 8.44)
    x1 = x_per_frame[block_base_frame_ind]

    alpha_num = 0
    alpha_den = 0
    for frame_ind in frame_inds:
        # other cameras, except world frame1
        if frame_ind == block_base_frame_ind: continue

        x2 = x_per_frame[frame_ind]

        x2_skew = skewSymmeticMat(x2)
        frame_R = frames_R[frame_ind]
        frame_T = frames_T[frame_ind]
        h1 = np.dot(x2_skew, frame_T)
        h2 = np.dot(np.dot(x2_skew, frame_R), x1)
        alpha_num += np.dot(h1, h2)
        alpha_den += LA.norm(h1) ** 2
        if not math.isfinite(alpha_num) or not math.isfinite(alpha_den) or math.isclose(0, alpha_den):
            print("error: nan")
            assert False

    if not math.isfinite(alpha_num) or not math.isfinite(alpha_den) or math.isclose(0, alpha_den):
        print("error: nan")
        assert False

    alpha = -alpha_num / alpha_den
    dist = 1 / alpha
    return dist

def FindDistancesTo3DPointsTwoFrames(pnts_life, pnt_ids, rel_RT, block_base_frame_ind, block_other_frame_ind, debug):
    points_num = len(pnt_ids)
    ess_R, ess_Tvec = rel_RT

    # find initial distances to all 3D points in frame1 (MASKS formula 8.43)
    alphas = None
    for i,pnt_id in enumerate(pnt_ids):
        pnt_ind = pnt_id
        x1 = pnts_life[pnt_ind].points_list_meter[block_base_frame_ind]
        x2 = pnts_life[pnt_ind].points_list_meter[block_other_frame_ind]
        if alphas is None:
            eltype = type(x1[0])
            alphas = np.zeros(points_num, dtype=eltype)

        x2_skew = skewSymmeticMat(x2)
        h1 = np.dot(x2_skew, ess_Tvec)
        h2 = np.dot(np.dot(x2_skew, ess_R), x1)
        alpha_num = np.dot(h1, h2)
        alpha = -alpha_num/LA.norm(h1)**2
        alphas[i] = alpha
        dist = 1/alpha
        assert dist > 0, "distances are positive"
    if debug >= 3: print("Dist: {0}".format(1 / alphas))
    return 1/alphas


def Estimate3DPointDepthFromFrames(base_frame_ind, frame_inds, x_per_frame, framei_from_base_RT):
    # find distances to all 3D points in frame1 given position [Ri,Ti] of each frame
    # (MASKS formula 8.44)
    x1 = x_per_frame[base_frame_ind]

    alpha_num = 0
    alpha_den = 0
    for frame_ind in frame_inds:
        # other cameras, except world frame1
        if frame_ind == base_frame_ind: continue

        x2 = x_per_frame[frame_ind]

        x2_skew = skewSymmeticMat(x2)
        frame_R, frame_T = framei_from_base_RT[frame_ind]
        h1 = np.dot(x2_skew, frame_T)
        h2 = np.dot(np.dot(x2_skew, frame_R), x1)
        alpha_num += np.dot(h1, h2)
        alpha_den += NormSqr(h1)
        if not math.isfinite(alpha_num) or not math.isfinite(alpha_den) or math.isclose(0, alpha_den):
            print("error: nan")
            assert False

    if not math.isfinite(alpha_num) or not math.isfinite(alpha_den) or math.isclose(0, alpha_den):
        print("error: nan")
        assert False

    alpha = -alpha_num / alpha_den
    dist = 1 / alpha
    return dist


# Do SLAM on calibrated coordinates (images of 3D points in meters).
# MASKS Algo 8.1, Factorization algorithm for multiple-view reconstruction, page 281
def calcCameras(xs_per_image_meter, min_reproj_err = 0.01, max_iter=20, debug = 3):
    frames_num = len(xs_per_image_meter)
    assert frames_num >= 2, "Algo works on multiple image frames"

    # find transformation [R,T] of camera (frame2) in world (frame1)
    xs1_meter = np.array(xs_per_image_meter[0])
    xs2_meter = np.array(xs_per_image_meter[1])

    # perror = [0.0]
    # suc, ess_mat = FindEssentialMat8Point(xs1_meter, xs2_meter, unity_translation=True, plinear_sys_err=perror)
    # assert suc

    suc, ess_mat_list = FindEssentialMat5PointStewenius(xs1_meter, xs2_meter, proj_ess_space=True, check_constr=True, debug=debug)
    assert suc, "Essential matrix on consensus set must be calculated"

    # choose essential matrix with minimal sampson error
    sampson_dist_calculator = SampsonDistanceCalc()
    samps_errs = [sampson_dist_calculator.DistanceMult(e, xs1_meter, xs2_meter) for e in ess_mat_list]
    ess_mat_best_item = min(zip(ess_mat_list, samps_errs), key=lambda item: item[1])
    ess_mat = ess_mat_best_item[0]

    if debug >= 3: print("essMat=\n{}".format(ess_mat))

    refine_ess = False # refinement is not implemented
    if refine_ess:
        suc, ess_mat_refined = RefineFundMat(ess_mat, xs1_meter, xs2_meter, debug=debug)
        if suc:
            if debug >= 3: print("calcCameras refined_ess_mat=\n{0}".format(ess_mat_refined))
    if not (refine_ess and suc):
        ess_mat_refined = ess_mat

    #
    # TODO: what if R,T can't be extracted from ess_mat
    suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat_refined, xs1_meter, xs2_meter, debug=debug)
    if not suc:
        return False, None, None, None
    ess_T = skewSymmeticMat(ess_Tvec)
    ess_wvec, ess_wang = logSO3(ess_R)
    if debug >= 3: print("R|T: w={0} ang={1}\n{2}\n{3}".format(ess_wvec, math.degrees(ess_wang), ess_R, ess_Tvec))

    # find initial distances to all 3D points in frame1 (MASKS formula 8.43)
    points_num = len(xs1_meter)
    alphas = np.zeros(points_num, dtype=np.float32)
    for pnt_ind in range(0, points_num):
        x1 = xs_per_image_meter[0][pnt_ind]
        x2 = xs_per_image_meter[1][pnt_ind]

        x2_skew = skewSymmeticMat(x2)
        h1 = np.dot(x2_skew, ess_Tvec)
        h2 = np.dot(np.dot(x2_skew, ess_R), x1)
        alpha_num = np.dot(h1, h2)
        alpha = -alpha_num/LA.norm(h1)**2
        alphas[pnt_ind] = alpha
        dist = 1/alpha
        assert dist > 0, "distances are positive"
    if debug >= 3: print("Dist: {0}".format(1 / alphas))

    # the first view frame is the world frame
    # frame's [Ri,Ti], Ri is the world axes as viewed from i-s frame
    # Ti is the translation from camera to world frames
    iter = 1
    reproj_err = 0
    frames_R = np.zeros((frames_num,3,3), dtype=np.float32)
    frames_T = np.zeros((frames_num,3), dtype=np.float32)
    world_pnts =  np.zeros((points_num,3), dtype=np.float32)
    reproj_err_history = []
    while iter < max_iter:
        if debug >= 3: print("iter={0}".format(iter))

        # estimage camera position [R,T] given distances to all 3D points pj in frame1

        A = np.zeros((3*points_num, 12), dtype=np.float32)
        xs_frame1 = xs_per_image_meter[0]
        for frame_ind in range(0, frames_num):
            points_per_frame = xs_per_image_meter[frame_ind]

            A.fill(0)
            for pnt_ind in range(0, points_num):
                x1 = xs_frame1[pnt_ind]

                x_img =  points_per_frame[pnt_ind]
                x_img_skew = skewSymmeticMat(x_img)
                block_left = np.kron(x1.reshape(1,3), x_img_skew)
                A[3 * pnt_ind:3 * (pnt_ind + 1), 0:9] = block_left
                A[3 * pnt_ind:3 * (pnt_ind + 1), 9:12] = alphas[pnt_ind]*x_img_skew

            dVec1, u1, vt1 = cv2.SVDecomp(A)
            r_and_t = vt1.T[:,-1]
            cam_R_noisy = r_and_t[0:9].reshape(3,3, order='F') # unstack
            cam_Tvec_noisy = r_and_t[9:12]

            # project noisy [R,T] onto SO(3) (see MASKS, formula 8.41 and 8.42)
            dVec2, u2, vt2 = cv2.SVDecomp(cam_R_noisy)
            no_guts = np.dot(u2, vt2)
            sign1 = np.sign(LA.det(no_guts))
            frame_R = sign1 * no_guts

            t_factor = sign1 / rootCube(LA.det(np.diag(dVec2.ravel())))
            cam_Tvec = t_factor * cam_Tvec_noisy

            frames_R[frame_ind,:,:] = frame_R[:,:]
            frames_T[frame_ind,:] = cam_Tvec[:]

        if debug >= 3:
            for i in range(0, frames_num):
                rotn, ang = logSO3(frames_R[i,:,:])
                print("cam:{0} T:{1} R:n={2} ang={3}\n{4}".format(i, frames_T[i,:], rotn, math.degrees(ang), frames_R[i,:,:]))

        # find distances to all 3D points in frame1 given position [Ri,Ti] of each frame
        # (MASKS formula 8.44)
        for pnt_ind in range(0, points_num):
            x1 = xs_per_image_meter[0][pnt_ind]

            alpha_num = 0
            alpha_den = 0
            for frame_ind in range(1, frames_num): # other cameras, except world frame1
                x2 = xs_per_image_meter[frame_ind][pnt_ind]

                x2_skew = skewSymmeticMat(x2)
                frame_R = frames_R[frame_ind, :, :]
                frame_T = frames_T[frame_ind, :]
                h1 = np.dot(x2_skew, frame_T)
                h2 = np.dot(np.dot(x2_skew, frame_R), x1)
                alpha_num += np.dot(h1, h2)
                alpha_den += LA.norm(h1) ** 2

            alpha = -alpha_num / alpha_den
            alphas[pnt_ind] = alpha
            dist = 1 / alpha
            #assert dist > 0, "distances are positive"

        if debug: print("it: {0} dist: {1}".format(iter, 1/alphas))

        # calculate reprojection error
        reproj_err = 0
        for pnt_ind in range(0, points_num):
            dist = 1 / alphas[pnt_ind] # distance to 3D point in frame1 (world)
            x1 = xs_per_image_meter[0][pnt_ind]
            x3D_cam1 = x1 * dist
            world_pnts[pnt_ind, :] = x3D_cam1 # updated coordinate of the world point

            for frame_ind in range(1, frames_num):
                x_expect = xs_per_image_meter[frame_ind][pnt_ind]

                frame_R = frames_R[frame_ind,:,:]
                frame_T = frames_T[frame_ind,:]
                xact_cami = np.dot(frame_R, x3D_cam1) + frame_T
                xact_camiN = xact_cami / xact_cami[-1]
                err = LA.norm(x_expect / x_expect[-1] - xact_camiN)**2
                if debug >= 3: print("xexpect={0} xact={1} err={2} meters".format(x_expect, xact_camiN, err))
                reproj_err += err

        if debug >= 3: print("Xs3D:\n{0}".format(world_pnts))
        if debug >= 3: print("reproj_err={0} meters".format(reproj_err))

        reproj_err_history.append(reproj_err)

        if reproj_err < min_reproj_err:
            break
        iter += 1
    if debug >= 3: print("Search succeeded in {0} iterations, reproj_err: {1}".format(iter, reproj_err))
    if debug >= 3: print("reproj_err_history={0}".format(reproj_err_history))

    if iter >= max_iter:
        return False, frames_R, frames_T, world_pnts # is not converged

    return True, frames_R, frames_T, world_pnts

def calcCamerasNew(xs_per_image_meter, block_base_frame_ind, block_other_frame_ind, rel_RT, min_reproj_err = 0.01, max_iter=20, debug = 3):
    frames_num = len(xs_per_image_meter)
    assert frames_num >= 2, "Algo works on multiple image frames"

    ess_R, ess_Tvec = rel_RT

    # find initial distances to all 3D points in frame1 (MASKS formula 8.43)
    points_num = len(xs_per_image_meter[block_base_frame_ind])
    alphas = np.zeros(points_num, dtype=np.float32)
    for pnt_ind in range(0, points_num):
        x1 = xs_per_image_meter[block_base_frame_ind][pnt_ind]
        x2 = xs_per_image_meter[block_other_frame_ind][pnt_ind]

        x2_skew = skewSymmeticMat(x2)
        h1 = np.dot(x2_skew, ess_Tvec)
        h2 = np.dot(np.dot(x2_skew, ess_R), x1)
        alpha_num = np.dot(h1, h2)
        alpha = -alpha_num/LA.norm(h1)**2
        alphas[pnt_ind] = alpha
        dist = 1/alpha
        assert dist > 0, "distances are positive"
    if debug >= 3: print("Dist: {0}".format(1 / alphas))

    if sum(1 for a in alphas if not math.isfinite(a)) > 0:
        print("error: nan")

    # the first view frame is the world frame
    # frame's [Ri,Ti], Ri is the world axes as viewed from i-s frame
    # Ti is the translation from camera to world frames
    iter = 1
    diverge = False
    reproj_err = 0
    frames_R = np.zeros((frames_num,3,3), dtype=np.float32)
    frames_T = np.zeros((frames_num,3), dtype=np.float32)
    world_pnts =  np.zeros((points_num,3), dtype=np.float32)
    reproj_err_history = []
    while iter < max_iter:
        if debug >= 3: print("iter={0}".format(iter))

        # estimage camera position [R,T] given distances to all 3D points pj in frame1

        A = np.zeros((3*points_num, 12), dtype=np.float32)
        xs_frame1 = xs_per_image_meter[block_base_frame_ind]
        for frame_ind in range(0, frames_num):
            points_per_frame = xs_per_image_meter[frame_ind]

            A.fill(0)
            for pnt_ind in range(0, points_num):
                x1 = xs_frame1[pnt_ind]

                x_img =  points_per_frame[pnt_ind]
                x_img_skew = skewSymmeticMat(x_img)
                block_left = np.kron(x1.reshape(1,3), x_img_skew)
                A[3 * pnt_ind:3 * (pnt_ind + 1), 0:9] = block_left
                A[3 * pnt_ind:3 * (pnt_ind + 1), 9:12] = alphas[pnt_ind]*x_img_skew

            dVec1, u1, vt1 = cv2.SVDecomp(A)
            r_and_t = vt1.T[:,-1]
            cam_R_noisy = r_and_t[0:9].reshape(3,3, order='F') # unstack
            cam_Tvec_noisy = r_and_t[9:12]

            # project noisy [R,T] onto SO(3) (see MASKS, formula 8.41 and 8.42)
            dVec2, u2, vt2 = cv2.SVDecomp(cam_R_noisy)
            no_guts = np.dot(u2, vt2)
            sign1 = np.sign(LA.det(no_guts))
            frame_R = sign1 * no_guts

            det_den = LA.det(np.diag(dVec2.ravel()))
            if math.isclose(0, det_den):
                diverge = True
                break
            t_factor = sign1 / rootCube(det_den)
            cam_Tvec = t_factor * cam_Tvec_noisy

            frames_R[frame_ind,:,:] = frame_R[:,:]
            frames_T[frame_ind,:] = cam_Tvec[:]
            if sum(1 for a in cam_Tvec if not math.isfinite(a)) > 0:
                print("error: nan")

        if debug >= 3:
            for i in range(0, frames_num):
                rotn, ang = logSO3(frames_R[i,:,:])
                print("cam:{0} T:{1} R:n={2} ang={3}\n{4}".format(i, frames_T[i,:], rotn, math.degrees(ang), frames_R[i,:,:]))

        # find distances to all 3D points in frame1 given position [Ri,Ti] of each frame
        # (MASKS formula 8.44)
        for pnt_ind in range(0, points_num):
            x1 = xs_per_image_meter[block_base_frame_ind][pnt_ind]

            alpha_num = 0
            alpha_den = 0
            for frame_ind in range(1, frames_num): # other cameras, except world frame1
                x2 = xs_per_image_meter[frame_ind][pnt_ind]

                x2_skew = skewSymmeticMat(x2)
                frame_R = frames_R[frame_ind, :, :]
                frame_T = frames_T[frame_ind, :]
                h1 = np.dot(x2_skew, frame_T)
                h2 = np.dot(np.dot(x2_skew, frame_R), x1)
                alpha_num += np.dot(h1, h2)
                alpha_den += LA.norm(h1) ** 2
                if not math.isfinite(alpha_num) or not math.isfinite(alpha_den):
                    print("error: nan")

            alpha = -alpha_num / alpha_den
            alphas[pnt_ind] = alpha
            dist = 1 / alpha
            #assert dist > 0, "distances are positive"
        if sum(1 for a in alphas if not math.isfinite(a)) > 0:
            print("error: nan")

        if debug: print("it: {0} dist: {1}".format(iter, 1/alphas))

        # calculate reprojection error
        reproj_err = 0
        for pnt_ind in range(0, points_num):
            dist = 1 / alphas[pnt_ind] # distance to 3D point in frame1 (world)
            x1 = xs_per_image_meter[block_base_frame_ind][pnt_ind]
            x3D_cam1 = x1 * dist
            world_pnts[pnt_ind, :] = x3D_cam1 # updated coordinate of the world point

            for frame_ind in range(0, frames_num):
                if frame_ind == block_base_frame_ind: continue
                x_expect = xs_per_image_meter[frame_ind][pnt_ind]
                x_expectN = x_expect / x_expect[-1]

                frame_R = frames_R[frame_ind,:,:]
                frame_T = frames_T[frame_ind,:]
                xact_cami = np.dot(frame_R, x3D_cam1) + frame_T
                xact_camiN = xact_cami / xact_cami[-1]

                err = LA.norm(x_expectN - xact_camiN)**2
                if debug >= 3: print("tgxexpect={0} xact={1} err={2} meters".format(x_expect, xact_camiN, err))
                reproj_err += err

        if debug >= 3: print("Xs3D:\n{0}".format(world_pnts))
        if debug >= 3: print("reproj_err={0} meters".format(reproj_err))

        reproj_err_history.append(reproj_err)

        if reproj_err < min_reproj_err:
            break
        iter += 1
    if debug >= 3: print("Search succeeded in {0} iterations, reproj_err: {1}".format(iter, reproj_err))
    if debug >= 3: print("reproj_err_history={0}".format(reproj_err_history))

    if iter >= max_iter or diverge:
        return False, frames_R, frames_T, world_pnts # is not converged

    return True, frames_R, frames_T, world_pnts

def convertPointPixelToMeter(x_pix, world_from_camera_mat):
    assert len(x_pix) == 2 or len(x_pix) == 3

    pnt_homog = x_pix if len(x_pix) == 3 else [x_pix[0], x_pix[1], 1]
    pnt_meter = np.dot(world_from_camera_mat, pnt_homog)
    pnt_meter = pnt_meter.astype(type(x_pix[0]))
    return pnt_meter

# xs_per_image_meter = result points
def convertPixelToMeterPoints(camMat, xs_per_image, xs_per_image_meter):
    assert len(xs_per_image[0][0]) == 2, "Provide 2D coordinates (x,y) of pixels"

    suc, Km1 = cv2.invert(camMat)
    assert suc

    # convert image coords to unaclibrated coordinates
    for img_ind in range(0, len(xs_per_image)):
        xs = xs_per_image[img_ind]

        xs_meter = []
        for i in range(0, len(xs)):
            pnt = xs[i]
            xmeter = convertPointPixelToMeter(pnt, Km1)
            # assert len(pnt) == 2
            # pnt_hom = [pnt[0], pnt[1], 1]
            # xmeter = np.dot(Km1, pnt_hom)
            # print("i:{0} x'={1} x={2} xRe'={3}".format(i, pnt, xmeter, np.dot(camMat, xmeter)))
            # xs_meter.append((xmeter[0], xmeter[1]))
            xs_meter.append(xmeter)
        xs_meter = np.array(xs_meter)
        xs_per_image_meter.append(xs_meter)

def convertPixelToMeterPointsNew(cam_mat_meter_from_pixel, xs_per_image, xs_per_image_meter):
    assert len(xs_per_image[0][0]) == 2, "Provide 2D coordinates (x,y) of pixels"

    Km1 = cam_mat_meter_from_pixel

    # convert image coords to unaclibrated coordinates
    for img_ind in range(0, len(xs_per_image)):
        xs = xs_per_image[img_ind]

        xs_meter = []
        for i in range(0, len(xs)):
            pnt = xs[i]
            xmeter = convertPointPixelToMeter(pnt, Km1)
            # assert len(pnt) == 2
            # pnt_hom = [pnt[0], pnt[1], 1]
            # xmeter = np.dot(Km1, pnt_hom)
            # print("i:{0} x'={1} x={2} xRe'={3}".format(i, pnt, xmeter, np.dot(camMat, xmeter)))
            # xs_meter.append((xmeter[0], xmeter[1]))
            xs_meter.append(xmeter)
        xs_meter = np.array(xs_meter)
        xs_per_image_meter.append(xs_meter)


def experimentWithEssentialMat(camMat, xs_per_image_meter):
    assert len(xs_per_image_meter[0][0]) == 3, "Provide 2D homogeneous coordinates [x,y,w] (in meters)"

    suc, Km1 = cv2.invert(camMat)
    assert suc

    xs1_meter = np.array(xs_per_image_meter[0])
    xs2_meter = np.array(xs_per_image_meter[1])

    perror = [0.0]
    suc, essMat = FindEssentialMat8Point(xs1_meter, xs2_meter, unity_translation=True, plinear_sys_err=perror)
    assert suc
    print("essMat=\n{0} err={1}".format(essMat, perror[0]))

    # find epipoles
    (dVec, u, vt) = cv2.SVDecomp(essMat)
    epi1 = vt.T[:, -1]
    epi1 = epi1 / epi1[-1]
    epi1_img = np.dot(camMat, epi1)
    epi1_img = epi1_img / epi1_img[-1]
    print("essMat epipole1={0} epipole1'={1}".format(epi1, epi1_img))

    (dVec2, u2, vt2) = cv2.SVDecomp(essMat.T)
    epi2 = vt2.T[:, -1]
    epi2 = epi2 / epi2[-1]
    epi2_img = np.dot(camMat, epi2)
    epi2_img = epi2_img / epi2_img[-1]
    print("essMat epipole2={0} epipole2'={1}".format(epi2, epi2_img))

    #
    suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(essMat, xs1_meter, xs2_meter)
    assert suc
    ess_T = skewSymmeticMat(ess_Tvec)
    ess_wvec, ess_wang = logSO3(ess_R)
    print("R|T: w={0} ang={1}\n{2}\n{3}".format(ess_wvec, math.degrees(ess_wang), ess_R, ess_Tvec))

    pix_R = np.dot(np.dot(camMat, ess_R), Km1)
    pix_Tvec = np.dot(camMat, ess_Tvec)
    print("P'[qR,qT]:\n{0}\n{1}".format(pix_R, pix_Tvec))

    # construct fund mat
    # MASKS formula 6.12
    fundMatFromEss = np.dot(np.dot(Km1.T, essMat), Km1)
    # fundMatFromEss = np.dot(np.dot(np.dot(skewSymmeticMat(np.dot(camMat, ess_Tvec)), camMat), ess_R), Km1) # also works
    fundMatFromEss = fundMatFromEss / fundMatFromEss[-1, -1]
    print("fundMatFromEssMat=\n{0}".format(fundMatFromEss))

    (dVec, u, vt) = cv2.SVDecomp(fundMatFromEss)
    epi1 = vt.T[:, -1]
    epi1 = epi1 / epi1[-1]
    print("fundMatFromEss1 epipole1={0}".format(epi1))

    (dVec, u, vt) = cv2.SVDecomp(fundMatFromEss.T)
    epi2 = vt.T[:, -1]
    epi2 = epi2 / epi2[-1]
    print("fundMatFromEss1 epipole2={0}".format(epi2))

# Gets the space in which the 2D coordinates can be extended starting from the given center without overflowing an image's boundary.
# center=[1x2]=(x,y)
def Exten2D_ExtendPoint(center, win_width, img_shape):
    half_width = int(win_width / 2)

    imgh, imgw = img_shape[0:2]
    off_left =  half_width if center[0] >= half_width else center[0]
    off_right = half_width if center[0] + half_width < imgw else imgw - center[0] - 1
    off_top =   half_width if center[1] >= half_width else center[1]
    off_bot =   half_width if center[1] + half_width < imgh else imgh - center[1] - 1
    assert off_left >= 0
    assert off_right >= 0
    assert off_top >= 0
    assert off_bot >= 0
    return off_left, off_right, off_top, off_bot

def Exten2D_ApplyToPoint(center, off_left, off_right, off_top, off_bot):
    rect_lef = center[0] - off_left
    rect_rig = center[0] + off_right + 1
    rect_top = center[1] - off_top
    rect_bot = center[1] + off_bot + 1
    return rect_lef, rect_rig, rect_top, rect_bot


# image1 - gray image
# x1:int[1x2], x2:int[1,2]
# returns value in the [-1,1] range.
def NormalizedCrossCorrelation(image1, image2, p1, p2, win_width):
    assert len(image1.shape) == 2, "Correlation is computed on gray images" # w,h
    assert isinstance(p1[0], int) and isinstance(p1[1], int), "Provide point with integer coordinates"
    assert isinstance(p2[0], int) and isinstance(p2[1], int)

    off1_lef, off1_rig, off1_top, off1_bot = Exten2D_ExtendPoint(p1, win_width, image1.shape)
    off2_lef, off2_rig, off2_top, off2_bot = Exten2D_ExtendPoint(p2, win_width, image2.shape)

    # p1 and p2 extensions
    p1_lef, p1_rig, p1_top, p1_bot = Exten2D_ApplyToPoint(p1, off1_lef, off1_rig, off1_top, off1_bot)
    p2_lef, p2_rig, p2_top, p2_bot = Exten2D_ApplyToPoint(p2, off2_lef, off2_rig, off2_top, off2_bot)

    # compute average intensity of the part of the image [c_lef, c_rig, c_top, c_bot]
    def AvgIntens(img, c_lef, c_rig, c_top, c_bot):
        sum = 0.0
        for row in range(c_top, c_bot):
            for col in range(c_lef, c_rig):
                sum += img[row, col]
        num = (c_rig - c_lef) * (c_bot - c_top)
        res = sum / num
        return res

    avg_intens1 = AvgIntens(image1, p1_lef, p1_rig, p1_top, p1_bot)
    avg_intens2 = AvgIntens(image2, p2_lef, p2_rig, p2_top, p2_bot)

    # find common part of two neighbourhoods of both points
    comm_lef = min(off1_lef, off2_lef)
    comm_rig = min(off1_rig, off2_rig)
    comm_top = min(off1_top, off2_top)
    comm_bot = min(off1_bot, off2_bot)

    # compute cross-corelation, see MASKS p 386, formula 11.11
    corr = 0.0
    den1 = 0.0
    den2 = 0.0
    for yoff in range(-comm_top, comm_bot+1):
        for xoff in range(-comm_lef, comm_rig+1):
            v1 = image1[p1[1]+yoff, p1[0]+xoff] - avg_intens1
            v2 = image2[p2[1]+yoff, p2[0]+xoff] - avg_intens2
            corr += v1 * v2
            den1 += v1 * v1
            den2 += v2 * v2
    res = corr / math.sqrt(den1 * den2)
    return res

# Calls cv2.imshow with image, merged from two given images with using given fraction.
# transit=0 - previous image
# transit=1 - next image
def FuseInteractive(win_name, img1_adorn, img2_adorn, transit = 0, print_transit=True):
    img3 = np.zeros_like(img1_adorn)
    update_image = True
    while True:
        if update_image:
            img3 = cv2.addWeighted(img1_adorn, 1-transit, img2_adorn, transit, 0, dst=img3)
            update_image = False
        cv2.imshow(win_name, img3)
        k = cv2.waitKey()

        transit_new = transit
        if k == 81 or k == 52: # left or NumPad4
            transit_new = transit - 0.3
        elif k == 83 or k == 54: # right or NumPad6
            transit_new = transit + 0.3
        elif k == 27: # Esc
            break

        transit_new = clamp(transit_new, 0, 1)
        if transit != transit_new:
            transit = transit_new
            update_image = True
            if print_transit: print('transit={0}'.format(transit))
    cv2.destroyWindow(win_name)

def ShowMatches(win_name, image1, image2, points, next_pts, transit = None, print_transit=False):
    color_from = [255,0,0]
    #color_to = [255,0,255]
    color_to = color_from
    color_new = [0,255,0] # p1=None p2=Obj
    color_deleted = [0,0,255] # p1=Obj p2=None
    img1_adorn = image1.copy()
    img2_adorn = image2.copy()
    for p1f,p2f in zip(points, next_pts):
        p1 = IntPnt(p1f)
        p2 = IntPnt(p2f)
        if not p1 is None:
            is_del = p2 is None
            c = color_deleted if is_del else color_from
            cv2.circle(img1_adorn, (p1[0], p1[1]), 5, c)
        # if not p2 is None:
        #     cv2.circle(img1_adorn, (p2[0], p2[1]), 3, color_to)

        if not p1 is None and not p2 is None:
            cv2.line(img1_adorn, (p1[0], p1[1]), (p2[0], p2[1]), color_from)

        # if not p1 is None:
        #     cv2.circle(img2_adorn, (p1[0], p1[1]), 5, color_to)
        if not p2 is None:
            is_new = p1 is None
            c = color_new if is_new else color_from
            cv2.circle(img2_adorn, (p2[0], p2[1]), 3, c)

    transit = transit if not transit is None else 0.5
    FuseInteractive(win_name, img1_adorn, img2_adorn, transit=transit,print_transit=print_transit)


# Fills 3x3 fundamental matrix from 8-parameters parameterization fvec[8].
# This ensures that fundamental matrix has rank 2 by construction.
#MASKS page 392, formula 11.13
def NewFundMatFrom8Params(fvec, m):
    f1, f2, f4, f5, a1, b1, a2, b2 = fvec

    if m is None:
        m = np.zeros((3,3), np.float)
    m[0, 0] = f1
    m[1, 0] = f2
    m[0, 1] = f4
    m[1, 1] = f5
    m[0, 2] = a1 * f1 + b1 * f4
    m[1, 2] = a1 * f2 + b1 * f5
    m[2, 0] = a2 * f1 + b2 * f2
    m[2, 1] = a2 * f4 + b2 * f5
    m[2, 2] = a1 * a2 * f1 + a1 * b2 * f2 + b1 * a2 * f4 + b1 * b2 * f5
    return m

# Represents fundamental matrix in the form of 8-parameters.
# fund_mat[3x3]
# fvec[8] result 8 parameters.
def Get8ParamsFromFundMat(fund_mat, fvec):
    #MASKS page 392, formula 11.13
    f1 = fund_mat[0, 0]
    f2 = fund_mat[1, 0]
    f4 = fund_mat[0, 1]
    f5 = fund_mat[1, 1]

    a1, b1 = LA.solve(fund_mat[0:2, 0:2],   fund_mat[0:2, 2]) # from last row
    a2, b2 = LA.solve(fund_mat[0:2, 0:2].T, fund_mat[2, 0:2]) # from last col

    elem33 = a1*a2*f1 + a1*b2*f2 + b1*a2*f4 + b1*b2*f5
    assert np.isclose(fund_mat[2,2], elem33)

    if fvec is None:
        fvec = np.zeros(8, fund_mat.dtype)
    np.copyto(fvec, [f1, f2, f4, f5, a1, b1, a2, b2])
    return fvec

def RefineFundMat(fund_mat, xs1, xs2, debug = 0):
    fvec0 = Get8ParamsFromFundMat(fund_mat, None)

    cur_fund_mat = np.zeros_like(fund_mat)
    NewFundMatFrom8Params(fvec0, cur_fund_mat)

    sampson = SampsonDistanceCalc()
    dist = sampson.DistanceMult(cur_fund_mat, xs1, xs2)

    # Objective function for minimization Z::R8->R
    # fvec = [f1, f2, f4, f5, a1, b1, a2, b2]
    def SampsonDistFun(fvec):
        NewFundMatFrom8Params(fvec, cur_fund_mat)
        return sampson.DistanceMult(cur_fund_mat, xs1, xs2)
    def SampsonDistPrimeFun(fvec):
        fvec_prime = np.zeros_like(fvec)
        suriko.sampson.SampsonDistanceMultPrime(fvec, xs1, xs2, fvec_prime)
        return fvec_prime

    dist_init = SampsonDistFun(fvec0)
    if debug >= 3: print("initial dist={0}".format(dist_init))

    #maxiter = 16 # T=(1,0,0) NOTE: small num of iters may result in shift
    maxiter = 96
    if debug >= 3: print("optimizing, maxiter={0}".format(maxiter))

    # Way1
    t1 = time.time()
    opt_res = optimize.minimize(SampsonDistFun, fvec0, jac=SampsonDistPrimeFun, options={'maxiter': maxiter})
    t2 = time.time()
    #assert opt_res.success
    if opt_res.success:
        fvec_opt = opt_res.x
        dist_minimize = SampsonDistFun(fvec_opt)
        if debug >= 3: print("method=minimize dist={0} took={1:.4f}s {2}".format(dist_minimize, t2-t1, fvec_opt))
    elif opt_res.status==2 or opt_res.status==1:
        # target optimization criteria is not met
        # 1=Maximum number of iterations has been exceeded.
        # 2=Desired error not necessarily achieved due to precision loss.
        fvec_opt = opt_res.x
        dist_minimize = SampsonDistFun(fvec_opt)
        if debug >= 3: print("method=minimize term criteria was not met, dist={0} took={1:.4f}s {2}".format(dist_minimize, t2 - t1, fvec_opt))
    else:
        if debug >= 3: print("method=minimize fails took={0:.4f}s".format(t2 - t1))
        return False, None

    # # Way2
    # t1 = time.time()
    # opt_res = optimize.fmin_ncg(SampsonDist, fvec0, maxiter=maxiter, fprime=SampsonDistPrime)
    # t2 = time.time()
    # fvec_opt = opt_res
    # dist_fmin_cg = SampsonDist(fvec_opt)
    # print("method=fmin_ncg dist={0} took={1:.4f}s {2}".format(dist_fmin_cg, t2 - t1, fvec_opt))
    #
    # # Way3
    # t1 = time.time()
    # opt_res = optimize.fmin_cg(SampsonDist, fvec0, maxiter=maxiter, fprime=SampsonDistPrime)
    # t2 = time.time()
    # fvec_opt = opt_res
    # dist_fmin_cg = SampsonDist(fvec_opt)
    # print("method=fmin_cg dist={0} took={1:.4f}s {2}".format(dist_fmin_cg, t2 - t1, fvec_opt))

    assert dist_minimize <= dist_init, "The Samson distance is not decreased"

    NewFundMatFrom8Params(fvec_opt, cur_fund_mat)
    return True, cur_fund_mat

def TestSampsonDistance():
    cur_fund_mat = np.zeros((3,3), np.float)
    fvec0 = np.array([2, 3, 4, 5, 6, 7, 8, 9], np.float)
    NewFundMatFrom8Params(fvec0, cur_fund_mat)

    # sampson distance
    sampson = SampsonDistanceCalc()
    d1 = sampson.DistanceMult(cur_fund_mat, [[2, 3, 4]], [[5, 6, 7]])
    assert np.isclose(1330.31282486, d1)

    # gradient
    fvec = np.zeros(8)
    suriko.sampson.SampsonDistanceMultPrime([2, 3, 4, 5, 6, 7, 8, 9], [[2, 3, 4]], [[5, 6, 7]], fvec)
    prime_expect = [36.045, 36.4307, -16.1044, -23.3929, 109.888, 196.978, 18.3806, 26.3365]
    assert np.allclose(prime_expect, fvec)

    # test gradient precision
    grad_prec = True
    if grad_prec:
        cur_fund_mat2 = np.zeros((3, 3), np.float)
        deltas_for_grad = np.zeros_like(fvec0)
        grad = np.zeros_like(fvec0)
        xs1 = [[2, 3, 4]]
        xs2 = [[5, 6, 7]]
        for ten_power in range(1, 10):
            delt = 1.0 / 10**ten_power

            for pos in range(0, len(deltas_for_grad)):
                deltas_for_grad.fill(0)
                deltas_for_grad[pos] = delt
                NewFundMatFrom8Params(fvec0 + deltas_for_grad, cur_fund_mat2)

                d2 = sampson.DistanceMult(cur_fund_mat2, xs1, xs2)
                g = (d2 - d1)/delt
                grad[pos] = g
            print("{0}: {1}".format(ten_power, grad))


# Estimates the number of times to sample in RANSAC algorithm.
# TUM, Lecture 7, "Visual Navigation for Flying Robots (Dr. Jürgen Sturm)" t=6m3s
# https://www.youtube.com/watch?v=5E5n7fhLHEM
# suc_prob = desired probability of success
def RansacIterationsCount(samp_size, outlier_ratio, suc_prob):
    sample_times = math.log(1 - suc_prob) / math.log(1 - (1 - outlier_ratio) ** samp_size)
    return math.ceil(sample_times)

# Performs RANSAC to find the maximal (cardinality) subset of items which agree on computation of some model.
# Those items which marked True are called inliers, the rest is outliers.
# samp_size = number of points in each sample to select
# consensus_fun(items_count, samp_group_inds, cons_set_mask)->consensus_set_card
# MASKS page 389
def GetMaxSubsetInConsensus(items_count, samp_size, outlier_ratio, suc_prob, consensus_fun, consensus_mask, debug = 3):
    sample_times = RansacIterationsCount(samp_size, outlier_ratio, suc_prob)

    best_cons_set_mask = np.zeros(items_count, np.uint8)
    cur_cons_mask = np.zeros(items_count, np.uint8)
    best_cons_set_card = -1

    random.seed(313)

    good = range(0, items_count)
    for samp_ind in range(0, sample_times):
        # generate new sample
        num_points = samp_size
        samp_group_inds = random.sample(good, num_points)

        cur_cons_mask.fill(0)
        consensus_set_card = consensus_fun(items_count, samp_group_inds, cur_cons_mask)
        if consensus_set_card > best_cons_set_card:
            best_cons_set_card = consensus_set_card
            #if debug >= 3: print("best_cons_set_card={0}".format(best_cons_set_card))

            best_cons_set_mask[:] = cur_cons_mask[:]


    # reestimate essential mat on all matched points in consensus
    assert best_cons_set_card >= 0
    consensus_mask[:] = best_cons_set_mask

    return best_cons_set_card


# (wide baseline)
# samp_size = number of points in each sample to select
# attempt of my impl
def MatchKeypointsAndGetEssentialMatNarrowBaselineCore(samp_size, cam_mat, xs1_meter, xs2_meter, image1, image2, xs1_pixels, xs2_pixels, debug = 3):
    e3 = [0, 0, 1]
    e3_hat = skewSymmeticMat(e3)

    # calculate consensus set - the points which agree with current essential set
    def CalcConsensus(good, ess_mat, fit_ess_mat_thr, consensus_set=None):
        cons_set_card = 0
        dist_list = []
        for i in good:
            pt1 = xs1_meter[i]
            pt2 = xs2_meter[i]
            pt1_normed = pt1
            pt2_normed = pt2

            # MASKS page 388, formula 11.12
            h1 = np.dot(np.dot(pt1_normed, ess_mat), pt2_normed)
            den1 = np.dot(np.dot(e3_hat, ess_mat), pt1_normed)
            den2 = np.dot(pt2_normed, np.dot(ess_mat, e3_hat))
            samps_dist = h1 * h1 / (LA.norm(den1) ** 2 + LA.norm(den2) ** 2)

            if samps_dist < fit_ess_mat_thr:
                cons_set_card += 1
                if not consensus_set is None:
                    consensus_set.append(i)

            # print("samps_dist={0}m_normed, {1} thr={2}".format(samps_dist, samps_dist < fit_ess_mat_thr, fit_ess_mat_thr))
            # img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, [good[i]], None, flags=2)
            # cv2.imshow("matches", img3)
            # cv2.waitKey()
            dist_list.append(samps_dist)

        # plt.hist(dist_list, bins='auto')
        # plt.hist(dist_list, bins=np.arange(0,0.2,0.01))
        # plt.show()
        return cons_set_card

    def GetMatchedCoords(samp_group):
        xs1 = []
        xs2 = []
        for i in samp_group:
            pt1 = xs1_meter[i]
            pt2 = xs2_meter[i]
            xs1.append(pt1)
            xs2.append(pt2)

        return xs1, xs2

    expectAvgDist = math.sqrt(2)
    point_fit_ess_mat_thr_pix = 3  # pixels
    sample_times = 200 # number of times to sample

    best_ess_mat = None
    best_cons_set_inds = None
    best_cons_set_card = 0
    best_samp_group_inds = None
    best_T1 = None # used if norming is done
    best_T2 = None

    random.seed(313)

    good = range(0, len(xs1_meter))
    for samp_ind in range(0, sample_times):
        # generate new sample
        num_points = samp_size # min=8
        samp_group_inds = random.sample(good, num_points)

        samp_xs1_meter = [xs1_meter[i] for i in samp_group_inds]
        samp_xs2_meter = [xs2_meter[i] for i in samp_group_inds]

        if debug >= 994:
            samp_xs1_pixels = [xs1_pixels[i] for i in samp_group_inds]
            samp_xs2_pixels = [xs2_pixels[i] for i in samp_group_inds]
            ShowMatches("sample", image1, image2, samp_xs1_pixels, samp_xs2_pixels)

        perror = [0.0]
        suc, cur_ess_mat = FindEssentialMat8Point(samp_xs1_meter, samp_xs2_meter, unity_translation=True, plinear_sys_err=perror, debug=debug)
        assert suc
        #print("essMat=\n{0} err={1}".format(ess_mat, perror[0]))

        pix_to_meter = LA.inv(cam_mat)
        point_fit_ess_mat_thr_meter = point_fit_ess_mat_thr_pix * pix_to_meter[0, 0]  # meters

        cons_set = []
        consensus_set_card = CalcConsensus(good, cur_ess_mat, point_fit_ess_mat_thr_meter, consensus_set=cons_set)
        if consensus_set_card > best_cons_set_card:
            best_ess_mat = cur_ess_mat
            best_cons_set_inds = cons_set
            best_cons_set_card = consensus_set_card
            best_samp_group_inds = samp_group_inds
            if debug >= 3: print("best_cons_set_card={0} best_ess_mat=\n{1} err={2}".format(best_cons_set_card, best_ess_mat, perror[0]))

            if debug >= 994:
                cons_xs1_pixels = [xs1_pixels[i] for i in cons_set]
                cons_xs2_pixels = [xs2_pixels[i] for i in cons_set]
                ShowMatches("cons", image1, image2, cons_xs1_pixels, cons_xs2_pixels)
        else:
            #if debug >= 3: print("not best, candidate consensus={0}".format(consensus_set_card))
            pass

    if best_ess_mat is None:
        if debug >= 3: print("Can't find essential matrix")
        return None

    if debug >= 4:
        print("Found best essential matrix")
        print("best_cons_set_card={0} best_ess_mat=\n{1}".format(best_cons_set_card, best_ess_mat))

        samp_xs1_pixels = [xs1_pixels[i] for i in best_samp_group_inds]
        samp_xs2_pixels = [xs2_pixels[i] for i in best_samp_group_inds]
        ShowMatches("best_samp_group", image1, image2, samp_xs1_pixels, samp_xs2_pixels)

    # reestimate essential mat on all matched points in consensus
    assert not best_cons_set_inds is None
    assert len(best_cons_set_inds) == best_cons_set_card

    xs1_cons = [xs1_meter[i] for i in best_cons_set_inds]
    xs2_cons = [xs2_meter[i] for i in best_cons_set_inds]

    if debug >= 4:
        xs1_cons_pixels = [xs1_pixels[i] for i in best_cons_set_inds]
        xs2_cons_pixels = [xs2_pixels[i] for i in best_cons_set_inds]
        ShowMatches("consensus", image1, image2, xs1_cons_pixels, xs2_cons_pixels)

    perror = [0.0]
    suc, cons_ess_mat = FindEssentialMat8Point(xs1_cons, xs2_cons, unity_translation=True, plinear_sys_err=perror, debug=debug)
    assert suc
    if debug: print("cons_ess_mat=\n{0} err={1}".format(cons_ess_mat, perror[0]))

    return cons_ess_mat, best_cons_set_inds

def MatchKeypointsAndGetEssentialMatNarrowBaseline(samp_size, cam_mat, image1, image2, do_norming, debug = 3):
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    kpd = cv2.xfeatures2d.SIFT_create(nfeatures=50)  # OpenCV-contrib-3

    kp1 = kpd.detect(img1_gray, None)

    points = np.array([kp.pt for kp in kp1], np.float32)

    #term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    #points2 = cv2.cornerSubPix(img1_gray, points, (10,10), (-1,-1), term)

    # try to find the corner in the next image
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, points, None)
    #next_pts2 = cv2.cornerSubPix(img1_gray, next_pts, (10, 10), (-1, -1), term)

    status = status.ravel()
    xs1_pixels = [p for s,p in zip(status,points) if s]
    xs2_pixels = [p for s,p in zip(status,next_pts) if s]

    if debug >= 4:
        ShowMatches("initial match", image1, image2, xs1_pixels, xs2_pixels)

    # convert pixel to image coordinates (pixel -> meters)
    xs1_meter = []
    xs2_meter = []
    convertPixelToMeterPoints(cam_mat, [xs1_pixels], xs1_meter)
    convertPixelToMeterPoints(cam_mat, [xs2_pixels], xs2_meter)
    xs1_meter = xs1_meter[0]
    xs2_meter = xs2_meter[0]

    # perform normalization
    expectAvgDist = math.sqrt(2)
    T1 = None
    T2 = None
    xs1_meter_norm = xs1_meter
    xs2_meter_norm = xs2_meter
    if do_norming:
        T1 = calcNormalizationTransform(xs1_meter, expectAvgDist)
        T2 = calcNormalizationTransform(xs2_meter, expectAvgDist)

        xs1_meter_norm = np.dot(xs1_meter, T1.T)
        xs2_meter_norm = np.dot(xs2_meter, T2.T)


    cons_ess_mat, cons_set_inds = MatchKeypointsAndGetEssentialMatNarrowBaselineCore(samp_size, cam_mat, xs1_meter_norm, xs2_meter_norm, image1, image2, xs1_pixels, xs2_pixels, debug=debug)

    # denormalize F
    if do_norming:
        assert not T1 is None
        assert not T2 is None
        cons_ess_mat_denormed = np.dot(T2.T, np.dot(cons_ess_mat, T1))
        # Xmeter_normed = T1*Xmeter
        # T1_inv = LA.inv(T1)
        # T2_inv = LA.inv(T2)
        # xs1_meter = np.dot(xs1_normed, T1_inv.T)
        # xs2_meter = np.dot(xs2_normed, T2_inv.T)
        # cons_xs1 = [xs1_meter_norm[i] for i in cons_set_inds]
        # cons_xs2 = [xs2_meter_norm[i] for i in cons_set_inds]
    else:
        cons_ess_mat_denormed = cons_ess_mat

    cons_xs1 = [xs1_meter[i] for i in cons_set_inds]
    cons_xs2 = [xs2_meter[i] for i in cons_set_inds]

    return cons_ess_mat_denormed, cons_xs1, cons_xs2


# (wide baseline)
# attempt of my impl
def MatchKeypointsAndGetEssentialMatNcc(image1, image2):
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # BRISK detector - very good!
    # thresh = 30-50: good, >80: not all chessboard corners detected
    brisk = cv2.BRISK_create(thresh=30, octaves=0)
    key_points1 = brisk.detect(image_gray1, mask=None)
    key_points2 = brisk.detect(image_gray2, mask=None)

    img_kps1 = np.zeros(image1.shape, dtype=np.uint8)
    img_kps2 = np.zeros(image1.shape, dtype=np.uint8)
    cv2.drawKeypoints(image1, key_points1, img_kps1, color=(0, 255, 0), flags=0)
    cv2.drawKeypoints(image2, key_points2, img_kps2, color=(0, 255, 0), flags=0)
    cv2.imshow("corners1", img_kps1)
    cv2.imshow("corners2", img_kps2)
    cv2.waitKey(0)

    n1 = len(key_points1)
    n2 = len(key_points2)
    match_score_table = np.zeros((n1, n2), dtype=np.float32)
    match_score_ladder = [] # sorted list of (x1, x2, score), descending

    # the pair of points with NCC (Normalized Cross Correlation) greater than this value are accepted as a match
    neigh_match_thresh = 0.7

    for i1 in range(0, n1):
        pt1 = key_points1[i1].pt
        pt1_int = IntPnt(pt1)
        for i2 in range(0, n2):
            pt2 = key_points2[i2].pt
            pt2_int = IntPnt(pt2)
            score = NormalizedCrossCorrelation(image1, image2, pt1_int, pt2_int, win_width=15)
            match_score_table[i1, i2] = score

            if score > neigh_match_thresh:
                match_score_ladder.append((pt1_int, pt2_int, score))

    match_score_ladder.sort(key=lambda tup: tup[2], reverse=True) # sort by score
    print(match_score_ladder)

    # TODO: construct list of matches

# Computes essential matrix between two images.
# (wide baseline)
# It finds corners in both images. Then it matches the corners to get correspondences.
# Then it uses RANSAC technique to select 5 corners which result in the essential
# matrix with the biggest consensus.
# NOTE: fails to recognize correspondences for both narraow and wide baselines.
#       In the narrow case, the points may be close, but the SIFT-matcher prefer some another distant point.
def MatchKeypointsAndGetEssentialMatSift(samp_size, cam_mat, image1, image2, do_norming, debug = 3):
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    # edgeThreshold, default=10, lesser value discards more corners on the line segments
    kpd = cv2.xfeatures2d.SIFT_create(nfeatures=50, edgeThreshold=5) # OpenCV-contrib-3

    # find the keypoints and descriptors with SIFT
    kp1, des1 = kpd.detectAndCompute(img1_gray, None)
    kp2, des2 = kpd.detectAndCompute(img2_gray, None)

    img_kps1 = cv2.drawKeypoints(image1, kp1, None, color=(0, 255, 0), flags=0)
    img_kps2 = cv2.drawKeypoints(image2, kp2, None, color=(0, 255, 0), flags=0)
    if debug >= 1:
        cv2.imshow("corners1", img_kps1)
        cv2.imshow("corners2", img_kps2)
        cv2.waitKey()

    bf = cv2.BFMatcher()  # for SIFT

    # http://docs.opencv.org/3.0-beta/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html#descriptormatcher-knnmatch
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    #
    if debug >= 4:
        img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, random.sample(good, 10), None, flags=2)  # Draw first 10 matches.
        cv2.imshow("good features, matches={0}, good={1}".format(len(matches), len(good)), img3)
        cv2.waitKey()

    # suc, Km1 = cv2.invert(cam_mat)
    # assert suc

    # convert matched points pixels->meters
    xs_per_image1 = []
    xs_per_image2 = []
    for i in range(0, len(matches)):
        match = matches[i][0]
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        xs_per_image1.append(pt1)
        xs_per_image2.append(pt2)

    xs_per_image1_meter = []
    xs_per_image2_meter = []
    convertPixelToMeterPoints(cam_mat, [xs_per_image1], xs_per_image1_meter)
    convertPixelToMeterPoints(cam_mat, [xs_per_image2], xs_per_image2_meter)
    xs_per_image1_meter = xs_per_image1_meter[0]
    xs_per_image2_meter = xs_per_image2_meter[0]
    print("xs_per_image1={0}".format(xs_per_image1))
    print("xs_per_image1_meter={0}".format(xs_per_image1_meter))

    e3 = [0, 0, 1]
    e3_hat = skewSymmeticMat(e3)

    # calculate consensus set - the points which agree with current essential set
    def CalcConsensus(good, ess_mat, fit_ess_mat_thr, do_norming, T1=None, T2=None, consensus_set=None):
        cons_set_card = 0
        dist_list = []
        for i in range(0, len(good)):
            match = good[i][0]
            pt1 = xs_per_image1_meter[match.queryIdx]
            pt2 = xs_per_image2_meter[match.trainIdx]

            pt1_normed = pt1
            pt2_normed = pt2
            if do_norming:
                pt1_normed = np.dot(T1, pt1)
                pt2_normed = np.dot(T2, pt2)

            # MASKS page 388, formula 11.12
            h1 = np.dot(np.dot(pt1_normed, ess_mat), pt2_normed)
            den1 = np.dot(np.dot(e3_hat, ess_mat), pt1_normed)
            den2 = np.dot(pt2_normed, np.dot(ess_mat, e3_hat))
            samps_dist = h1 * h1 / (LA.norm(den1) ** 2 + LA.norm(den2) ** 2)

            if samps_dist < fit_ess_mat_thr:
                cons_set_card += 1
                if not consensus_set is None:
                    consensus_set.append([match])

            # print("samps_dist={0}m_normed, {1} thr={2}".format(samps_dist, samps_dist < fit_ess_mat_thr, fit_ess_mat_thr))
            # img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, [good[i]], None, flags=2)
            # cv2.imshow("matches", img3)
            # cv2.waitKey()
            dist_list.append(samps_dist)

        # plt.hist(dist_list, bins='auto')
        # plt.hist(dist_list, bins=np.arange(0,0.2,0.01))
        # plt.show()
        return cons_set_card

    def GetMatchedCoords(samp_group, do_norming):
        xs1 = []
        xs2 = []
        for i in range(0, len(samp_group)):
            match = samp_group[i][0]
            pt1 = xs_per_image1_meter[match.queryIdx]
            pt2 = xs_per_image2_meter[match.trainIdx]
            xs1.append(pt1)
            xs2.append(pt2)

        # perform normalization
        T1 = None
        T2 = None
        if do_norming:
            T1 = calcNormalizationTransform(xs1, expectAvgDist)
            T2 = calcNormalizationTransform(xs2, expectAvgDist)

            xs1_normed = np.dot(xs1, T1.T)
            xs2_normed = np.dot(xs2, T2.T)
        else:
            xs1_normed = xs1
            xs2_normed = xs2
        return xs1_normed, xs2_normed, T1, T2

    expectAvgDist = math.sqrt(2)
    point_fit_ess_mat_thr_pix = 5  # pixels
    sample_times = 200 # number of times to sample

    best_ess_mat = None
    best_cons_set = None
    best_cons_set_card = 0
    best_samp_group = None
    best_T1 = None # used if norming is done
    best_T2 = None

    random.seed(312)

    for samp_ind in range(0, sample_times):
        # generate new sample
        num_points = samp_size
        samp_group = random.sample(good, num_points)

        if debug >= 4:
            img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, samp_group, None, flags=2)  # Draw first 10 matches.
            cv2.imshow("matches", img3)
            cv2.waitKey()

        xs1_normed, xs2_normed, T1, T2 = GetMatchedCoords(samp_group, do_norming)

        perror = [0.0]
        suc, cur_ess_mat = FindEssentialMat8Point(xs1_normed, xs2_normed, unity_translation=True, plinear_sys_err=perror)
        assert suc
        #print("essMat=\n{0} err={1}".format(ess_mat, perror[0]))

        pix_to_meter = LA.inv(cam_mat)
        # TODO: why T1?
        if do_norming:
            pix_to_meter = np.dot(T1, pix_to_meter)
        point_fit_ess_mat_thr_meter = point_fit_ess_mat_thr_pix * pix_to_meter[0, 0]  # meters

        cons_set = []
        consensus_set_card = CalcConsensus(good, cur_ess_mat, point_fit_ess_mat_thr_meter, do_norming, T1, T2, consensus_set=cons_set)
        if consensus_set_card > best_cons_set_card:
            best_ess_mat = cur_ess_mat
            best_cons_set = cons_set
            best_cons_set_card = consensus_set_card
            best_samp_group = samp_group
            best_T1 = T1
            best_T2 = T2
            if debug >= 3: print("best_cons_set_card={0} best_ess_mat=\n{1} err={2}".format(best_cons_set_card, best_ess_mat, perror[0]))

            if debug >= 4:
                img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, samp_group, None,flags=2)  # Draw first 10 matches.
                cv2.imshow("matches", img3)
                cv2.waitKey()
        else:
            if debug >= 3: print("not best, candidate consensus={0}".format(consensus_set_card))

    if best_ess_mat is None:
        if debug >= 3: print("Can't find essential matrix")
        return None

    if debug >= 3:
        print("Found best essential matrix")
        print("best_cons_set_card={0} best_ess_mat=\n{1}".format(best_cons_set_card, best_ess_mat))

        img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, best_samp_group, None, flags=2)
        cv2.imshow("best_samp_group", img3)
        cv2.waitKey()

    # reestimate essential mat on all matched points in consensus
    assert not best_cons_set is None
    assert len(best_cons_set) == best_cons_set_card

    if debug >= 1:
        img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, best_cons_set, None, flags=2)
        cv2.imshow("best consensus set", img3)
        cv2.waitKey()

    xs1_normed, xs2_normed, T1, T2 = GetMatchedCoords(best_cons_set, do_norming)

    perror = [0.0]
    suc, cons_ess_mat = FindEssentialMat8Point(xs1_normed, xs2_normed, unity_translation=True, plinear_sys_err=perror)
    assert suc
    if debug: print("cons_ess_mat=\n{0} err={1}".format(cons_ess_mat, perror[0]))

    # denormalize F
    cons_ess_mat_denormed = cons_ess_mat
    xs1_meter = xs1_normed
    xs2_meter = xs2_normed
    if do_norming:
        assert not T1 is None
        assert not T2 is None
        cons_ess_mat_denormed = np.dot(T2.T, np.dot(cons_ess_mat, T1))
        # Xmeter_normed = T1*Xmeter
        T1_inv = LA.inv(T1)
        T2_inv = LA.inv(T2)
        xs1_meter = np.dot(xs1_normed, T1_inv.T)
        xs2_meter = np.dot(xs2_normed, T2_inv.T)
    return cons_ess_mat_denormed, xs1_meter, xs2_meter


# Finds fundamental matrix from image points correspondence.
# xs1 list([1x2]), tuple=(x,y)
# source: MVGCV page 282
def findFundamentalMatBasic8Point(xs1, xs2):
    if not xs1.any():
        return None
    onex = xs1[0]
    assert len(onex) == 2 or (len(onex) == 3 and onex[2] == 1), "xs must be normalized"

    pointsCount = len(xs1)
    assert (pointsCount >= 8)

    # perform normalization
    expectAvgDist = math.sqrt(2)

    T1 = calcNormalizationTransform(xs1, expectAvgDist)
    T2 = calcNormalizationTransform(xs2, expectAvgDist)
    # NOTE: without normalization the reconstruction doesn't work; try the line below
    # T1 = T2 = np.eye(3)

    ones = np.ones((len(xs1),1))
    xs1Normed = np.dot(T1, np.hstack((xs1, ones)).T).T
    xs2Normed = np.dot(T2, np.hstack((xs2, ones)).T).T

    # FRank3tmp = findFundamentalMatBasic8PointCore(xs1Normed, xs2Normed)
    #
    # for i in range(pointsCount):
    #     xs1Normed[i] = xs1Normed[i] / LA.norm(xs1Normed[i])

    #
    FRank3 = findFundamentalMatBasic8PointCore(xs1Normed, xs2Normed)
    #error = [-99]
    #FRank3_AI3V = findFundamentalMatBasic8PointCore_MASKS(xs1Normed, xs2Normed, error)

    # enforce constraint rank(F)=2
    def enforceRank2Constraint(FRank3):
        dVec2, u2, vt2 = cv2.SVDecomp(FRank3)
        dVec2[-1] = 0 # zero last eigenvector
        d2 = np.diag(dVec2.ravel())
        FRank2 = np.dot(np.dot(u2, d2), vt2)

        dVec3, u3, vt3 = cv2.SVDecomp(FRank2) # debug
        return FRank2

    # enforce singularity constraint before denormalization (MVGCV page 282)
    FRank2 = enforceRank2Constraint(FRank3)

    # denormalize F
    FDenorm = np.dot(T2.T, np.dot(FRank2, T1))

    FResult = FDenorm
    return FResult

def normLastCellToOne(m):
    m = m / m[-1,-1]
    return m

def findFundamentalMatBasic8PointCore(xs1, xs2):
    pointsCount = len(xs1)
    A = np.zeros((pointsCount, 9), dtype=np.float32);
    for i in range(0, pointsCount):
        x1 = xs1[i]
        x2 = xs2[i]

        A[i, 0] = x2[0] * x1[0]
        A[i, 1] = x2[0] * x1[1]
        A[i, 2] = x2[0]
        A[i, 3] = x2[1] * x1[0]
        A[i, 4] = x2[1] * x1[1]
        A[i, 5] = x2[1]
        A[i, 6] = x1[0]
        A[i, 7] = x1[1]
        A[i, 8] = 1

    dVec, u, vt = cv2.SVDecomp(A)
    #d = np.diag(dVec.ravel())
    # A = np.dot(np.dot(u, d), vt)
    f = vt.T[:, -1]
    # ???assert(size(f,2) == 1);
    # convert vector (row-major) into 3x3 matrix
    #f = f / f[-1]
    FRank3 = f.reshape((3, 3))
    assert FRank3.size == 9
    return FRank3

# MASKS page 121
# MASKS page 212
def findFundamentalMatBasic8PointCore_MASKS(xs1, xs2, error):
    pointsCount = len(xs1)
    assert len(xs1[0]) == 3, "Must be a homogeneous image point (x,y,w)"
    A = np.zeros((pointsCount, 9), dtype=np.float32)
    for i in range(0, pointsCount):
        x1 = xs1[i]
        x2 = xs2[i]

        A[i, 0] = x1[0] * x2[0]
        A[i, 1] = x1[0] * x2[1]
        A[i, 2] = x1[0] * x2[2]
        A[i, 3] = x1[1] * x2[0]
        A[i, 4] = x1[1] * x2[1]
        A[i, 5] = x1[1] * x2[2]
        A[i, 6] = x1[2] * x2[0]
        A[i, 7] = x1[2] * x2[1]
        A[i, 8] = x1[2] * x2[2]

    dVec, u, vt = cv2.SVDecomp(A)
    #d = np.diag(dVec.ravel())
    # A = np.dot(np.dot(u, d), vt)
    f = vt.T[:, -1]
    # ???assert(size(f,2) == 1);
    # convert vector (row-major) into 3x3 matrix
    #f = f / f[-1]
    FRank3 = f.reshape((3, 3), order='F') # reshape columnwise
    assert FRank3.size == 9

    if not error is None:
        error[0] = LA.norm(np.dot(A, f))
    return FRank3

# MVGCV Linear Triangulation Method, Ch12, page 312
# Finds the 3D coordinate of a point (in the first camera frame) from the 2D projection to the frist and to the second image.
# P1 [3x4]
# P2 [3x4]
# x1 [1x2] or [2x1]
# x2 [1x2] or [2x1]
# X [4x1] homogeneous world coordinate.
def triangulateDlt(p1, p2, x1, x2, normalize=True):
    assert len(x1) == 2 or (len(x1) == 3 and x1[2] == 1)
    assert len(x2) == 2 or (len(x2) == 3 and x2[2] == 1)

    A = np.zeros((4, 4))

    if len(x1)==2:
        x13 = x23 = 1
    else:
        x13 = x1[2]
        x23 = x2[2]

    A[0, :] = x1[0] * p1[2, :] - x13*p1[0, :]
    A[1, :] = x1[1] * p1[2, :] - x13*p1[1, :]
    A[2, :] = x2[0] * p2[2, :] - x23*p2[0, :]
    A[3, :] = x2[1] * p2[2, :] - x23*p2[1, :]

    # AX=0
    [dVec, u, vt] = cv2.SVDecomp(A)
    X_frame1 = vt.T[:, -1]
    if normalize:
        X_frame1 = X_frame1 / X_frame1[-1] # TODO: what if the last component is 0

    # postcondition
    # TODO: the conditions below are too flaky
    # r1=np.dot(p1, X)
    # r1=r1/r1[-1]
    # tol = 0.05
    # assert np.isclose(x1[0], r1[0], rtol=tol)
    # assert np.isclose(x1[1], r1[1], rtol=tol)
    #
    # r2=np.dot(p2, X)
    # r2=r2/r2[-1]
    # assert np.isclose(x2[0], r2[0], rtol=tol)
    # assert np.isclose(x2[1], r2[1], rtol=tol)
    return X_frame1

# Fix xs1 and xs2 so that each x1 and x2 conforms to the fundamental matrix.
# MVGCV Sampson approximation, page 315
def correctPointCorrespondenceSampson(fundMat, xs1, xs2):
    result = []
    delta = np.zeros((4,1))
    for i in range(0, len(xs1)):
        x1 = xs1[i]
        x2 = xs2[i]
        xcur = np.ones((4,1))
        xcur[0:2] = x1.reshape(2,1)
        xcur[2:4] = x2.reshape(2,1)
        while True:
            x1Tri = np.array((xcur[0,0], xcur[1,0], 1))
            x2Tri = np.array((xcur[2,0], xcur[3,0], 1))
            delUp = np.dot(fundMat.T, x1Tri)
            delUp = delUp / delUp[-1] # TODO: need normalization?
            delDn = np.dot(fundMat,   x2Tri)
            delDn = delDn / delDn[-1]

            numer = np.dot(np.dot(x2Tri.T, fundMat), x1Tri)
            denum = np.dot(delUp.ravel(), delUp.ravel()) + np.dot(delDn.ravel(), delDn.ravel())
            delta[0] = delUp[0]
            delta[1] = delUp[1]
            delta[2] = delDn[0]
            delta[3] = delDn[1]
            delta *= numer / denum
            deltaNorm = LA.norm(delta)
            xsEpsilon = 0.001
            if deltaNorm < xsEpsilon:
                break
            xcur -= delta

            print("i={0} delta={1} xcur={2}".format(i, delta.ravel(), xcur.ravel()))
        #
        result.append(xcur)
    return result

# MVGCV page 318
# Implemented by solving polynomial of degree six.
def correctPointCorrespondencePoly6(fundMat, xs1, xs2):
    fixedXs1 = []
    fixedXs2 = []
    homog = (len(xs1[0]) == 3)
    for i in range(0, len(xs1)):
        x1 = xs1[i]
        x2 = xs2[i]

        fixX1, fixX2 = correctPointCorrespondencePoly6_OnePoint(fundMat, x1, x2)
        fixX1 = fixX1 / fixX1[-1]
        fixX2 = fixX2 / fixX2[-1]
        if homog:
            fixedXs1.append((fixX1[0], fixX1[1], 1))
            fixedXs2.append((fixX2[0], fixX2[1], 1))
        else:
            fixedXs1.append((fixX1[0], fixX1[1]))
            fixedXs2.append((fixX2[0], fixX2[1]))
    return np.array(fixedXs1), np.array(fixedXs2)

# MVGCV page 318
def correctPointCorrespondencePoly6_OnePoint(fundMat, x1, x2):
    assert len(x1) == 2 or (len(x1) == 3 and x1[2] == 1)

    # construct translations that take x1 and x2 to the origin
    trans1 = np.eye(3)
    trans1[0,2] = -x1[0]
    trans1[1,2] = -x1[1]

    trans2 = np.eye(3)
    trans2[0,2] = -x2[0]
    trans2[1,2] = -x2[1]

    transl1Inv = LA.inv(trans1)
    transl2Inv = LA.inv(trans2)
    fundMat2 = np.dot(np.dot(transl2Inv.T, fundMat), transl1Inv)

    # find epipoles (epi1=null(F), epi2=null(F.transpose))
    (dVec, u, vt) = cv2.SVDecomp(fundMat2) # NOTE: search for epipoles using F2
    epi1 = vt.T[:, -1]
    epipoleScaleFactor = lambda epi: 1 / (epi[0]**2 + epi[1]**2) # scale epipole so that (epi[0]^2+epi[1]^2==1)
    epi1 = epi1 * epipoleScaleFactor(epi1)

    (dVec, u, vt) = cv2.SVDecomp(fundMat2.T)
    epi2 = vt.T[:, -1]
    epi2 = epi2 * epipoleScaleFactor(epi1)

    # construct rotations to take epipole into (1,0,f) coordinate
    rota1 = np.eye(3)
    rota1[0,0] = rota1[1,1] = epi1[0]
    rota1[0,1] = epi1[1]
    rota1[1,0] = -epi1[1]

    rota2 = np.eye(3)
    rota2[0,0] = rota2[1,1] = epi2[0]
    rota2[0,1] = epi2[1]
    rota2[1,0] = -epi2[1]

    # result fundMat must have the form formula 12.3 page 317
    fundMat3 = np.dot(np.dot(rota2, fundMat2), rota1.T)
    #print("fundMat transformed=\n{0}".format(fundMat3))

    a = fundMat3[1,1]
    b = fundMat3[1,2]
    c = fundMat3[2,1]
    d = fundMat3[2,2]
    f1 = epi1[2]
    f2 = epi2[2]
    expectedFundMat = np.array([(f1*f2*d, -f2*c, -f2*d), (-f1*b, a, b), (-f1*d, c, d)])
    #print("expectedFundMat=\n{0}".format(expectedFundMat))
    # TODO: assertion doesn't work for epipoles np.isclose(10e-5,10e-15)==False
    # assert np.isclose(fundMat3[0,0], f1*f2*d)
    # assert np.isclose(fundMat3[0,1], -f2*c)
    # assert np.isclose(fundMat3[0,2], -f2*d)
    # assert np.isclose(fundMat3[1,0], -f1*b)
    # assert np.isclose(fundMat3[2,0], -f1*d)
    x1Perp, x2Perp = correctPointCorrespondencePoly6_Transformed(a, b, c, d, f1, f2)

    # restore transforms
    x1Res = np.dot(transl1Inv, np.dot(rota1.T, x1Perp))
    x2Res = np.dot(transl2Inv, np.dot(rota2.T, x2Perp))
    # assert x1Res and x2Res conform with fundMat
    epiConstrBef = np.dot((x2[0], x2[1], 1),    np.dot(fundMat, (x1[0], x1[1], 1)))
    epiConstrAft = np.dot(x2Res.T, np.dot(fundMat, x1Res))
    #print("epipolar constraint before={0} after={1}".format(epiConstrBef, epiConstrAft))
    assert np.isclose(0, epiConstrAft, atol=1e-3), "expect result x1 and x2 to satisfy epipolar constraint"
    return (x1Res, x2Res)

# both image points are in the origin (0,0)
# both epipoles are in (1,0,f1) and (1,0,f2) positions
def correctPointCorrespondencePoly6_Transformed(a, b, c, d, f1, f2):
    # g(t) function in the MVGCV page 317
    # terms of the derivative of the error function
    gTerms = np.zeros(7) # from higher to lower degree (6 to 0)
    gTerms[6] = b*d*(b*c-a*d)
    gTerms[5] = b**4 + (b*b*c*c-a*a*d*d) + f2*f2*d*d*(2*b*b + d*d*f2*f2)
    gTerms[4] = 4*a*b*b*b + (b*c-a*d)*(a*c + 2*b*d*f1*f1) + 4*f2*f2*(b*b*c*d + a*b*d*d + c*d*d*d*f2*f2)
    gTerms[3] = 6*a*a*b*b + 2*(b*b*c*c - a*a*d*d)*f1*f1 + f2*f2*(2*b*b*c*c + 8*a*b*c*d + 2*a*a*d*d + 6*c*c*d*d*f2*f2)
    gTerms[2] = 4*a*a*a*b + (b*c-a*d)*f1*f1*(2*a*c + b*d*f1*f1) + 4*f2*f2*(a*b*c*c + a*a*c*d + c*c*c*d*f2*f2)
    gTerms[1] = a**4 + (b*b*c*c-a*a*d*d) * f1**4 + c*c*f2*f2*(2*a*a + c*c*f2*f2)
    gTerms[0] = a*c*(b*c-a*d) * f1**4
    rs = np.roots(gTerms)

    # choose real ts (skip complex ts)
    realTs = [t.real for t in rs if t.imag == 0]
    assert(len(realTs) > 0)

    # =s(t) function in the MVGCV book
    def imageErrFun(a, b, c, d, f1, f2, t):
        return t*t/(1+f1*f1*t*t) + (c*t+d)**2/((a*t+b)**2 + f2*f2*(c*t+d)**2)

    errors = [imageErrFun(a,b,c,d,f1,f2,realT) for realT in realTs]
    minInd = np.argmin(errors)

    # T with min error
    tmin = realTs[minInd]

    # check t=Inf
    errTinf = 1/(f1*f1) + c*c/(a*a+f2*f2*c*c)
    if errTinf < errors[minInd]:
        assert False, "TODO: how to process TInf?"

    #
    line1 = (tmin*f1, 1, -tmin)
    line2 = (-f2*(c*tmin+d), a*tmin+b, c*tmin+d)

    # put perpendicular from origin to each line
    def perpOriginToLine(line):
        return (-line[0]*line[2], -line[1]*line[2], line[0]*line[0] + line[1]*line[1])

    x1Perp = perpOriginToLine(line1)
    x2Perp = perpOriginToLine(line2)
    return x1Perp, x2Perp

# line(3x1)
def lineSegment(line, imgWidth):
    # lineY = -(line(1)*xi+line(3))/line(2);
    def lineY(x): return -(line[0] * x + line[2]) / line[1]

    yLeft = lineY(0)
    yRight = lineY(imgWidth - 1)

    return ((0, yLeft), (imgWidth - 1, yRight))

# computes H so that
# p1 = p1*H
# p2 = p2*H
# TODO: doesn't work
def computeP1P2Transform(p1,p2,p1New,p2New):
    h,w = p1.shape
    A = np.zeros((h*2, w))
    B = np.zeros((h*2, 1))
    result = np.zeros((w,w))

    for colInd in range(0, w):
        # upper part
        A[0:h, :] = p1
        B[0:h, 0] = p1New[:, colInd]
        # lower part
        A[h:(2*h), :] = p2
        B[h:(2*h), 0] = p2New[:, colInd]

        suc, column = cv2.solve(A, B, None, cv2.DECOMP_SVD)
        if not suc:
            return False, None
        result[:,colInd] = column.ravel()

    testPostcondition = True
    if testPostcondition:
        p1Test = np.dot(p1, result)
        p2Test = np.dot(p2, result)

        # p1Test should be equal p1New
        p1Test = p1Test / p1Test[-1,-1] * p1New[-1,-1]
        p2Test = p2Test / p2Test[-1,-1] * p2New[-1,-1]
        print("p1Test=\n{0}".format(p1Test))
        print("p2Test=\n{0}".format(p2Test))

    return True, result

def triangAndReprojectCorrespondences(p1, p2, xs1, xs2, calc2dReproj):
    xs3d = []
    reprojXs1 = []
    reprojXs2 = []
    for ptInd in range(0, len(xs1)):
        pt1 = xs1[ptInd]
        pt2 = xs2[ptInd]
        X = triangulateDlt(p1, p2, pt1, pt2)
        xs3d.append((X[0], X[1], X[2]))
        if not calc2dReproj:
            print("ind={0} x1={1} x2={2} X={3}".format(ptInd, pt1, pt2, X))

        if calc2dReproj:
            pt1Re = np.dot(p1, X)
            pt1Re = pt1Re / pt1Re[-1]
            pt2Re = np.dot(p2, X)
            pt2Re = pt2Re / pt2Re[-1]
            print("ind={0} x1={1} x2={2} X={3} x1Re={4} x2Re={5}".format(ptInd, pt1, pt2, X, pt1Re, pt2Re))

            reprojXs1.append(pt1Re)
            reprojXs2.append(pt2Re)
    return (xs3d, reprojXs1, reprojXs2)

def mapHomography(xs, H, normHomog):
    h,w = H.shape
    assert(h == 3)
    assert(w == 3)

    ysResult = []
    for pt in xs:
        ptTri = pt if len(pt) == 3 else (pt[0], pt[1], 1)
        y = np.dot(H, ptTri)
        if normHomog:
            y = y / y[-1]
        ysResult.append(y)
    return ysResult

# Constructs homography from three vanishing points in two images.
def constructHomogResult136(fundMat, epi2, M1, M2):
    A = np.dot(skewSymmeticMat(epi2), fundMat)

    # init B
    B = np.zeros((3,1))
    for i in range(0, 3):
        x1 = M1[i, :]
        # x2homo = np.dot(homogCV, x1)
        # x2homo = x2homo / x2homo[-1]
        x2 = M2[i, :]

        # constraints
        checkConstraints = True
        if checkConstraints:
            c1 = np.cross(epi2, x2)
            c1 = c1 / c1[-1]
            c2 = np.dot(fundMat, x1)
            c2 = c2 / c2[-1]
            print("constraints={0} and {1}".format(c1, c2))

        # par1=par2*b, b=const; par1,par2=vectors
        par1 = np.cross(x2, np.dot(A, x1))
        par2 = np.cross(x2, epi2)
        b = np.dot(par1, par2) / np.dot(par2, par2)
        B[i] = b
        print("b={0}".format(b))

    # M must be nonsingualar
    dVec,u,vt = cv2.SVDecomp(M1)
    suc, v = cv2.solve(M1, B, None, cv2.DECOMP_SVD)
    assert suc

    H = A - np.dot(epi2.reshape(3,1), v.reshape(1,3))
    H = H / H[-1, -1] # NOTE: with this normalization, affine reconstruction may not work TODO: true???
    return H

def calcImageOfAbsConicW_ZeroSkewSquarePixels(xs):
    pass

# Computes IAC [3x3] matrix with zero skew and specified image size and field of view.
# fovXDeg = camera's field of view angle in horizontal plane, in degrees
def imageOfAbsConic_AllDefault(width, height, fovXDeg, fovYDeg):
    halfWidth, halfHeight = width / 2.0, height / 2.0
    fx = halfWidth  / math.tan(np.deg2rad(fovXDeg) / 2)
    fy = halfHeight / math.tan(np.deg2rad(fovYDeg) / 2)

    w = np.eye(3)
    w[0,0] = fy*fy
    w[0,2] = w[2,0] = -fy*fy*halfWidth
    w[1,1] = fx*fx
    w[1,2] = w[2,1] = -fy*fy*halfHeight
    w[2,2] = fx*fx*fy*fy + halfWidth**2 * fy*fy + halfHeight**2 * fx*fx
    w = w / w[-1,-1] # without normalization, the 3d reconstructed points are too big, order of 10^4
    return w

# Computes IAC [3x3] matrix.
# v1 and v2 are two vanishing points, corresponding to orthogonal directions.
# constraints: zero skew, specified image center (x0,y0), square pixels
# unknown: ax(=ay)
def imageOfAbsConic_DefaultCenterSquarePixels(width, height, v1, v2):
    x0, y0 = width / 2.0, height / 2.0
    right = v1[0]*v2[0]+v1[1]*v2[1] - x0*(v1[2]*v2[0]+v1[0]*v2[2])-y0*(v1[2]*v2[1]+v1[1]*v2[2])+v1[2]*v2[2]*(x0**2+y0**2)
    ax2 = -right / (v1[2]*v2[2])

    w = np.eye(3)
    w[0,2] = w[2,0]=-x0
    w[1,2] = w[2,1]=-y0
    w[2,2] = ax2 + x0**2 + y0**2
    w = w / w[-1,-1] # without normalization, the 3d reconstructed points are too big, order of 10^4
    return w

# Computes IAC [3x3] matrix.
# orthoVanishPnts = list of pairs of vanishing points, corresponding to orthogonal directions.
# orthoVanishPnts[i] = (v1,v2); v1 and v2 are orthogonal vanishing points
# constraints: zero skew, specified image center (x0,y0)
# unknown: ax, ay
def imageOfAbsConic_DefaultCenter2(width, height, orthoVanishPnts):
    assert len(orthoVanishPnts) == 2
    x0, y0 = width / 2.0, height / 2.0

    pair = orthoVanishPnts[0]
    v11,v12,v13 = pair[0]
    v21,v22,v23 = pair[1]
    pair = orthoVanishPnts[1]
    v31,v32,v33 = pair[0]
    v41,v42,v43 = pair[1]
    ax2 = (v12*(v31 - v33*x0)*(-v41 + v43*x0)*(v22 - v23*y0) +
       v11*(v21 - v23*x0)*(v32 - v33*y0)*(v42 - v43*y0) +
       v13*(v22*(v31 - v33*x0)*(v41 - v43*x0)*y0 +
          v21*x0*(v32 - v33*y0)*(-v42 + v43*y0) +
          v23*(v32*x0**2*(v42 - v43*y0) +
             y0*(v31*(-v41 + v43*x0)*y0 +
                v33*x0*(-(v42*x0) + v41*y0)))))/(v12*v33*
        v43*(v22 - v23*y0) +
       v13*(-(v22*v33*v43*y0) +
          v23*(-(v32*v42) + v33*v42*y0 + v32*v43*y0)))
    ay2 = (v12*(v31 - v33*x0)*(v41 - v43*x0)*(v22 - v23*y0) + 
   v11*(v21 - v23*x0)*(v32 - v33*y0)*(-v42 + v43*y0) + 
   v13*(v22*(v31 - v33*x0)*(-v41 + v43*x0)*y0 + 
      v21*x0*(v32 - v33*y0)*(v42 - v43*y0) + 
      v23*(v32*x0**2*(-v42 + v43*y0) + 
         y0*(v31*(v41 - v43*x0)*y0 + 
            v33*x0*(v42*x0 - v41*y0)))))/(v11*v33*
    v43*(v21 - v23*x0) + 
   v13*(-(v21*v33*v43*x0) + 
      v23*(-(v31*v41) + v33*v41*x0 + v31*v43*x0)))

    ok = ax2 > 0 and ay2 > 0
    if not ok:
        return None

    w = np.eye(3)
    w[0,0] = ay2
    w[0,2] = w[2,0] = -ay2*x0
    w[1,1] = ax2
    w[1,2] = w[2,1] = -ax2*y0
    w[2,2] = ax2*ay2 + ay2*x0**2 + ax2*y0**2
    w = w / w[-1,-1] # without normalization, the 3d reconstructed points are too big, order of 10^4
    return w

# Computes IAC [3x3] matrix with zero skew and specified image size and field of view.
# (zero skew, square pixels)
# vs1[i] and vs2[i] are two vanishing points, corresponding to orthogonal directions.
# Direction (corresponding to the vanishing line 'v') and plane (corresponding to line at infinity 'line') are orthogonal.
# TODO: method doesn't work; seems, vy1 or line have big measurement error
# constraints: zero skew, square pixels
# unknown: ax, x0, y0
def imageOfAbsConic_OrthoLineAndPlane(width, height, orthoVanishPnts, orthoVanishLines):
    A = []
    b = []

    # orhogonal directions defined by two vanishing points
    for i in range(0, len(orthoVanishPnts)):
        v1 = orthoVanishPnts[i][0]
        v2 = orthoVanishPnts[i][1]
        v1 = v1 / LA.norm(v1)
        v2 = v2 / LA.norm(v2)
        data = np.zeros(3)
        data[0] = v1[2]*v2[0] + v1[0]*v2[2]
        data[1] = v1[2]*v2[1] + v1[1]*v2[2]
        data[2] = v1[2]*v2[2]
        right = -(v1[0]*v2[0] + v1[1]*v2[1])
        A.append(data)
        b.append(right)

    # orhogonal line and plane
    for i in range(0, len(orthoVanishLines)):
        v = orthoVanishLines[i][0]
        line = orthoVanishLines[i][1]
        v = v / LA.norm(v)
        line = line / LA.norm(line) # NOTE: normalization is mandatory for correct result
        # each pair contributes two independent equations
        data = np.zeros(3)
        # data[0] = v[2]
        # data[3] = -line[0]
        # right = - v[0]
        data[0] = line[1]*v[0]
        data[1] = line[1]*v[1]-line[2]*v[2]
        data[2] = line[1]*v[2]
        right = line[2]*v[1]
        A.append(data)
        b.append(right)
        #
        data = np.zeros(3)
        # data[1] = v[2]
        # data[3] = -line[1]
        # right = - v[1]
        data[0] = -line[0]*v[0]+line[2]*v[2]
        data[1] = -line[0]*v[1]
        data[2] = -line[0]*v[2]
        right = -line[2]*v[0]
        A.append(data)
        b.append(right)

        data = np.zeros(3)
        # data[0] = v[0]
        # data[1] = v[1]
        # data[2] = v[2]
        # data[3] = -line[2]
        # right = 0
        data[0] = -line[1]*v[2]
        data[1] = line[0]*v[2]
        data[2] = 0
        right = -line[0]*v[1]+line[1]*v[0]
        A.append(data)
        b.append(right)

    AMat = np.array(A, dtype=np.float64)
    BMat = np.array(b, dtype=np.float64)
    suc, res = cv2.solve(AMat, BMat, None, cv2.DECOMP_SVD)

    # res = (a,b,c,k)
    testRes = False
    if testRes and len(orthoVanishPnts) > 0 and len(orthoVanishLines) > 0:
        v11,v12,v13 = orthoVanishPnts[0][0]
        v21,v22,v23 = orthoVanishPnts[0][1]
        v1,v2,v3 = orthoVanishLines[0][0]
        l1,l2,l3 = orthoVanishLines[0][1]
        a = -((-(l2*v1*v13*v2*v23) + l1*v13*v2**2*v23 + l2*v1*v13*v22*v3 -
          l1*v13*v2*v22*v3 + l2*v1*v12*v23*v3 + l3*v1*v13*v23*v3 -
          l1*v12*v2*v23*v3 + l1*v11*v21*v3**2 +
          l1*v12*v22*
           v3**2)/(v3*(-(l1*v1*v13*v23) - l2*v13*v2*v23 + l1*v13*v21*v3 +
            l2*v13*v22*v3 + l1*v11*v23*v3 + l2*v12*v23*v3 +
            l3*v13*v23*v3)))
        b = (-(l2*v1**2*v13*v23) + l1*v1*v13*v2*v23 + l2*v1*v13*v21*v3 -
           l1*v13*v2*v21*v3 + l2*v1*v11*v23*v3 - l1*v11*v2*v23*v3 -
           l3*v13*v2*v23*v3 - l2*v11*v21*v3**2 -
           l2*v12*v22*
           v3**2)/(v3*(-(l1*v1*v13*v23) - l2*v13*v2*v23 + l1*v13*v21*v3 +
           l2*v13*v22*v3 + l1*v11*v23*v3 + l2*v12*v23*v3 + l3*v13*v23*v3))
        c = -((l2*v1*v2 - l1*v2**2 - l3*v1*v3)/(l1*
           v3**2)) + ((l1*v1 + l2*v2 - l3*v3)*(-(l2*v1*v13*v2*v23) +
           l1*v13*v2**2*v23 + l2*v1*v13*v22*v3 - l1*v13*v2*v22*v3 +
           l2*v1*v12*v23*v3 + l3*v1*v13*v23*v3 - l1*v12*v2*v23*v3 +
           l1*v11*v21*v3**2 + l1*v12*v22*v3**2))/(l1*
         v3**2*(-(l1*v1*v13*v23) - l2*v13*v2*v23 + l1*v13*v21*v3 +
           l2*v13*v22*v3 + l1*v11*v23*v3 + l2*v12*v23*v3 +
           l3*v13*v23*v3))
        k = -((-(v1**2*v13*v23) - v13*v2**2*v23 +
          v1*v13*v21*v3 + v13*v2*v22*v3 + v1*v11*v23*v3 + v12*v2*v23*v3 -
          v11*v21*v3**2 - v12*v22*v3**2)/(l1*v1*v13*v23 + l2*v13*v2*v23 -
          l1*v13*v21*v3 - l2*v13*v22*v3 - l1*v11*v23*v3 - l2*v12*v23*v3 -
          l3*v13*v23*v3))
    #res = (a,b,c,k)

    w = np.eye(3)
    w[0,2] = w[2,0]=res[0]
    w[1,2] = w[2,1]=res[1]
    w[2,2] = res[2]
    w = w / w[-1,-1] # without normalization, the 3d reconstructed points are too big, order of 10^4
    return w

# H[3x3] inifinite homography
def imageOfAbsConic_SameCameraOrthoVanish(H, orthoVanishPnts):
    A = []
    UnkCount = 6
    # orhogonal directions defined by two vanishing points
    for i in range(0, len(orthoVanishPnts)):
        v1 = orthoVanishPnts[i][0]
        v2 = orthoVanishPnts[i][1]
        # v1 = v1 / LA.norm(v1) # TODO: this normalization makes results worse
        # v2 = v2 / LA.norm(v2)
        # v1 = v1 / v1[-1] # this normalization doesn't worsen results
        # v2 = v2 / v2[-1]
        w11 = v1[0]*v2[0]
        w12 = v1[1]*v2[0] + v1[0]*v2[1]
        w13 = v1[2]*v2[0] + v1[0]*v2[2]
        w22 = v1[1]*v2[1]
        w23 = v1[2]*v2[1] + v1[1]*v2[2]
        w33 = v1[2]*v2[2]
        data = np.array([w11, w12, w13, w22, w23, w33])
        A.append(data)

    # same camera in both images constraint
    # w'=Hinv.T*w*Hinv
    Hinv = LA.inv(H)
    h11,h12,h13,h21,h22,h23,h31,h32,h33 = Hinv.reshape(9)
    # h99 = np.kron(Hinv, Hinv).T
    # h99Left = h99 - np.eye(9) # h99Left * W[9,1] = 0
    if True:
        # -1 at diagonal to bring w' on other side of equation
        A.append(np.array([h11**2-1, 2*h11*h21, 2*h11*h31, h21**2, 2*h21*h31, h31**2])) # w11
        A.append(np.array([h11*h12, h12*h21+h11*h22-1, h12*h31+h11*h32,  h21*h22, h22*h31+h21*h32, h31*h32])) # w12
        A.append(np.array([h11*h13, h13*h21+h11*h23, h13*h31+h11*h33-1, h21*h23, h23*h31+h21*h33, h31*h33])) # w13
        A.append(np.array([h12**2, 2*h12*h22, 2*h12*h32, h22**2-1, 2*h22*h32, h32**2])) # w22
        A.append(np.array([h12*h13, h13*h22+h12*h23, h13*h32+h12*h33, h22*h23, h23*h32+h22*h33-1, h32*h33])) # w23
        A.append(np.array([h13**2, 2*h13*h23, 2*h13*h33, h23**2, 2*h23*h33, h33**2-1])) # w33

    HMat = np.array(A, dtype=np.float64)
    #HMat = np.vstack((HMat, h99Left))

    n = HMat.shape[0]
    suc,resWUp = cv2.solve(HMat, np.zeros((n,1)), None, cv2.DECOMP_SVD)
    assert suc

    dVec,u,vt = cv2.SVDecomp(HMat)
    resWUp2 = vt.T[:,-1]
    w11,w12,w13,w22,w23,w33=resWUp2
    w = np.array([[w11,w12,w13],[w12,w22,w23],[w13,w23,w33]])

    debugError = True
    if debugError:
        for i in range(0, len(orthoVanishPnts)):
            val = np.dot(orthoVanishPnts[i][0],np.dot(w, orthoVanishPnts[i][1]))
            print("expected=0 actual={0}".format(val))
        wAct = np.dot(Hinv.T, np.dot(w, Hinv))
        wUpDiag = (wAct[0,0],wAct[0,1],wAct[0,2],wAct[1,1],wAct[1,2],wAct[2,2])
        for i in range(0, len(wUpDiag)):
            print("w items: expected=0 actual={0}".format(resWUp2[i]-wUpDiag[i]))
    return w

# calculate affine->metric transformation
def calcAffineToMetricTransform(pAff, w):
    M = pAff[:, 0:3]
    triple = np.dot(np.dot(np.transpose(M), w), M)
    suc, doubleA = cv2.invert(triple)
    assert suc
    #print("AA={0}".format(doubleA))

    A = LA.cholesky(doubleA) # doubleA=A*At
    #print("A={0}".format(A))

    hAffToMetricInv = np.eye(4)
    hAffToMetricInv[0:3, 0:3] = A
    return hAffToMetricInv

# Decomposes F so that [epi]*A=F and returns A.
# OBSOLETE: use A = [epi]*F
def decomposeFundMat(epi, fundMat):
    # decompose F
    eMat = np.zeros((9,9))
    eMat[0,3] = -epi[2]
    eMat[0,6] = epi[1]
    eMat[1,4] = -epi[2]
    eMat[1,7] = epi[1]
    eMat[2,5] = -epi[2]
    eMat[2,8] = epi[1]
    eMat[3,0] = epi[2]
    eMat[3,6] = -epi[0]
    eMat[4,1] = epi[2]
    eMat[4,7] = -epi[0]
    eMat[5,2] = epi[2]
    eMat[5,8] = -epi[0]
    eMat[6,0] = -epi[1]
    eMat[6,3] = epi[0]
    eMat[7,1] = -epi[1]
    eMat[7,4] = epi[0]
    eMat[8,2] = -epi[1]
    eMat[8,5] = epi[0]
    fMat = fundMat.ravel()
    suc, aVec = cv2.solve(eMat, fMat, None, cv2.DECOMP_SVD)
    assert suc
    aMat = aVec.reshape((3,3))
    return aMat

def affineReconstruct3V(fundMat, p1Proj, p2Proj, vanish1, vanish2, epi1, epi2):
    # implicit homography
    # TODO: implicit method doesn't work (doesn't produce any reconstruction)
    vanishXs1 = np.vstack((vanish1, epi1))
    vanishXs2 = np.vstack((vanish2, epi2))
    assert np.all([vanishXs1[:,-1]==1])
    Himplicit = findHomogDlt(vanishXs1[:,0:2], vanishXs2[:,0:2])
    Himplicit = Himplicit / LA.norm(Himplicit[-1,:])
    print("Himplicit=\n{0}".format(Himplicit))
    if True:
        for i in range(len(vanish1)):
            v2Map = np.dot(Himplicit, vanish1[i])
            v2Map = v2Map / v2Map[-1]
            print("v1~v2: expect v2={0} actual v2={1}".format(vanish2[i], v2Map))

    # triangulated 3d points
    vanish3D = [triangulateDlt(p1Proj, p2Proj, vanish1[i,:], vanish2[i,:]) for i in range(0,len(vanish1))]
    assert vanish3D[0][3] == 1, "Must be normalized 3D point"
    #vanish3D.append(np.array([0,1,0]))
    pt1 = vanish3D[0][0:3]
    threePnts = True
    if threePnts:
        pt2 = vanish3D[1][0:3]
        pt3 = vanish3D[2][0:3]
        v1 = pt1-pt3
        v2 = pt2-pt3
        # v1 = pt1-pt2
        # v2 = np.array([0,1,0])
        planeNorm = np.cross(v1, v2)
        planeNorm = planeNorm / LA.norm(planeNorm)
        # formula in MVGCV Example 3.1 page 67
        # dCoef2 = -np.dot(pt3, np.cross(pt1, pt2))
    else:
        # corridor
        planeNorm = (0,0,1)
    dCoef = -np.dot(planeNorm, pt1)
    assert abs(dCoef) > 0.001, "TODO: MVGCV Ch10.4.1 Final coordinate of plane at inf is zero"

    # NOTE: normalize plane at infinity so that last coef=1 (source: MVGCV Result 13.1)
    planeNorm = planeNorm / dCoef
    dCoef = 1

    hInf3D = p2Proj[:,0:3] - np.dot(p2Proj[:,-1].reshape((3,1)), planeNorm.reshape((1,3)))
    hInf3D = hInf3D / LA.norm(hInf3D[-1,:])
    print("hInf3D=\n{0}".format(hInf3D))
    if True:
        for i in range(len(vanish1)):
            v2Map = np.dot(hInf3D, vanish1[i])
            v2Map = v2Map / v2Map[-1]
            print("v1~v2: expect v2={0} actual v2={1}".format(vanish2[i], v2Map))

    reflectPlane3D = np.hstack((planeNorm, dCoef))
    #reflectPlane3D = reflectPlane3D / reflectPlane3D[-1]
    print("reflectPlane3D={0}".format(reflectPlane3D))
    #print("reflectPlaneNorm={0}".format(reflectPlane3D[0:3]/LA.norm(reflectPlane3D[0:3])))

    # construct proj->affine transformation
    Hpm1 = np.eye(4, 4, dtype=np.float64)
    Hpm1[-1, :] = reflectPlane3D
    dVec2, u2, vt2 = cv2.SVDecomp(Hpm1)

    suc, Hp = cv2.invert(Hpm1)

    p1AffineExpl = np.dot(p1Proj, Hp)
    p2AffineExpl = np.dot(p2Proj, Hp)
    p2AffineExpl = p2AffineExpl / LA.norm(p2AffineExpl[-1,0:3])
    print("p1AffineExpl3D=\n{0}".format(p1AffineExpl))
    print("p2AffineExpl3D=\n{0}".format(p2AffineExpl))
    #print("p2AffineExpl3DNorm={0}".format(p2AffineExpl[-1,0:3]/LA.norm(p2AffineExpl[-1,0:3])))

    # explicit reconstruction without actually recovering 3D coordinates of vanishing points
    # works OK
    # MVGCV, Result 13.6 page 331
    Hexpl2 = constructHomogResult136(fundMat, epi2, vanish1, vanish2)
    print("Hexpl2=\n{0}".format(Hexpl2))
    if True:
        for i in range(len(vanish1)):
            v2Map = np.dot(Hexpl2, vanish1[i])
            v2Map = v2Map / v2Map[-1]
            print("v1~v2: expect v2={0} actual v2={1}".format(vanish2[i], v2Map))

    useH = False
    if useH:
        #H = Himplicit
        #H = hInf3D
        H = Hexpl2
        p1Affine = np.hstack((np.eye(3), np.zeros((3,1))))
        p2Affine = np.hstack((H, epi2.reshape(3,1)))
    else:
        p1Affine = p1AffineExpl
        p2Affine = p2AffineExpl
        H = hInf3D
    print("p1Affine=\n{0}".format(p1Affine))
    print("p2Affine=\n{0}".format(p2Affine))
    print("p2AffineNorm={0}".format(p2Affine[-1,0:3]/LA.norm(p2Affine[-1,0:3])))

    # check projective to affine upgrade matrix
    # Hp^-1 * X = (x,y,z,0) -- transforms vanishing points to coord at infinity from the world's center
    for i in range(0, len(vanish3D)):
        X = np.dot(Hpm1, vanish3D[i])
        assert X[3] < 0.001, "Vanishing points in projective space must have zero W in affine space"

    # TODO: triangulation and then get plane at inf or Result136 - both seems to get correct result!
    # TODO: why normal of plane at infinity becomes (0,0,1)
    return p1Affine, p2Affine, H


def RunIterateSecondImageAndReconstruct():
    img1 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_0.png")
    for i2 in range(1, 7):
        img2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_{0}.png".format(i2))
        ess_mat, xs1_meter, xs2_meter = MatchKeypointsAndGetEssentialMatNarrowBaseline(cam_mat, img1, img2)

        suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat, xs1_meter, xs2_meter)
        if suc:
            ess_T = skewSymmeticMat(ess_Tvec)
            ess_wvec, ess_wang = logSO3(ess_R)
            print("R|T: w={0} ang={1}\n{2}\n{3}".format(ess_wvec, math.degrees(ess_wang), ess_R, ess_Tvec))
        else:
            print("Failed i2:{0}".format(i2))

# x is directed to the right-top
# y is directed to the bottom
# z is directed to the left-top
def GenerateTwoOrthoChess(debug=0):
    h = 0.18 / 8
    d = w = h
    xs3D = np.float32([
        [0, -4, 8], # line1 - top
        [0, -4, 4],
        [0, -4, 0],
        [4, -4, 0],
        [8, -4, 0],
        [0, -2, 6], # line2
        [0, -2, 2],
        [2, -2, 0],
        [6, -2, 0],
        [0, 0, 8], # line3 - central
        [0, 0, 4],
        [0, 0, 0], # central point
        [4, 0, 0],
        [8, 0, 0],
        [0, 2, 6], # line4
        [0, 2, 2],
        [2, 2, 0],
        [6, 2, 0],
        [0, 4, 8],  # line5 - bottom
        [0, 4, 4],
        [0, 4, 0],
        [4, 4, 0],
        [8, 4, 0],
    ])
    xs3D = np.vstack((xs3D[:,0]*w, xs3D[:,1]*h, xs3D[:,2]*d)).T
    if debug >= 3: print("xs3D={0}".format(xs3D))

    # 1=first side, 2=second side, 3=first or second side
    side_mask = np.int32([
        1,1,3,2,2,
        1,1,2,2,
        1,1,3,2,2,
        1,1,2,2,
        1, 1, 3, 2, 2,
    ])

    cell_width = h

    return xs3D, cell_width, side_mask

class ImageFeatureTracker:
    """ Detects and tracks features in the image """
    def __init__(self, min_num_3Dpoints):
        self.kpd = cv2.xfeatures2d.SIFT_create(nfeatures=self.min_num_3Dpoints)  # OpenCV-contrib-3
        self.lk_win_size = (21, 21)

    def DetectFeats(self, img_ind, img_gray):
        kp1 = self.kpd.detect(img_gray, None)
        # OpenCV cv2.xfeatures2d.SIFT_create.detect may return duplicates, - remove them
        xs2_pixel = np.array(sorted(set([kp.pt for kp in kp1])), np.float32)
        map_point_ids = None
        return map_point_ids, xs2_pixel

    def TrackFeats(self, img1_ind, img1_gray, img2_ind, img2_gray, xs1_map_point_ids, xs1_pixel):
        xs1_pixel_array = np.array(xs1_pixel, np.float32)
        next_pts_old, status_old, err_old = cv2.calcOpticalFlowPyrLK(self.img1_gray, img2_gray, xs1_pixel_array, None, winSize=self.lk_win_size)
        FixOpenCVCalcOpticalFlowPyrLK(xs1_pixel, next_pts_old, status_old, self.lk_win_size)
        return next_pts_old, status_old


class VirtualImageFeatureTracker:
    def __init__(self, min_num_3Dpoints):
        self.img_ind_to_detected_2d_points = {}

    def DetectFeats(self, img_ind, img_gray):
        xs_objs = self.img_ind_to_detected_2d_points[img_ind]
        map_point_ids, xs_pixel = unzip(xs_objs)
        return map_point_ids, xs_pixel

    def SetImage2DPoints(self, img_ind, xs_pixel_clipped):
        self.img_ind_to_detected_2d_points[img_ind] = xs_pixel_clipped

    def TrackFeats(self, img1_ind, img1_gray, img2_ind, img2_gray, xs1_map_point_ids, xs1_pixel):
        xs2_objs = self.img_ind_to_detected_2d_points[img2_ind]

        N = len(xs1_map_point_ids)
        status = np.zeros(N, np.uint8)
        next_pts = [None] * N
        for i,map_point_id1 in enumerate(xs1_map_point_ids):
            # find this feature in the second image
            match_pixel = None
            for map_point_id2, x2_pixel in xs2_objs:
                if map_point_id1 == map_point_id2:
                    match_pixel = x2_pixel
                    break

            stat = 0 if match_pixel is None else 1
            status[i] = stat
            next_pts[i] = match_pixel
        return next_pts, status


def BuildPointInFrameRelatonMatrix(pnt_life_list, sort_points=False):
    """ Build Point(horizontal OX) and Frame (vertical OY) image, 0(black)=foreground, 255(white)=background """
    points_count = len(pnt_life_list) # width, columns
    frames_count = 0 # height, rows
    if points_count > 0:
        frames_count = len(pnt_life_list[0].points_list_meter)

    pnt_life_sorted = pnt_life_list

    # align life points (columns) to the left
    if sort_points:
        def CmpTwoPointTracks(p1, p2):
            for p1_pnt2D, p2_pnt2D in zip(p1.points_list_meter, p2.points_list_meter):
                if p1_pnt2D is None and not p2_pnt2D is None:
                    return 1
                if not p1_pnt2D is None and p2_pnt2D is None:
                    return -1
                # both non-null or both null
                continue
            return 0

        pnt_life_sorted = sorted(pnt_life_list, key=functools.cmp_to_key(CmpTwoPointTracks))

    pnt_frame = np.zeros((frames_count, points_count), np.uint8)
    pnt_frame.fill(255) # background
    for pnt_ind, pnt_life in enumerate(pnt_life_sorted):
        for frame_ind, pos2D in enumerate(pnt_life.points_list_meter):
            if not pos2D is None:
                pnt_frame[frame_ind, pnt_ind] = 0 # foreground
    return pnt_frame

# Represents point across multiple frame images.
class PointLife:
    def __init__(self):
        self.virtual_feat_id = None # the identifier created for map point in a virtual world TODO: rename into map_point_virtual_id
        self.track_id = None # the identifier associated with a track  TODO: rename into map_point_track_id
        # ids vary in such a way:
        # x) if a track is lost, another track_id may be assiciated with the same map point later
        # x) there may be map point, but a track is not associated because the tracker is overloaded

        self.inception_frame_ind = None # the frame when tracker decided to create a tracked point, may be later than the first time, the point was observed
        self.start_frame_ind = None # the first frame when position of a point is available
        self.last_frame_ind = None # the last+1 frame when position of a point is available
        self.points_list_meter = []
        self.points_list_pixel = []
        self.is_mapped = False

class PointsWorld:
    def __init__(self):
        self.elem_type = np.float32
        self.use_mpmath = 0 # whether to use mpmath package, see http://mpmath.org/
        self.frame_ind = -1
        self.last_known_pos_frame_ind = 0 # the latest frame with determined [R,T]
        self.num_tracked_points_per_frame = []
        self.points_life = [] # coordinates of 2D points in images
        self.visual_host = None
        self.ground_truth_relative_motion = None
        self.ground_truth_map_pnt_pos = None # gets 3D coordinate of the point in a camera of interest

        # the factor to multiply the ground truth translation to get the scaled distances, so that the initial camera move has unity length
        self.first_unity_translation_scale_factor = None

        self.img1_bgr = None
        self.img1_gray = None
        self.img1_ind = None
        self.img_trackview_bgr = None

        # try to add new track points if the number of current tracked points is below this threshold
        # some big number means to always try to add new points
        self.min_num_3Dpoints = None
        self.img_feat_tracker = None
        self.cam_mat_pixel_from_meter = None
        self.cam_mat_meter_from_pixel = None # inverse of cam_mat (aka K matrix)
        self.focal_len = None # foucus distance in millimeters
        self.slam_impl = None

        # direct RT converts camera frame into world
        # inverse RT converts world into camera
        # source http://slam-plus-plus.sourceforge.net/documentation20/phunzip.php/documentation_21.zip/d1/dbd/rot3d.html
        self.framei_from_world_RT_list = [] # inverse RT
        self.world_pnts = []
        self.check_drift = False # whether to check divergence of points' coordinates and camera relative motion from ground truth
        self.drift = 3e-1
        self.hack_camera_location = False
        self.hack_world_mapping = False
        self.hack_camera_location_in_batch_refine = False
        self.la_engine = "scipy" # default="scipy", ["opencv", "scipy"]
        # True to pretend that low precision algorithm is actually works in hight precision
        # default = False
        # Generally it is a bad idea to hide the real precision of the algorithm and here is used to test how other
        # parts of the program work with higher precision.
        # eg. if True, algorithm sqrt(f64)->f32 is coerced into sqrt(f64)->f64 by upconverting result from f32 into f64
        self.conceal_lost_precision = False
        self.reproj_err_history = []

    def PointById(self, pnt_id):
        pnt_ind = pnt_id # NOTE: assumes point id == index of the point in the array
        return self.points_life[pnt_ind]

    def SetCamMat(self, cam_mat_pixel_from_meter):
        self.cam_mat_pixel_from_meter = cam_mat_pixel_from_meter
        Km1 = scipy.linalg.inv(cam_mat_pixel_from_meter)
        Km1 = Km1.astype(self.elem_type)
        self.cam_mat_meter_from_pixel = Km1

        # sx=number of pixels per one millimeter
        # pix_len=number of millimieters per one pixel
        # sx = 1/pix_len, if pix_len=0.006mm => sx=166.667
        pix_len_mm = 0.006  # millimieters
        sx = 1 / pix_len_mm
        alpha_x = cam_mat_pixel_from_meter[0, 0]
        alpha_y = cam_mat_pixel_from_meter[1, 1]
        focal_len_x = alpha_x / sx
        focal_len_y = alpha_y / sx
        focal_len = max(focal_len_x, focal_len_y)
        self.focal_len = focal_len

    def PrintStat(self):
        print("allocated map points: {}".format(len(self.points_life)))

    # checks if there is an existent feature point close to the point in question
    def IsCloseToExistingFeatures(self, x_pixel, radius_pixel, targ_frame_ind):
        for pnt_life in self.points_life:
            pix = pnt_life.points_list_pixel[targ_frame_ind]
            if pix is None: continue
            len = LA.norm(pix - x_pixel)

            if len < radius_pixel: return True
        return False

    def __AllocateNewPoints(self, inception_frame_ind, new_pnts_count, virtual_point_ids):
        """map_point_ids virtual ids associated with new points or None"""
        if not virtual_point_ids is None:
            assert new_pnts_count == len(virtual_point_ids)
        track_pnt_ids = []

        frames_count = self.frame_ind + 1
        for i in range(0, new_pnts_count):
            id = len(self.points_life)

            map_point_id = None
            if not virtual_point_ids is None:
                map_point_id = virtual_point_ids[i]
                assert not map_point_id is None

            pnt_life = PointLife()
            pnt_life.inception_frame_ind = inception_frame_ind
            pnt_life.virtual_feat_id = map_point_id
            pnt_life.track_id = id
            pnt_life.points_list_meter = [None] * frames_count
            pnt_life.points_list_pixel = [None] * frames_count

            self.points_life.append(pnt_life)
            track_pnt_ids.append(id)
        return track_pnt_ids

    def __SetPoints2DCoords(self, targ_frame_ind, pnt_ids, points2D_meter_with_gaps, points2D_pixel_with_gaps):
        assert len(pnt_ids) == len(points2D_meter_with_gaps)

        has_pixels = not points2D_pixel_with_gaps is None
        if has_pixels:
            assert len(pnt_ids) == len(points2D_pixel_with_gaps)

        tracked_pnts_count = 0
        lost_pnts_count = 0
        for i, id in enumerate(pnt_ids):
            pnt2_meter = points2D_meter_with_gaps[i]
            pnt2_pixel = points2D_pixel_with_gaps[i] if not points2D_pixel_with_gaps is None else None
            if has_pixels:
                assert pnt2_meter is None and pnt2_pixel is None or not pnt2_meter is None and not pnt2_pixel is None

            pnt_life = self.points_life[id]
            if pnt2_meter is None:
                lost_pnts_count += 1
            else:
                pnt_life.points_list_meter[targ_frame_ind] = pnt2_meter
                pnt_life.points_list_pixel[targ_frame_ind] = pnt2_pixel
                pnt_life.start_frame_ind = min(pnt_life.start_frame_ind, targ_frame_ind) if not pnt_life.start_frame_ind is None else targ_frame_ind
                pnt_life.last_frame_ind = max(pnt_life.last_frame_ind, targ_frame_ind + 1) if not pnt_life.last_frame_ind is None else targ_frame_ind+1
                tracked_pnts_count += 1
        return lost_pnts_count, tracked_pnts_count

    def StartNewFrame(self):
        self.frame_ind += 1

        # reserve space
        for pnt_life in self.points_life:
            pnt_life.points_list_meter.append(None)
            pnt_life.points_list_pixel.append(None)

    def PutNewPoints2D(self, first_coord_frame_ind, inception_frame_ind, points2D_meter, points2D_pixel, virtual_point_ids):
        """
        :param first_coord_frame_ind: the first frame where the point was detected  
        :param inception_frame_ind: the frame when tracker decided to create a tracked point, may be later then it was observed at first 
        """
        track_pnt_ids = self.__AllocateNewPoints(inception_frame_ind, len(points2D_meter), virtual_point_ids)

        # if targ_frame_ind is None: # default to the current frame
        #     targ_frame_ind = self.frame_ind

        lost,tracked = self.__SetPoints2DCoords(first_coord_frame_ind, track_pnt_ids, points2D_meter, points2D_pixel)
        assert lost == 0, "there can't be lost points amont new points"

        return track_pnt_ids

    def PutMatchedPoints2D(self, pnt_ids, points2D_meter_with_gaps, points2D_pixel_with_gaps):
        old_lost, old_tracked  = self.__SetPoints2DCoords(self.frame_ind, pnt_ids, points2D_meter_with_gaps, points2D_pixel_with_gaps)
        return None

    # calculate statistics of points change
    def __CalcMapPointsStat(self):
        matched_pnts_count = 0
        new_pnts_count = 0
        lost_pnts_count = 0
        for pnt_life in self.points_life:
            x = pnt_life.points_list_meter[self.frame_ind]
            x_prev = None
            if self.frame_ind > 0:
                x_prev = pnt_life.points_list_meter[self.frame_ind - 1]
            if not x_prev is None:
                if x is None:
                    lost_pnts_count += 1
                else:
                    if pnt_life.inception_frame_ind == self.frame_ind:
                        new_pnts_count += 1
                    else:
                        matched_pnts_count += 1
            else:
                if not x is None:
                    assert pnt_life.inception_frame_ind == self.frame_ind
                    new_pnts_count += 1

        return matched_pnts_count, new_pnts_count, lost_pnts_count

    def ProcessNextImage(self, img_ind, image_bgr, debug = 0):
        img2_ind = img_ind
        img2_bgr = image_bgr
        img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

        def ShowPntCorresp(title, pnt_ids):
            cons_xs1_pixels = []
            cons_xs2_pixels = []
            for id in pnt_ids:
                pnt_life = self.points_life[id]
                x1 = pnt_life.points_list_pixel[self.frame_ind-1]
                x2 = pnt_life.points_list_pixel[self.frame_ind]
                assert not x1 is None
                assert not x2 is None
                cons_xs1_pixels.append(x1)
                cons_xs2_pixels.append(x2)
            ShowMatches(title, self.img1_bgr, img2_bgr, cons_xs1_pixels, cons_xs2_pixels)

        self.StartNewFrame()

        if self.frame_ind == 0:
            map_point_ids, xs2_pixel = self.img_feat_tracker.DetectFeats(img_ind, img2_bgr)

            # convert pixel to image coordinates (pixel -> meters)
            xs2_meter = []
            convertPixelToMeterPointsNew(self.cam_mat_meter_from_pixel, [xs2_pixel], xs2_meter)
            xs2_meter = xs2_meter[0]

            self.PutNewPoints2D(self.frame_ind, self.frame_ind, xs2_meter, xs2_pixel, map_point_ids)  # initial points
        else:
            # 1. Match OLD features
            # try to find the corner in the next image
            # track old (existing) features
            old_map_pnt_ids = []
            old_track_pnt_ids = []
            xs1_pixel = []
            for pnt_life in self.points_life:
                x1_pixel = pnt_life.points_list_pixel[self.frame_ind-1]
                if not x1_pixel is None:
                    xs1_pixel.append(x1_pixel)
                    old_track_pnt_ids.append(pnt_life.track_id)

                    map_pnt_id = pnt_life.virtual_feat_id
                    old_map_pnt_ids.append(map_pnt_id)


            persist_feat_count_old = 0
            if len(xs1_pixel) > 0:
                next_pts_old, status_old = self.img_feat_tracker.TrackFeats(self.img1_ind, self.img1_gray, img2_ind, img2_gray, old_map_pnt_ids, xs1_pixel)

                persist_feat_count_old = sum([1 for s in status_old if s > 0])

                # convert pixel to image coordinates (pixel -> meters)
                xs2_meter_with_gaps = []
                xs2_pixel_with_gaps = []
                for i in range(0, len(status_old)):
                    x2_pixel = None
                    x2_meter = None
                    if status_old[i]:
                        x2_pixel = next_pts_old[i]
                        x2_meter = convertPointPixelToMeter(x2_pixel, self.cam_mat_meter_from_pixel)
                    xs2_pixel_with_gaps.append(x2_pixel)
                    xs2_meter_with_gaps.append(x2_meter)

                if debug >= 4:
                    xs1_pixel_tmp = [x for s, x in zip(status_old, xs1_pixel) if s > 0]
                    xs2_pixel_tmp = [x for s, x in zip(status_old, next_pts_old) if s > 0]
                    ShowMatches("matches", self.img1_bgr, img2_bgr, xs1_pixel_tmp, xs2_pixel_tmp)

                self.PutMatchedPoints2D(old_track_pnt_ids, xs2_meter_with_gaps, xs2_pixel_with_gaps)

            # 2. Match NEW features
            # check if new features are required
            next_pts_new = []
            status_new = []
            err_new = []
            xs2_new_pixel_no_gaps = []
            xs2_new_meter_no_gaps = []
            if persist_feat_count_old < self.min_num_3Dpoints:
                # try to find more features in the previous frame, which persist in current frame
                new_point_virtual_ids_tmp, xs1_pixel_tmp = self.img_feat_tracker.DetectFeats(self.img1_ind, self.img1_gray)

                # we are interested only in those features, which do not overlap with existing ones
                new_map_point_ids = []
                xs1_new_pixel = []
                for i,x1 in enumerate(xs1_pixel_tmp):
                    x1 = np.array(x1, np.float32)
                    dist_to_existing_feat = 2.0
                    if self.IsCloseToExistingFeatures(x1, dist_to_existing_feat, self.frame_ind-1):
                        continue
                    xs1_new_pixel.append(x1)
                    map_point_id = new_point_virtual_ids_tmp[i]
                    new_map_point_ids.append(map_point_id)

                new_points_batch_size = 8 # min number of points to add at once; >5 to more reiliably run the 5points algorithm

                if len(xs1_new_pixel) >= new_points_batch_size:
                    # xs1_new_pixel_array = np.array(xs1_new_pixel, np.float32)
                    # next_pts_new, status_new, err_new = cv2.calcOpticalFlowPyrLK(self.img1_gray, img2_gray,xs1_new_pixel_array, None,winSize=lk_win_size)
                    # FixOpenCVCalcOpticalFlowPyrLK(xs1_new_pixel, next_pts_new, status_new, lk_win_size)
                    next_pts_new, status_new = self.img_feat_tracker.TrackFeats(self.img1_ind, self.img1_gray, img2_ind, img2_gray, new_map_point_ids, xs1_new_pixel)

                    if debug >= 4:
                        xs1_pixel_tmp = [x for s, x in zip(status_new, xs1_new_pixel) if s > 0]
                        xs2_pixel_tmp = [x for s, x in zip(status_new, next_pts_new) if s > 0]
                        ShowMatches("matches", self.img1_bgr, img2_bgr, xs1_pixel_tmp, xs2_pixel_tmp)

                    # convert pixel to image coordinates (pixel -> meters)
                    xs1_new_meter_no_gaps = []
                    xs1_new_pixel_no_gaps = []
                    new_map_point_ids_no_gaps = []
                    for i in range(0, len(status_new)):
                        if status_new[i]:
                            x1_pixel = xs1_new_pixel[i]
                            xs1_new_pixel_no_gaps.append(x1_pixel)

                            x1_meter = convertPointPixelToMeter(x1_pixel, self.cam_mat_meter_from_pixel)
                            xs1_new_meter_no_gaps.append(x1_meter)

                            x2_pixel = next_pts_new[i]
                            xs2_new_pixel_no_gaps.append(x2_pixel)

                            x2_meter = convertPointPixelToMeter(x2_pixel, self.cam_mat_meter_from_pixel)
                            xs2_new_meter_no_gaps.append(x2_meter)

                            new_map_point_ids_no_gaps.append(new_map_point_ids[i])

                    # append new points into the previous frame
                    # initial points
                    new_track_ids = self.PutNewPoints2D(self.frame_ind-1, self.frame_ind, xs1_new_meter_no_gaps, xs1_new_pixel_no_gaps, new_map_point_ids_no_gaps)
                    # set coordinates
                    self.PutMatchedPoints2D(new_track_ids, xs2_new_meter_no_gaps, xs2_new_pixel_no_gaps)
        draw_camera_trackview = False
        if draw_camera_trackview:
            if self.img_trackview_bgr is None:
                self.img_trackview_bgr = img2_bgr.copy()
            else:
                self.img_trackview_bgr[:,:,:] = img2_bgr[:,:,:]
            self.DrawCameraTrackView(self.img_trackview_bgr) # draw on a dedicated (image) surface
            cv2.waitKey(1)

        self.Process(on_pnt_corresp=ShowPntCorresp, debug=debug)

        # update cursor
        self.img1_gray = img2_gray
        self.img1_bgr = img2_bgr
        self.img1_ind = img2_ind
        return None

    def Process(self, on_pnt_corresp = None, debug = 1):
        matched_pnts_count, new_pnts_count, lost_pnts_count = self.__CalcMapPointsStat()
        self.num_tracked_points_per_frame.append(matched_pnts_count)

        if debug >= 3 or True:
            print("img={} change of points, matched:{} lost:{} new:{}".format(self.frame_ind, matched_pnts_count, lost_pnts_count, new_pnts_count))

        if self.slam_impl == 1:
            self.LocateCamAndMap_LatestFrameSequentialNaive(on_pnt_corresp, debug)
        elif self.slam_impl == 3:
            self.LocateCamAndMap_MultiViewFactorization(on_pnt_corresp, debug)


    def LocateCamAndMap_LatestFrameSequentialNaive(self, on_pnt_corresp, debug):
        """ Naive SLAM implementation.
        Localization: tries to find the relative motion between the last frame with determined 3D camera position and the latest (current) frame.
        Mapping: if relative motion to the latest frame is found, then it triangulates all new points and adds them to the map (forever). 
        """
        if self.frame_ind == 0:
            # assiciate the world frame with the first camera frame
            worldR = np.eye(3, 3)
            worldT = np.zeros(3)
            with self.visual_host.cameras_lock:
                self.visual_host.world_to_cam_R.append(worldR)
                self.visual_host.world_to_cam_T.append(worldT)

            with self.visual_host.continue_computation_lock:
                self.visual_host.world_map_changed_flag = True
            return
        # initialization step
        if self.frame_ind < 50 and True:
            worldR = None
            worldT = None
            with self.visual_host.cameras_lock:
                self.visual_host.world_to_cam_R.append(worldR)
                self.visual_host.world_to_cam_T.append(worldT)

            with self.visual_host.continue_computation_lock:
                self.visual_host.world_map_changed_flag = True
            return

        # enumerate matched points
        xs1_meter = []
        xs2_meter = []
        matched_pnt_ids = []
        for pnt_life in self.points_life:
            x1 = pnt_life.points_list_meter[self.last_known_pos_frame_ind]
            x2 = pnt_life.points_list_meter[self.frame_ind]

            # check that all frames between [known_frame_ind, latest] have 2D point value
            is_continuous_match = not x1 is None and not x2 is None
            if is_continuous_match:
                assert not x1 is None
                assert not x2 is None
                xs1_meter.append(x1)
                xs2_meter.append(x2)

                matched_pnt_ids.append(pnt_life.point_id)

        num_matched_pnts = len(matched_pnt_ids)
        assert num_matched_pnts >= 5, "required for 5point relative motion algorithm"

        if debug >= 3: print("finding relative motion between img:{}-{} using {} points".format(self.last_known_pos_frame_ind, self.frame_ind, num_matched_pnts))

        sampson_dist_calculator = SampsonDistanceCalc()

        def MeasureConsensusFun(matched_points_count, samp_group_inds, cons_set_mask):
            assert len(matched_pnt_ids) == matched_points_count, "must sample matched points"
            samp_pnt_ids = [matched_pnt_ids[i] for i in samp_group_inds]

            samp_xs1_meters = []
            samp_xs2_meters = []
            for id in samp_pnt_ids:
                pnt_life = self.points_life[id]
                samp_xs1_meters.append(pnt_life.points_list_meter[self.frame_ind - 1])
                samp_xs2_meters.append(pnt_life.points_list_meter[self.frame_ind])

            if debug >= 94:
                on_pnt_corresp("sample", samp_pnt_ids)

            suc, ess_mat_list = FindEssentialMat5PointStewenius(samp_xs1_meters, samp_xs2_meters, True, True, check_constr=False, debug=debug, expected_ess_mat=None)
            if not suc:
                return 0 # no consensus

            def CalcConsensus(ess_mat, dist_thr, cons_set_mask):
                cons_set_card = 0
                for i in range(0, matched_points_count):
                    pt1 = xs1_meter[i]
                    pt2 = xs2_meter[i]

                    dist = sampson_dist_calculator.Distance(ess_mat, pt1, pt2)
                    cons = dist < dist_thr
                    # if debug >= 3: print("err={0} include={1}".format(err, cons))
                    if cons:
                        cons_set_card += 1
                    cons_set_mask[i] = cons
                return cons_set_card

            dist_thr = 0.0001 # TODO: choose distance

            cand_cons_set_mask = np.zeros_like(cons_set_mask)
            best_cons_set_card = 0
            for ess_mat in ess_mat_list:
                cons_set_card = CalcConsensus(ess_mat, dist_thr, cand_cons_set_mask)
                if cons_set_card > best_cons_set_card:
                    best_cons_set_card = cons_set_card
                    cons_set_mask[:] = cand_cons_set_mask[:]

            return best_cons_set_card

        consens_mask = np.zeros(num_matched_pnts, np.uint8)
        suc_prob = 0.99
        outlier_ratio = 0.3
        cons_set_card = GetMaxSubsetInConsensus(num_matched_pnts, 5, outlier_ratio, suc_prob, MeasureConsensusFun, consens_mask)
        assert cons_set_card >= 5, "need >= 5 points to compute essential mat"
        if debug >= 3: print("cons_set_card={0}".format(cons_set_card))

        cons_xs1_meter = np.array([p for i, p in enumerate(xs1_meter) if consens_mask[i]])
        cons_xs2_meter = np.array([p for i, p in enumerate(xs2_meter) if consens_mask[i]])
        cons_ids = np.array([matched_pnt_ids[i] for i in range(0, len(consens_mask)) if consens_mask[i]])

        if not on_pnt_corresp is None and debug >= 4:
            on_pnt_corresp("cons", cons_ids)

        expected_ess_mat = None
        if not self.ground_truth_relative_motion is None:
            expR,expT = self.ground_truth_relative_motion(self.last_known_pos_frame_ind, self.frame_ind)
            expected_ess_mat = skewSymmeticMat(expT/LA.norm(expT)).dot(expR)
            expected_ess_mat /= LA.norm(expected_ess_mat)

        cur_cam_from_world_R = None
        cur_cam_from_world_T = None
        suc, ess_mat_list = FindEssentialMat5PointStewenius(cons_xs1_meter, cons_xs2_meter, proj_ess_space=True, check_constr=True, debug=debug, expected_ess_mat=expected_ess_mat)
        #assert suc, "Essential matrix on consensus set must be calculated"
        if not suc:
            print("Failed xs->E")
            ess_mat = None
        else:
            # choose essential matrix with minimal sampson error
            samps_errs = [sampson_dist_calculator.DistanceMult(e, cons_xs1_meter, cons_xs2_meter) for e in ess_mat_list]
            ess_mat_best_item = min(zip(ess_mat_list, samps_errs), key=lambda item: item[1])
            ess_mat = ess_mat_best_item[0]

            if debug >= 3: print("ess_mat=\n{0}".format(ess_mat))

        # TODO: refine essential matrix; RefineFundMat refines fundamental matrix and destroys internal constraints of essential matrix
        refine_ess = False
        if refine_ess:
            suc, ess_mat_refined = RefineFundMat(ess_mat, cons_xs1_meter, cons_xs2_meter, debug=debug)
            if not suc:
                if debug >= 3: print("img_ind={} can't refine ess mat".format(self.frame_ind))
            else:
                if debug >= 3: print("refined_ess_mat=\n{0}".format(ess_mat_refined))

            perr = [""]
            assert IsEssentialMat(ess_mat_refined, perr), "essential mat is valid after refinement, " + perr[0]
        else:
            ess_mat_refined = ess_mat

        #
        if not ess_mat is None:
            cur_cam_from_world_R, cur_cam_from_world_T = None, None
            suc, rel_R, rel_T = ExtractRotTransFromEssentialMat(ess_mat_refined, cons_xs1_meter, cons_xs2_meter, debug=debug)
            if not suc:
                if debug >= 3: print("Failed E->R,T")
                #on_pnt_corresp("cons", cons_ids)
            else:
                ess_wvec, ess_wang = logSO3(rel_R)

                expect_R = None
                expect_T = None
                expect_Rw = None
                expect_Rang_deg = None
                if not self.ground_truth_relative_motion is None:
                    expect_R, expect_T = self.ground_truth_relative_motion(self.last_known_pos_frame_ind, self.frame_ind)
                    expect_Rw, expect_Rang = logSO3(expect_R)
                    expect_Rang_deg = math.degrees(expect_Rang)
                    # make T to be the unity
                    expect_Tuni = expect_T / LA.norm(expect_T)


                if debug >= 3:
                    print("R|T: w={0} ang={1}deg T:{2} expectRw:{3} expectRang:{4}deg expectT:{5} R:expectR\n{6}\n{7}"
                         .format(ess_wvec, math.degrees(ess_wang), rel_T,
                         expect_Rw, expect_Rang_deg, expect_Tuni, rel_R, expect_R))

                # relative camera motion -> camera from world
                head_R, head_T = None, None
                with self.visual_host.cameras_lock:
                    head_R = self.visual_host.world_to_cam_R[self.last_known_pos_frame_ind]
                    head_T = self.visual_host.world_to_cam_T[self.last_known_pos_frame_ind]

                #use_rel_T = rel_T
                use_rel_T = rel_T / self.focal_len
                cur_cam_from_world = SE3Compose((rel_R, use_rel_T), (head_R, head_T))
                cur_cam_from_world_R, cur_cam_from_world_T = cur_cam_from_world

                # reconstruct map
                reconstruct_map = True
                if reconstruct_map:
                    world_from_head = SE3Inv((head_R, head_T))
                    world_from_cur_cam = SE3Inv(cur_cam_from_world)
                    pZ = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
                    for pnt_life in self.points_life:
                        if pnt_life.is_mapped: continue
                        # TODO: map all points in consensus, which are not mapped yet
                        x1_meter = pnt_life.points_list_meter[self.last_known_pos_frame_ind]
                        x2_meter = pnt_life.points_list_meter[self.frame_ind]
                        if x1_meter is None or x2_meter is None:
                            continue
                        pA = np.hstack((rel_R, rel_T.reshape((3, 1))))
                        q3D = triangulateDlt(pZ, pA, x1_meter, x2_meter, normalize=False)
                        almost_zero = 1e-4  # slightly bigger value to catch 1.059e-5
                        if abs(q3D[-1]) < almost_zero:
                            # algorithm failed to determine the valid [R,T] candidate
                            continue
                        pos3D_head = q3D / q3D[-1]
                        pos3D_head = pos3D_head[0:3]
                        pos3D_head_f = pos3D_head / self.focal_len # measure distance in terms of focal lengths
                        # TODO: how to scale pos3D_head?

                        lam1 = x1_meter.dot(pos3D_head) / LA.norm(x1_meter)**2
                        pos3D_cur = rel_R.dot(pos3D_head) + rel_T
                        lam2 = x2_meter.dot(pos3D_cur) / LA.norm(x2_meter)**2
                        aaaleft = lam2 * x2_meter
                        aaarigh = lam1 * rel_R.dot(x1_meter) + rel_T
                        # assert left==right

                        pos3D_world = SE3Apply(world_from_head, pos3D_head)
                        #pos3D_world_f = SE3Apply(world_from_head, pos3D_head_f)
                        if debug >= 3:
                            map_pnt_true_pos = None
                            if not self.ground_truth_map_pnt_pos is None and not pnt_life.map_point_id is None:
                                map_pnt_true_pos = self.ground_truth_map_pnt_pos(0, pnt_life.map_point_id)
                                #map_pnt_true_pos /= self.focal_len # measure distance in terms of focal lengths
                            print("reconstructed track_id:{} map_point_id:{} pos:{} true_pos:{}".format(pnt_life.point_id, pnt_life.map_point_id, pos3D_world,  map_pnt_true_pos))
                        self.visual_host.xs3d.append(pos3D_world)
                        pnt_life.is_mapped = True

        with self.visual_host.cameras_lock:
            # map to the world frame
            if suc:
                # head_R = self.visual_host.world_to_cam_R[self.last_known_pos_frame_ind]
                # head_T = self.visual_host.world_to_cam_T[self.last_known_pos_frame_ind]
                # camera_from_world_R = np.dot(head_R, ess_R)
                # camera_from_world_T = np.dot(head_R, ess_Tvec) + head_T

                # camera pos in the world = world_from_camera*[0,0,0,1]=inv(camera_from_world)*[0,0,0,1]
                pos_prev = -head_R.T.dot(head_T)
                pos = -cur_cam_from_world_R.T.dot(cur_cam_from_world_T)
                dx = LA.norm(pos - pos_prev)
                if dx > 10:
                    print("too big relative motion")

            self.visual_host.world_to_cam_R.append(cur_cam_from_world_R)
            self.visual_host.world_to_cam_T.append(cur_cam_from_world_T)
            pass

        # on failure to find [R,T] the head_ind doesn't change
        if not cur_cam_from_world_R is None:
            #suc, frames_R, frames_T, world_pnts = self.EstimateMap(debug=debug)
            suc = False
            if suc:
                with self.visual_host.cameras_lock:
                    self.visual_host.world_to_cam_R_alt = frames_R
                    self.visual_host.world_to_cam_T_alt = frames_T
                    self.visual_host.xs3d = world_pnts
                    pass

            self.last_known_pos_frame_ind = self.frame_ind

        with self.visual_host.continue_computation_lock:
            self.visual_host.world_map_changed_flag = True
            self.visual_host.processed_images_counter += 1

        return None

    def LocateCamAndMap_MultiViewFactorization(self, on_pnt_corresp, debug):
        show_pnt_frame_participation = True
        if show_pnt_frame_participation:
            pnt_to_frame = BuildPointInFrameRelatonMatrix(self.points_life, sort_points=True)
            cv2.imshow("pnt_to_frame", pnt_to_frame)

        # on failure to find [R,T] the head_ind doesn't change
        suc = self.MultiViewFactorizationIntegrateNewFrame(self.framei_from_world_RT_list, self.world_pnts, debug=debug)
        if not suc:
            return
        with self.visual_host.cameras_lock:
            self.visual_host.world_to_cam_R_alt = [i[0] for i in self.framei_from_world_RT_list]
            self.visual_host.world_to_cam_T_alt = [i[1] for i in self.framei_from_world_RT_list]

            world_pnts_no_gaps = [p for p in self.world_pnts if not p is None]
            self.visual_host.xs3d = world_pnts_no_gaps

        with self.visual_host.continue_computation_lock:
            self.visual_host.world_map_changed_flag = True
            self.visual_host.processed_images_counter += 1

        return None


    def IsLive3DPoint(self, pnt_id, from_frame_ind, to_frame_ind):
        """Whether the point has registered 2D point in each frame"""
        pnt_life = self.points_life[pnt_id]
        for fid in range(from_frame_ind, to_frame_ind):
            x = pnt_life.points_list_meter[fid]
            if x is None: return False
        return True

    def LocateCamAndMapBatchRefine(self, debug, min_proj_err):
        reproj_err_history = []

        initial_reproj_err = self.CalcReprojErr(debug)
        if debug >= 3: print("initial_reproj_err={} meters".format(initial_reproj_err))
        reproj_err_history.append(initial_reproj_err)

        frames_count = self.frame_ind + 1
        points_count = len(self.points_life)

        max_iter = 9
        it = 1
        learn_rate = 0.3 # the speed of change from old to new value in range=[0,1], 0=old value, 1=new value
        do_update = True

        class IterCursor:
            def __init__(self):
                self.next_frame_ind = None
                self.next_pnt_ind = None
                self.localize_or_map = None
                self.localize_finished = None
                self.map_finished = None
                self.Reset()
            def Reset(self):
                self.next_frame_ind = 1
                self.next_pnt_ind = 0
                self.localize_or_map = 'localize'
                self.localize_finished = False
                self.map_finished = False

        cursor = IterCursor()

        while True:
            if cursor.localize_finished and cursor.map_finished:
                reproj_err = self.CalcReprojErr(debug)
                if debug >= 3: print("reproj_err={} meters".format(reproj_err))

                reproj_err_history.append(reproj_err)

                if reproj_err < min_proj_err:
                    break

                it += 1
                if it > max_iter:
                    print("reproj_err_history={}".format(reproj_err_history))
                    assert False, "Failed to converge the reconstruction"
                    break
                # reset cursor to the initial state
                cursor.Reset()
                continue

            # learn_rate=[0,1]
            def SmoothUpdate(x1, x2, learn_rate):
                delta = x2 - x1
                result = x1 + delta * learn_rate
                return result

            if cursor.localize_or_map == 'localize':
                if cursor.next_frame_ind >= frames_count:
                    cursor.localize_finished = True
                    cursor.localize_or_map = 'map'
                    continue

                # determine the set of all points in the latest frame
                next_frame_pnt_ids = []
                next_frame_pnt_ids_set = set([])
                for pnt_life in self.points_life:
                    x_meter = pnt_life.points_list_meter[cursor.next_frame_ind]
                    if x_meter is None: continue

                    next_frame_pnt_ids.append(pnt_life.track_id)
                    next_frame_pnt_ids_set.add(pnt_life.track_id)

                anchor_frame_ind, common_pnts_count = self.FindAnchorFrame(cursor.next_frame_ind, next_frame_pnt_ids_set)

                # find relative motion of the new frame
                common_pnt_ids = []
                self.CountCommonPoints(next_frame_pnt_ids_set, anchor_frame_ind, common_pnt_ids)

                # TODO: relative RT routine diverges very quickly, what to do?
                frame_ind_from_anchor_RT = self.GetFrameRelativeRT(anchor_frame_ind, cursor.next_frame_ind, common_pnt_ids)

                # hack: replace relative RT with ground truth
                if self.hack_camera_location_in_batch_refine and not self.ground_truth_relative_motion is None:
                    gtruth_R, gtruth_T = self.ground_truth_relative_motion(anchor_frame_ind, cursor.next_frame_ind)
                    scaled_T = gtruth_T * self.first_unity_translation_scale_factor
                    frame_ind_from_anchor_RT = (gtruth_R, scaled_T)

                anchor_from_world_RT = self.framei_from_world_RT_list[anchor_frame_ind]
                frame_ind_from_world_RT = SE3Compose(frame_ind_from_anchor_RT, anchor_from_world_RT)

                old_RT = self.framei_from_world_RT_list[cursor.next_frame_ind]
                diff1 = LA.norm(frame_ind_from_world_RT[0] - old_RT[0])
                diff2 = LA.norm(frame_ind_from_world_RT[1] - old_RT[1])
                if diff1 > 1 or diff2 > 1:
                    print("NoConvergence: too big RT, next_frame_ind: {}, diffR: {} diffT: {}".format(cursor.next_frame_ind, diff1, diff2))
                    print("old: {}".format(old_RT))
                    print("new: {}".format(frame_ind_from_world_RT))
                print("frame_ind: {}-{}, diffR: {} diffT: {}".format(anchor_frame_ind, cursor.next_frame_ind, diff1, diff2))

                # do smooth update of RT
                upd_T = SmoothUpdate(old_RT[1], frame_ind_from_world_RT[1], learn_rate)

                old_N, old_Ang = logSO3(old_RT[0], check_rot_mat=True)
                new_N, new_Ang = logSO3(frame_ind_from_world_RT[0], check_rot_mat=True)
                upd_N = SmoothUpdate(old_N, new_N, learn_rate)
                upd_N = upd_N / LA.norm(upd_N)
                upd_Ang = SmoothUpdate(old_Ang, new_Ang, learn_rate)

                upd_R = rotMat(upd_N, upd_Ang, check_log_SO3=True)
                upd_RT = (upd_R, upd_T)

                if do_update:
                    self.framei_from_world_RT_list[cursor.next_frame_ind] = upd_RT

                cursor.next_frame_ind += 1
                cursor.localize_or_map = 'map'
                continue

            if cursor.localize_or_map == 'map':
                if cursor.next_pnt_ind >= points_count:
                    cursor.map_finished = True
                    cursor.localize_or_map = 'localize'
                    continue

                next_pnt_id = cursor.next_pnt_ind
                x3D_world = self.Estimate3DPointFromFrames(next_pnt_id, check_drift=self.check_drift)
                if not x3D_world is None:
                    old_pnt = self.world_pnts[cursor.next_pnt_ind]
                    diff = LA.norm(old_pnt - x3D_world)
                    if diff > 1:
                        print("NoConvergence: too big point, pnt_id: {} diff: {}".format(next_pnt_id, diff))
                        print("old: {}".format(old_pnt))
                        print("new: {}".format(x3D_world))

                    upd_pnt = SmoothUpdate(old_pnt, x3D_world, learn_rate)
                    self.world_pnts[cursor.next_pnt_ind] = upd_pnt

                cursor.next_pnt_ind += 1
                cursor.localize_or_map = 'localize'
                continue
        pass # optimization loop

        print("exiting LocateCamAndMapBatchRefine")


    # gets the depth of a point in the given frame
    def Get3DPointDepth(self, pnt_id, anchor_frame_ind):
        anchor_RT = self.framei_from_world_RT_list[anchor_frame_ind]
        assert not anchor_RT[0] is None
        assert not anchor_RT[1] is None

        pnt_ind = pnt_id
        x3D_world = self.world_pnts[pnt_ind]
        if x3D_world is None:
            print("error: x3D_world is None")
            assert False

        x3D_anchor = SE3Apply(anchor_RT, x3D_world)

        if not self.ground_truth_map_pnt_pos is None:
            pnt_ind = pnt_id
            pnt_life = self.points_life[pnt_ind]
            w3D = self.ground_truth_map_pnt_pos(anchor_frame_ind, pnt_life.virtual_feat_id) * self.first_unity_translation_scale_factor
            diff = LA.norm(w3D - x3D_anchor)
            # delta = 1e-1 # must pass 0.027781092461721443
            delta = self.drift
            if not np.isclose(0, diff, atol=delta):
                print("error: can't find depth of a 3D point, diff={}".format(diff))
                if self.check_drift:
                    assert False
        return x3D_anchor[2]

    def CollectFrameInfoListForPoint(self, pnt_id):
        frames_count = self.frame_ind + 1
        pnt_ind = pnt_id
        pnt_life = self.points_life[pnt_ind]

        frame_inds = []
        x_per_frame = {}
        framei_from_base_RT_dict = {}
        block_base_frame_ind = None
        for fr_ind in range(0, frames_count):
            # # skip the latest frame, because we don't know its (R,T) yet
            # if fr_ind == self.frame_ind: continue

            x_meter = pnt_life.points_list_meter[fr_ind]
            if x_meter is None: continue

            x_per_frame[fr_ind] = x_meter
            frame_inds.append(fr_ind)

            if block_base_frame_ind is None:
                block_base_frame_ind = fr_ind

            fr_RT = self.framei_from_world_RT_list[fr_ind]
            assert not fr_RT[0] is None, "we are intereseted in frames with known (R,T) camera orientation"

            base_from_world = self.framei_from_world_RT_list[block_base_frame_ind]
            framei_from_world = self.framei_from_world_RT_list[fr_ind]
            framei_from_base = RelMotionBFromA(base_from_world, framei_from_world)
            framei_from_base_RT_dict[fr_ind] = framei_from_base
        return frame_inds, x_per_frame, framei_from_base_RT_dict

    def FindRelativeMotionMultiPoints(self, base_frame_ind, targ_frame_ind, pnt_ids, pnt_depthes):
        if self.use_mpmath:
            result = self.FindRelativeMotionMultiPoints_mpmath(base_frame_ind, targ_frame_ind, pnt_ids, pnt_depthes)
        else:
            result = self.FindRelativeMotionMultiPoints_float(base_frame_ind, targ_frame_ind, pnt_ids, pnt_depthes)
        return result

    def FindRelativeMotionMultiPoints_float(self, base_frame_ind, targ_frame_ind, pnt_ids, pnt_depthes):
        eltype = self.elem_type
        points_num = len(pnt_ids)
        assert points_num == len(pnt_depthes)

        # estimage camera position [R,T] given distances to all 3D points pj in frame1
        A = np.zeros((3 * points_num, 12), dtype=eltype)
        for i, pnt_id in enumerate(pnt_ids):
            pnt_ind = pnt_id
            pnt_life = self.points_life[pnt_ind]
            x1 = pnt_life.points_list_meter[base_frame_ind]

            x_img = pnt_life.points_list_meter[targ_frame_ind]
            x_img_skew = skewSymmeticMat(x_img)
            block_left = np.kron(x1.reshape(1, 3), x_img_skew)
            A[3 * i:3 * (i + 1), 0:9] = block_left
            depth = pnt_depthes[i]
            alph = 1 / depth
            A[3 * i:3 * (i + 1), 9:12] = alph * x_img_skew

        if self.la_engine == "opencv":
            dVec1, u1, vt1 = cv2.SVDecomp(A)
        elif self.la_engine == "scipy":
            u1, dVec1, vt1 = scipy.linalg.svd(A)
            if self.conceal_lost_precision:
                vt1 = vt1.astype(eltype)

        r_and_t = vt1.T[:, -1]

        cam_R_noisy = r_and_t[0:9].reshape(3, 3, order='F')  # unstack
        cam_Tvec_noisy = r_and_t[9:12]

        # project noisy [R,T] onto SO(3) (see MASKS, formula 8.41 and 8.42)
        if self.la_engine == "opencv":
            dVec2, u2, vt2 = cv2.SVDecomp(cam_R_noisy)
        elif self.la_engine == "scipy":
            u2, dVec2, vt2 = scipy.linalg.svd(cam_R_noisy)
            if self.conceal_lost_precision:
                vt2 = vt2.astype(eltype)

        no_guts = np.dot(u2, vt2)
        sign1 = np.sign(scipy.linalg.det(no_guts))
        frame_R = sign1 * no_guts

        det_den = LA.det(np.diag(dVec2.ravel()))
        if math.isclose(0, det_den):
            return False, None

        t_factor = sign1 / rootCube(det_den)
        cam_Tvec = cam_Tvec_noisy * t_factor

        if sum(1 for a in cam_Tvec if not math.isfinite(a)) > 0:
            print("error: nan")
            return False, None

        rel_R = frame_R[:, :]
        rel_T = cam_Tvec[:]

        p_err = [""]
        assert IsSpecialOrthogonal(rel_R, p_err), p_err[0]

        return True, (rel_R, rel_T)

    # Multi-precision math (mpmath) implementation.
    def FindRelativeMotionMultiPoints_mpmath(self, base_frame_ind, targ_frame_ind, pnt_ids, pnt_depthes):
        eltype = self.elem_type
        points_num = len(pnt_ids)
        assert points_num == len(pnt_depthes)

        # estimage camera position [R,T] given distances to all 3D points pj in frame1
        x_img_skew = mpmath.matrix(3,3)
        A = mpmath.matrix(3 * points_num, 12)
        for i, pnt_id in enumerate(pnt_ids):
            pnt_ind = pnt_id
            pnt_life = self.points_life[pnt_ind]
            x1 = pnt_life.points_list_meter[base_frame_ind]

            x_img = pnt_life.points_list_meter[targ_frame_ind]
            skewSymmeticMatWithAdapter(x_img, lambda x: mpmath.mpf(str(x)), x_img_skew)

            A[3 * i:3 * (i + 1), 0:3] = x_img_skew * mpmath.mpf(str(x1[0]))
            A[3 * i:3 * (i + 1), 3:6] = x_img_skew * mpmath.mpf(str(x1[1]))
            A[3 * i:3 * (i + 1), 6:9] = x_img_skew * mpmath.mpf(str(x1[2]))

            depth = mpmath.mpf(str(pnt_depthes[i]))
            alph = 1 / depth
            A[3 * i:3 * (i + 1), 9:12] = x_img_skew * alph

        #
        u1, dVec1, vt1 = mpmath.svd_r(A)

        # take the last column of V
        cam_R_noisy = mpmath.matrix(3,3)
        for col in range(0, 3):
            for row in range(0, 3):
                cam_R_noisy[row,col] = vt1[11,col*3+row]
        cam_Tvec_noisy = mpmath.matrix(3,1)
        for row in range(0, 3):
            cam_Tvec_noisy[row,0] = vt1[11, 9+row]

        # project noisy [R,T] onto SO(3) (see MASKS, formula 8.41 and 8.42)
        u2, dVec2, vt2 = mpmath.svd(cam_R_noisy)

        no_guts = u2 * vt2
        no_guts_det = mpmath.det(no_guts)
        sign1 = mpmath.sign(no_guts_det)
        frame_R = no_guts * sign1

        det_den = dVec2[0] * dVec2[1] * dVec2[2]
        if math.isclose(0, det_den):
            return False, None

        t_factor = sign1 / rootCube(det_den)
        cam_Tvec = cam_Tvec_noisy * t_factor

        if sum(1 for a in cam_Tvec if not math.isfinite(a)) > 0:
            print("error: nan")
            return False, None

        # convert results into default precision type
        rel_R = np.zeros((3,3), dtype=eltype)
        rel_T = np.zeros(3, dtype=eltype)
        for row in range(0, 3):
            for col in range(0, 3):
                rel_R[row, col] = float(frame_R[row, col])
            rel_T[row] = cam_Tvec[row, 0]

        p_err = [""]
        assert IsSpecialOrthogonal(rel_R, p_err), p_err[0]

        return True, (rel_R, rel_T)

    def Estimate3DPointFromFrames(self, pnt_id, check_drift):
        pnt_ind = pnt_id
        pnt_life = self.points_life[pnt_ind]

        frame_inds, x_per_frame, framei_from_base_RT_dict = self.CollectFrameInfoListForPoint(pnt_id)
        if len(frame_inds) <= 1:
            print("ignoring 3D point pnt_id:{} because it occurs in {} frames".format(pnt_id, len(frame_inds)))
            return None
        anchor_frame_ind = frame_inds[0]

        depth_anchor = Estimate3DPointDepthFromFrames(anchor_frame_ind, frame_inds, x_per_frame, framei_from_base_RT_dict)

        x_meter = pnt_life.points_list_meter[anchor_frame_ind]
        x3D_anchor = depth_anchor * x_meter

        if not self.ground_truth_map_pnt_pos is None:
            gtruth_x3D_world = self.ground_truth_map_pnt_pos(anchor_frame_ind, pnt_life.virtual_feat_id)
            scaled_x3D_world = gtruth_x3D_world
            if not self.first_unity_translation_scale_factor is None:
                scaled_x3D_world *= self.first_unity_translation_scale_factor
            diff = LA.norm(scaled_x3D_world - x3D_anchor)
            # delta = 1e-1 # must pass 0.00106298567037, 0.01252073431, 0.028073793742802243
            delta = self.drift
            if not np.isclose(0, diff, atol=delta):
                print("diff={}".format(diff))
                print("expect 3D: {}".format(scaled_x3D_world))
                print("actual 3D: {}".format(x3D_anchor))
                if check_drift:
                    assert False

        # convert point in base, into point in the world
        anchor_from_world = self.framei_from_world_RT_list[anchor_frame_ind]
        world_from_anchor = SE3Inv(anchor_from_world)
        x3D_world = SE3Apply(world_from_anchor, x3D_anchor)

        # HACK:
        # x3D_world = scaled_x3D_world
        return x3D_world

    def CalcReprojErr(self, debug):
        """Calculate projection error"""
        # MASKS formula 11.18, page 397
        result = 0.0
        points_count = len(self.points_life)
        one_err_count = 0
        for pnt_ind in range(0, points_count):
            pnt_life = self.points_life[pnt_ind]

            x3D_world = self.world_pnts[pnt_ind]
            if x3D_world is None:
                assert not pnt_life.is_mapped  # point hasn't been mapped yet
                continue

            for frame_ind in range(pnt_life.start_frame_ind, pnt_life.last_frame_ind):
                x_expect = pnt_life.points_list_meter[frame_ind]
                x_expectN = x_expect / x_expect[-1]

                # transform 3D point into the frame
                targ_from_world_RT = self.framei_from_world_RT_list[frame_ind]
                x3D_framei = SE3Apply(targ_from_world_RT, x3D_world)

                # project 3D point into the frame
                x3D_frameiN = x3D_framei / x3D_framei[-1]
                err = LA.norm(x_expectN - x3D_frameiN) ** 2
                # if debug >= 3: print("xexpect={0} xact={1} err={2} meters".format(x_expect, x3D_frameiN, err))
                result += err
                one_err_count += 1
        if one_err_count > 0:
            result /= one_err_count
        return result

    def GetFrameRelativeRT(self, base_frame_ind, targ_frame_ind, common_pnt_ids):
        pnt_depthes_base = [self.Get3DPointDepth(pnt_id, base_frame_ind) for pnt_id in common_pnt_ids]
        suc, frame_ind_from_base_RT = self.FindRelativeMotionMultiPoints(base_frame_ind, targ_frame_ind, common_pnt_ids, pnt_depthes_base)
        if not suc:
            print("error! can't determine the relative motion")
            assert suc

            # check the relative motion of a camera
        if not self.ground_truth_relative_motion is None:
            print("base->frame_ind={} frame_ind_RT={}".format((base_frame_ind, targ_frame_ind), frame_ind_from_base_RT))

            # world_frame_ind = 0
            gtruth_R, gtruth_T = self.ground_truth_relative_motion(base_frame_ind, targ_frame_ind)
            gtruth_Rw, gtruth_Rang = logSO3(gtruth_R)
            true_Rang_deg = math.degrees(gtruth_Rang)
            print("ground truth R\n{}".format(gtruth_R))

            scaled_T = gtruth_T
            if not self.first_unity_translation_scale_factor is None:
                scaled_T *= self.first_unity_translation_scale_factor
            print("scaled T: {}".format(scaled_T))

            diff_R = LA.norm(gtruth_R - frame_ind_from_base_RT[0])
            diff_T = LA.norm(scaled_T - frame_ind_from_base_RT[1])
            delta = self.drift
            failed = False
            if not np.isclose(0, diff_R, atol=delta):
                print("error: failed to get correct R diff_R={}".format(diff_R))
                failed = True
            if not np.isclose(0, diff_T, atol=delta):
                print("error: failed to get correct T diff_T={}".format(diff_T))
                failed = True
            if failed and self.check_drift:
                assert False

        return frame_ind_from_base_RT

    # counts the number of common points between base and target frames
    def CountCommonPoints(self, base_frame_pnt_ids_set, targ_frame_ind, common_pnt_ids=None):
        result = 0
        for pnt_life in self.points_life:
            x_meter = pnt_life.points_list_meter[targ_frame_ind]
            if x_meter is None: continue

            pnt_id = pnt_life.track_id
            pnt_ind = pnt_id
            pnt3D = self.world_pnts[pnt_ind]
            if pnt3D is None: continue

            if pnt_id in base_frame_pnt_ids_set:
                result += 1
                if not common_pnt_ids is None:
                    common_pnt_ids.append(pnt_id)
        return result

    def FindAnchorFrame(self, targ_frame_ind, targ_frame_pnt_ids_set):
        """ For given target frame, finds the frame which shares maximum of common points with given """

        # for this, find the frame which has the most common points with the target frame
        common_pnts_count_per_frame = [self.CountCommonPoints(targ_frame_pnt_ids_set, fr_ind) for fr_ind in range(0, targ_frame_ind)]
        anchor_frame_ind, common_pnts_count = max(enumerate(common_pnts_count_per_frame), key=operator.itemgetter(1))

        assert common_pnts_count >= 6, "Lost tracking: there must be >=6 shared points between consequent frames"

        return anchor_frame_ind, common_pnts_count

    def MultiViewFactorizationIntegrateNewFrame(self, framei_from_world_RT, world_pnts, debug, min_reproj_err = 1e-4):
        # allocate space for the new frame
        frames_count = self.frame_ind + 1
        framei_from_world_RT.append((None,None))

        # allocate space for the new frame
        points_count = len(self.points_life)
        while len(world_pnts) < points_count:
            world_pnts.append(None)

        if frames_count == 1:
            # world's origin
            rot = np.eye(3)
            cent = np.zeros(3)
            framei_from_world_RT[0] = (rot, cent)
            return True

        world_frame_ind = 0

        # find relative motion
        if frames_count == 2:
            points_count = len(self.points_life)
            latest_frame_pnt_ids = []
            xs1_meter = []
            xs2_meter = []
            base_frame_ind = 0
            other_frame_ind = 1
            for pnt_ind in range(0, points_count):
                pnt_life = self.points_life[pnt_ind]
                x1_meter = pnt_life.points_list_meter[base_frame_ind]
                x2_meter = pnt_life.points_list_meter[other_frame_ind]
                if not x1_meter is None and not x2_meter is None:
                    xs1_meter.append(x1_meter)
                    xs2_meter.append(x2_meter)
                    pnt_id = pnt_ind
                    latest_frame_pnt_ids.append(pnt_id)

            suc, frame_ind_from_base_RT = FindRelativeMotion(self.points_life, xs1_meter, xs2_meter, debug, la_engine=self.la_engine, conceal_lost_precision=self.conceal_lost_precision)
            if not suc:
                print("failed FindRelativeMotion")
                return False

            # check the relative motion of a camera
            if not self.ground_truth_relative_motion is None:
                print(frame_ind_from_base_RT)

                gtruth_R, gtruth_T = self.ground_truth_relative_motion(0, 1)
                gtruth_Rw, gtruth_Rang = logSO3(gtruth_R)
                true_Rang_deg = math.degrees(gtruth_Rang)
                print("ground truth R: {}".format(gtruth_R))

                # make the first translation of unity lenght
                self.first_unity_translation_scale_factor = 1.0 / LA.norm(gtruth_T)
                scaled_T = gtruth_T * self.first_unity_translation_scale_factor
                print("ground truth scaled T: {}".format(scaled_T))

                diff = LA.norm(scaled_T-frame_ind_from_base_RT[1])
                delta = self.drift
                if not np.isclose(0, diff, atol=delta):
                    if self.check_drift:
                        print("diff: {}".format(diff))
                        print("expect 3D: {}".format(scaled_T))
                        print("actual 3D: {}".format(frame_ind_from_base_RT[1]))
                        #assert False
                # HACK: assume process works for 2 frames
                if True or self.hack_camera_location:
                    frame_ind_from_base_RT = (gtruth_R, scaled_T)

            framei_from_world_RT[other_frame_ind] = frame_ind_from_base_RT

            # find coordinates of new 3D points
            dists_base = FindDistancesTo3DPointsTwoFrames(self.points_life, latest_frame_pnt_ids, frame_ind_from_base_RT, base_frame_ind, other_frame_ind, debug)

            for i, pnt_id in enumerate(latest_frame_pnt_ids):
                depth = dists_base[i]
                pnt_ind = pnt_id

                pnt_life = self.points_life[pnt_ind]
                x_meter = pnt_life.points_list_meter[base_frame_ind]
                x3D_base = depth * x_meter
                x3D_world = x3D_base

                if not self.ground_truth_map_pnt_pos is None:
                    gtruth_x3D_world = self.ground_truth_map_pnt_pos(base_frame_ind, pnt_life.virtual_feat_id)
                    scaled_x3D_world = gtruth_x3D_world * self.first_unity_translation_scale_factor
                    diff = LA.norm(scaled_x3D_world - x3D_world)
                    #delta = 1e-3
                    delta = self.drift
                    if not np.isclose(0, diff, atol=delta):
                        print("diff: {}".format(diff))
                        print("expect 3D: {}".format(scaled_x3D_world))
                        print("actual 3D: {}".format(x3D_world))
                        if self.check_drift:
                            assert False
                # HACK: assume process works for 2 frames
                if True or self.hack_world_mapping:
                    x3D_world = scaled_x3D_world

                assert not x3D_world is None
                world_pnts[pnt_ind] = x3D_world
        else:
            assert frames_count > 2
            latest_frame_ind = self.frame_ind

            # determine the set of all points in the latest frame
            latest_frame_pnt_ids = []
            latest_frame_pnt_ids_set = set([])
            for pnt_life in self.points_life:
                x_meter = pnt_life.points_list_meter[latest_frame_ind]
                if x_meter is None: continue

                latest_frame_pnt_ids.append(pnt_life.track_id)
                latest_frame_pnt_ids_set.add(pnt_life.track_id)

            anchor_frame_ind, common_pnts_count = self.FindAnchorFrame(latest_frame_ind, latest_frame_pnt_ids_set)

            common_pnt_ids = []
            self.CountCommonPoints(latest_frame_pnt_ids_set, anchor_frame_ind, common_pnt_ids)

            max_iter = 9
            it = 1
            while True:
                frame_ind_from_anchor_RT = self.GetFrameRelativeRT(anchor_frame_ind, latest_frame_ind, common_pnt_ids)

                if self.hack_camera_location and not self.ground_truth_relative_motion is None:
                    gtruth_R, gtruth_T = self.ground_truth_relative_motion(anchor_frame_ind, latest_frame_ind)
                    scaled_T = gtruth_T * self.first_unity_translation_scale_factor
                    frame_ind_from_anchor_RT = (gtruth_R, scaled_T)

                anchor_from_world_RT = self.framei_from_world_RT_list[anchor_frame_ind]
                frame_ind_from_world_RT = SE3Compose(frame_ind_from_anchor_RT, anchor_from_world_RT)

                # if self.hack_camera_location and not self.ground_truth_relative_motion is None:
                #     gtruth_R, gtruth_T = self.ground_truth_relative_motion(world_frame_ind, latest_frame_ind)
                #     scaled_T = gtruth_T * self.first_unity_translation_scale_factor
                #     frame_ind_from_world_RT = (gtruth_R, scaled_T)

                framei_from_world_RT[latest_frame_ind] = frame_ind_from_world_RT

                # determine the depthes of all points in the latest frame
                for pnt_id in latest_frame_pnt_ids:
                    x3D_world = self.Estimate3DPointFromFrames(pnt_id, check_drift=False)
                    if x3D_world is None: continue
                    if self.hack_world_mapping and not self.ground_truth_map_pnt_pos is None:
                        pnt_life = self.points_life[pnt_id]
                        gtruth_x3D_world = self.ground_truth_map_pnt_pos(world_frame_ind, pnt_life.virtual_feat_id)
                        scaled_x3D_world = gtruth_x3D_world * self.first_unity_translation_scale_factor
                        x3D_world = scaled_x3D_world
                    world_pnts[pnt_id] = x3D_world

                reproj_err = self.CalcReprojErr(debug)
                if debug >= 3: print("anchor-targ: {} reproj_err={} meters".format((anchor_frame_ind, latest_frame_ind), reproj_err))

                self.reproj_err_history.append(reproj_err)
                print("reproj_err_history={}".format(self.reproj_err_history))

                # hack: arbitraly fix reconstruction
                do_fix1 = self.frame_ind == 26
                if do_fix1:
                    for pnt_ind,pnt_life in enumerate(self.points_life):
                        gtruth_pos3D = self.ground_truth_map_pnt_pos(world_frame_ind, pnt_life.virtual_feat_id)
                        scaled_gtruth_pos3D = gtruth_pos3D * self.first_unity_translation_scale_factor
                        reconstr_pos3D = self.world_pnts[pnt_ind]
                        if reconstr_pos3D is None:
                            continue
                        # err_one = LA.norm(reconstr_pos3D - scaled_gtruth_pos3D)
                        # err_points += err_one
                        self.world_pnts[pnt_ind] =  scaled_gtruth_pos3D

                    err_r = 0
                    err_t = 0
                    for frame_ind, rt in enumerate(self.framei_from_world_RT_list):
                        gtruth_R, gtruth_T = self.ground_truth_relative_motion(world_frame_ind, frame_ind)
                        scaled_T = gtruth_T * self.first_unity_translation_scale_factor
                        reconstr_R, reconstr_T = rt
                        errR_one = LA.norm(reconstr_R - gtruth_R)
                        errT_one = LA.norm(reconstr_T - scaled_T)
                        err_r += errR_one
                        err_t += errT_one
                        self.framei_from_world_RT_list[frame_ind] = (gtruth_R, scaled_T)

                if reproj_err < min_reproj_err:
                    break

                if it > max_iter:
                    print("reproj_err_history={}".format(self.reproj_err_history))
                    assert False, "Failed to converge the reconstruction"
                    break

                # the iterative reconstruction diverged enough
                # try to coalesce it and minimize a reprojective error
                optimize_structure = "bundle-adjustment"
                if optimize_structure == "iterative-refine":
                    self.LocateCamAndMapBatchRefine(debug, min_reproj_err)
                elif optimize_structure == "bundle-adjustment":
                    ba = BundleAdjustmentKanatani(debug=debug)
                    ba.min_err_change_rel = 0.01
                    converged, err_msg = ba.ComputeInplace(self.points_life, self.cam_mat_pixel_from_meter, self.world_pnts,self.framei_from_world_RT_list)
                    print("BundleAdjustmentKanatani converged={} err_msg={}".format(converged, err_msg))

                it += 1
        return True

    def DrawCameraTrackView(self, img2_bgr, max_track_len=20):
        def DrawFeatTrack(pnt_life, color):
            # draw track of a point
            x_pix_pre = None
            start_frame_ind = max(0, self.frame_ind - max_track_len)
            for fr_ind in range(start_frame_ind, self.frame_ind + 1):
                x_pix = None
                if fr_ind >= 0 and fr_ind < len(pnt_life.points_list_pixel):
                    x_pix = pnt_life.points_list_pixel[fr_ind]

                if x_pix is None:
                    if x_pix_pre is None:
                        continue  # after the last pixel value
                    else:
                        break  # before the first pixel value
                if not x_pix_pre is None:
                    try:
                        cv2.line(img2_bgr, IntPnt(x_pix_pre), IntPnt(x_pix), color)
                    except SystemError:
                        print("")
                x_pix_pre = x_pix
        def DrawHead(pnt_life, color):
            # draw the last pixel
            head_pixel = None
            if self.frame_ind >= 0 and self.frame_ind < len(pnt_life.points_list_pixel):
                head_pixel = pnt_life.points_list_pixel[self.frame_ind]
            if not head_pixel is None:
                cv2.circle(img2_bgr, IntPnt(head_pixel), 3, color)

        track_colors = [
            (0,255,0),
            (0,0,255),
            (255,0,0),
            (255,255,0),# cyan
            (255,0,255),# magenta
            (0,255,255) # yellow
        ]
        for pnt_life in self.points_life:
            color_ind = pnt_life.point_id % len(track_colors)
            color = track_colors[color_ind]
            DrawFeatTrack(pnt_life, color)
            DrawHead(pnt_life, color)

        cv2.imshow("camera_trackview", img2_bgr)


# Tries to choose unique homography decomposition from two possible ones (for each frame).
# source: "Advances in Unmanned Aerial Vehicles", Kimon P. Valavanis, Springer, 2007, page 285
def ChooseSingleHomographyDecomposition(plane_normal_pair_list, thr, debug):
    # match the head N with all other N-candidates
    headN1, headN2 = plane_normal_pair_list[0]

    while True:
        match_count_n1 = 0
        match_count_n2 = 0
        frame_count = len(plane_normal_pair_list)
        if frame_count == 1:
            break
        for i in range(1, frame_count):
            n1, n2 = plane_normal_pair_list[i]
            d11 = LA.norm(headN1 - n1)
            d12 = LA.norm(headN1 - n2)
            d21 = LA.norm(headN2 - n1)
            d22 = LA.norm(headN2 - n2)
            matched_headN1 = d11 < thr or d12 < thr
            matched_headN2 = d21 < thr or d22 < thr
            if matched_headN1:
                match_count_n1 += 1
            if matched_headN2:
                match_count_n2 += 1

        matched_headN1 = match_count_n1 == frame_count - 1
        matched_headN2 = match_count_n2 == frame_count - 1
        if not matched_headN1 and not matched_headN2:
            # both decomposition are not found => increase threshold
            thr *= 1.1
            print("retrying with larger thr={} c1={} c2={}".format(thr, match_count_n1, match_count_n2))
            continue
        elif matched_headN1 and matched_headN2:
            # both decomposition are found => decrease threshold
            thr *= 0.9
            print("retrying with smaller thr={} c1={} c2={}".format(thr, match_count_n1, match_count_n2))
            continue
        else:
            break

    print("got unique match thr={} c1={} c2={}".format(thr, match_count_n1, match_count_n2))


class ReconstructDemo:
    def __init__(self):
        self.imgLeft = None
        self.imgRight = None
        self.xs1 = None
        self.xs2 = None
        self.xs_per_image = None
        self.xs3d = [] # 3D structure (mapping in SLAM)
        self.world_to_cam_R = [] # position and orientation of camera for each image (localization in SLAM)
        self.world_to_cam_T = []
        self.world_to_cam_R_alt = [] # alternative decomposition of planar homography
        self.world_to_cam_T_alt = []
        self.simplices = None
        self.eye = None
        self.center = None
        self.up = None
        self.orthoRadius = None
        self.scene_scale = None
        self.do_computation_flag = True
        self.processed_images_counter = 0
        self.world_map_changed_flag = False
        self.continue_computation_lock = threading.Lock()
        self.cameras_lock = threading.Lock()

    def draw2DCorners(self, image, xs, circleColor, textColor):
        for i in range(0, len(xs)):
            pt = xs[i]
            ptInt = (int(pt[0]), int(pt[1]))
            cv2.circle(image, ptInt, 5, circleColor)
            if textColor is not None:
                cv2.putText(image, str(i), ptInt, cv2.FONT_HERSHEY_PLAIN, 1, textColor)

    def run(self):
        # !!
        reconstruct = -5 # 1=projective, 2=affine, 3=metric, (-4)=multi-view
        # target data are ['roshen','corridor', 'shed']
        #dataTypeStr = 'corridor'
        #dataTypeStr = 'roshen'
        #dataTypeStr = 'shed'
        #dataTypeStr = 'chapel'
        #dataTypeStr = 'two_ortho_chess'
        dataTypeStr = 'roshen_sweets'
        cam_mat = None
        dist_coeffs = None
        # load our example image and convert it to grayscale
        if dataTypeStr == 'corridor':
            # image1 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\bt.000.png")
            # image2 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\bt.002.png")
            image1 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/bt.000.png")
            image2 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/bt.002.png")
        elif dataTypeStr == 'roshen':
            # calibration matrix
            # TODO: expected det(K)==1, which is false for cam_mat below (MASKS page 178)
            cam_mat = np.array([
                [5.2696329424435044e+002, 0., 3.2307199373780122e+002],
                [0., 5.2746802103114874e+002, 2.4116033688735058e+002],
                [0., 0., 1.]])
            dist_coeffs = [1.5360470953430292e-002, 1.4266851380984227e-001, -7.5601836600441033e-005,
                          -4.0737099186333663e-004, -4.4406644543259827e-001]

            # image1 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\IMG_5453.JPG")
            # image2 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\IMG_5455.JPG")
            image1 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/IMG_5453.JPG")
            image2 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/IMG_5455.JPG")
        elif dataTypeStr == 'shed':
            # image1 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\shed9.3a.png")
            # image2 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\shed9.3b.png")
            image1 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/shed9.3a.png")
            image2 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/shed9.3b.png")
        elif dataTypeStr == 'chapel':
            # image1 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\chapel00.png")
            # image2 = cv2.imread(R"E:\devb\bookax\MultViewGeomCV\chapel01.png ")
            image1 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/chapel00.png")
            image2 = cv2.imread(R"/media/mmore/sam642/devb/bookax/MultViewGeomCV/chapel01.png")
        elif dataTypeStr == 'two_ortho_chess':
            image1 = cv2.imread(R"/media/mmore/sam642/mediata/calib_my/two_ortho_chess/a01_marks.png")
            image2 = cv2.imread(R"/media/mmore/sam642/mediata/calib_my/two_ortho_chess/a02.png")
            #image2 = cv2.imread(R"/media/mmore/sam642/mediata/calib_my/two_ortho_chess/a03.png")
            # calibration matrix
            # TODO: expected det(K)==1, which is false for cam_mat below (MASKS page 178)
            cam_mat = np.array([
                [5.7231451642124046e+02, 0., 3.2393613004134221e+02],
                [0., 5.7231451642124046e+02, 2.8464798761067397e+02],
                [0., 0., 1.]])
            dist_coeffs = [3.8077767767090376e-02, -5.2525417414228982e-02, 1.4627079775487713e-03,
                          -8.6358849036256250e-04, -3.0821297243038270e-01]
        elif dataTypeStr == 'roshen_sweets':
            image1 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_0.png")
            image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_3.png")
            #image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_57.png")
            # image1 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_6.png")
            # image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_13.png")
            cam_mat = np.array([
                [5.7231451642124046e+02, 0., 3.2393613004134221e+02],
                [0., 5.7231451642124046e+02, 2.8464798761067397e+02],
                [0., 0., 1.]])
        self.imgLeft = image1.copy()
        self.imgRight = image2.copy()
        cv2.imshow("1", image1)
        cv2.imshow("2", image2)
        # cv2.waitKey(0)

        # initial statistics
        print("width,height={0}".format((self.imgLeft.shape[1],self.imgLeft.shape[0])))
        if not cam_mat is None:
            print("cam_mat=\n{0}".format(cam_mat))
            print("dist_coeffs={0}".format(dist_coeffs))

        # corridor, translational motion
        #epipole1 = [160.90405853   73.41662142    1.] using OpenCV.findFundamentalMat
        #epipole2 = [174.12939037   79.1102373     1.]
        bt0_xs = -1+np.float32([
                    (447.0, 294),  # right square top
                    (431.0, 160),  # right, second placard, top right
                    #(67.0, 179),  # left placard, top left
                    #(230.0, 436),  # D bot left
                    #(257.0, 275),  # O top left
                    #(148.0, 92),  # left video cam, top left
                    (125.0, 28),  # top left ridge
                    #(175.0, 193),  # second column, top left
                    (236.0, 146), # far top left corner
                    (308.0, 257), # far bot right corner
                    (236.0, 266), # far bot left corner
                    (295.0, 149),  # far air-conditioner, top right
                    #(40.0,  452),  # left-bot pad, left corner
                    (107.0, 461), # left-bot pad, right corner
                    (398.0, 447),  # right stick, upper corner
                    (376.0, 28)])   # lamp, top-right
        bt2_xs = -1+np.float32([
                    (497.0, 320),
                    (473.0, 158),
                    #(55.0, 178),
                    #(235.0, 470),
                    #(266.0, 280),
                    #(146.0, 81),
                    (115.0, 4),
                    #(179.0, 193),
                    (244.0, 145),
                    (319.0, 261),
                    (242.0, 257),
                    (305.0, 148),
                    #(18.0,  489),
                    (95.0, 501),
                    (429.0, 487),
                    (403.0, 7)])
        # roshen sweets
        rosh1_xs = -1+np.float32([
                    (62.2391, 187.9534),
                    (68.0634, 141.0438),
                    (86.8354, 276.6491),
                    (108.1154, 103.3310),
                    (109.6269, 109.9483),
                    (112.2884, 195.7367),
                    (145.5717, 135.8002),
                    (149.9615, 168.4625),
                    (159.1230, 81.3783),
                    (166.8681, 178.4082),
                    (183.2870, 195.4033),
                    (187.8188, 140.0263),
                    (189.8109, 188.3519),
                    (206.0901, 246.0901),
                    (206.8354, 307.9534),
                    (224.8585, 211.7702),
                    (239.5999, 321.8099),
                    (240.5384, 240.3688),
                    (273.6229, 244.9049),
                    (280.2772, 159.4400),
                    (293.2481, 250.1299),
                    (304.3107, 170.7399)])
        rosh2_xs = -1+np.float32([
                    (156.1340, 156.2624),
                    (168.6091, 123.2275),
                    (154.0795, 225.0907),
                    (219.1071, 106.1089),
                    (216.0717, 111.0698),
                    (176.6379, 170.7189),
                    (227.2236, 139.0764),
                    (212.3894, 163.4271),
                    (271.0669, 105.0033),
                    (219.3247, 174.9103),
                    (222.0045, 191.7659),
                    (256.1099, 154.3599),
                    (229.9216, 188.9016),
                    (219.8258, 237.4181),
                    (212.6348, 287.7552),
                    (243.5109, 217.1227),
                    (234.1266, 309.4246),
                    (242.7752, 243.0338),
                    (288.6926, 264.0208),
                    (319.7820, 198.2371),
                    (315.7797, 278.9001),
                    (335.3662, 214.7986)])
        shed1_xs = -1+np.float32([
                    (41.0, 553),  #% 1st-2nd pillar, 4st dirt to the left
                    (182.0, 494), # 1st pillar, bottom right
                    (156.0, 324), # 1st pillar, up, roof corner
                    (268.0, 345), #% left window, top left
                    (296.0, 213), # ridge, top back
                    (823.0, 139), # ridge, top front
                    (782.0, 217), # front edge top
                    (788.0, 535), # front edge bottom
                    (729.0, 311), # 4th pillar top
                    (906.0, 440), # back door panel, bot left
                    (493.0, 528), #% 3rd pillar, bottom right
                    (480.0, 322), # 3rd pillar, top left
                    (595.0, 417), #% right window, bot left
                    (779.0, 614), #% 4th pillar, brush to the bottom-left
                    (697.0, 548), # 4th pillar's base
                    (927.0, 497), # back door bot right
                    (926.0, 399), # back door top right
                    ])
        shed2_xs = -1+np.float32([
                    (42.0, 514), #%
                    (239.0, 472),
                    (217.0, 323),
                    (318.0, 339), #%
                    (367.0, 215),
                    (683.0, 106),
                    (581.0, 198),
                    (586.0, 550),
                    (457.0, 308),
                    (896.0, 448),
                    (357.0, 520), #%
                    (346.0, 321),
                    (473.0, 413), #%
                    (317.0, 628), #%
                    (430.0, 553),
                    (949.0, 517),
                    (951.0, 401)])
        chapel1_xs = -1+np.float32([
            [27, 239], # dirt, bottom left
            [91, 225], # left vertical edge, bottom corner
            [89, 141], # left vertical edge, top, under the roof
            [233, 248], # central vertical edge, bottom corner
            [226, 119], # central vertical edge, top, under the roof
            [413, 223], # right vertical edge, bottom, corner
            [399, 133], # right vertical edge (pseudo), to the up-left of wicket
            [144, 20], # roof top left corner, near the cross
            [371, 61], # roof top right corner, intersection of horizontal and slant edges
            [453, 134] # roof bottom right corner, intersection of vanishing lines
        ])
        chapel2_xs = -1+np.float32([
            [47, 254],
            [111, 238],
            [108, 151],
            [279, 259],
            [269, 128],
            [433, 233],
            [420, 143],
            [173, 27],
            [379, 73],
            [462, 143]
        ])
        two_ortho_chess_xs1 = np.float32([
            [67, 36],  # row1
            [180, 31],
            [357, 25],
            [476, 25],
            [546, 20],
            [125, 132],  # row2
            [260, 149],
            [426, 140],
            [507, 115],
            [86, 210], # row3
            [194, 239],
            [358, 275],
            [466, 219],
            [522, 181],
            [143, 305], # row4
            [272, 350],
            [417, 334],
            [490, 273],
            [104, 354], # row5
            [213, 401],
            [361, 464],
            [445, 373],
            [504, 314]
        ])
        two_ortho_chess_xs2 = np.float32([
            [182, 157], # row1
            [234, 102],
            [318, 11],
            [477, 79],
            [594, 123],
            [192, 199], # row2
            [260, 146],
            [404, 137],
            [543, 182],
            [163, 283], # row3
            [212, 259],
            [308, 219],
            [482, 253],
            [596, 280],
            [171, 351], # row4
            [244, 341],
            [404, 343],
            [548, 359],
            [139, 429], # row5
            [198, 444],
            [297, 461],
            [476, 458],
            [602, 456],
        ])
        two_ortho_chess_xs3 = np.float32([
            [35, 61], # row1
            [128, 117],
            [277, 204],
            [450, 90],
            [570, 8],
            [102, 174], # row2
            [213, 253],
            [378, 239],
            [505, 139],
            [92, 212], # row3
            [176, 282],
            [309, 371],
            [445, 258],
            [533, 171],
            [152, 303], # row4
            [253, 383],
            [386, 371],
            [484, 272],
            [135, 319], # row5
            [219, 386],
            [330, 475],
            [429, 365],
            [510, 280],
        ])

        if dataTypeStr == 'corridor':
            self.xs1 = bt0_xs
            self.xs2 = bt2_xs
        elif dataTypeStr == 'roshen':
            self.xs1 = rosh1_xs
            self.xs2 = rosh2_xs
        elif dataTypeStr == 'shed':
            self.xs1 = shed1_xs
            self.xs2 = shed2_xs
        elif dataTypeStr == 'chapel':
            self.xs1 = chapel1_xs
            self.xs2 = chapel2_xs
        elif dataTypeStr == 'two_ortho_chess':
            self.xs_per_image = [two_ortho_chess_xs1, two_ortho_chess_xs2, two_ortho_chess_xs3]
            self.xs1 = self.xs_per_image[0]
            self.xs2 = self.xs_per_image[1]

        if dataTypeStr == 'roshen':
            p1M = np.array([
                [40.861664, 525.83868, 322.31946, 156175.52],
                [-188.35384, -159.00006, 524.99774, 100498.09],
                [0.66997451, 0.11253661, 0.73380494, 379.86969]])
            p2M = np.array([
                [344.97925, 452.42664, 241.58908, 156862.33],
                [-154.63869, -8.3352928, 558.92596, 134420.17],
                [0.74725044, -0.19494215, 0.63530648, 397.72992]])
            print("p1M=\n{0}".format(p1M))
            print("p2M=\n{0}".format(p2M))
            princip1M = p1M[-1,0:3]
            princip1M = princip1M / LA.norm(princip1M)
            princip2M = p2M[-1,0:3]
            princip2M = princip2M / LA.norm(princip2M)
            print("princip1M={0} princip2M={1}".format(princip1M,princip2M))

        # draw point correspondence
        if not self.xs1 is None:
            circleColor = (0, 255, 0)
            textColor = (0, 0, 255)
            self.draw2DCorners(image1, self.xs1, circleColor, textColor)
            self.draw2DCorners(image2, self.xs2, circleColor, textColor)
            cv2.imshow("1", image1)
            cv2.imshow("2", image2)
            cv2.waitKey(0)

        xs_per_image_meter = []
        if not cam_mat is None:
            #convertPixelToMeterPoints(cam_mat, self.xs_per_image, xs_per_image_meter)

            #experimentWithEssentialMat(cam_mat, xs_per_image_meter)

            #self.xs3d = GenerateTwoOrthoChess()

            if False:
                #
                suc, frames_R, frames_T, world_pnts = calcCameras(xs_per_image_meter)
                if suc:
                    points_num = len(xs_per_image_meter[0])
                    self.xs3d = [world_pnts[i,:] for i in range(0,points_num)]
                    frames_num = len(xs_per_image_meter)
                    self.world_to_cam_R = [frames_R[i, :, :] for i in range(0, frames_num)]
                    self.world_to_cam_T = [frames_T[i, :] for i in range(0, frames_num)]
            if True:
                self.world_to_cam_R, self.world_to_cam_T = RunReconstructionSequential()
                self.xs3d = []


        #
        if not self.xs1 is None:
            fundMatCV,_ = cv2.findFundamentalMat(self.xs1, self.xs2)
            print("Fundamental matrix OpenCV=\n{0}".format(fundMatCV))

            branchOpenCV = True
            if branchOpenCV:
                (dVec, u, vt) = cv2.SVDecomp(fundMatCV)
                epi1 = vt.T[:, -1]
                epi1 = epi1 / epi1[-1]
                print("OpenCV epipole1={0}".format(epi1))

                (dVec, u, vt) = cv2.SVDecomp(fundMatCV.T)
                epi2 = vt.T[:, -1]
                epi2 = epi2 / epi2[-1]
                print("OpenCV epipole2={0}".format(epi2))


            fundMatNoNorm = findFundamentalMatBasic8PointCore(self.xs1, self.xs2)
            fundMatNoNorm = normLastCellToOne(fundMatNoNorm)
            print("Fundamental matrix no norm=\n{0}".format(fundMatNoNorm))

            printNoNormImgCoord = True
            if printNoNormImgCoord:
                (dVec, u, vt) = cv2.SVDecomp(fundMatNoNorm)
                epi1 = vt.T[:, -1]
                epi1 = epi1 / epi1[-1]
                print("fundMatNoNorm epipole1={0}".format(epi1))

                (dVec, u, vt) = cv2.SVDecomp(fundMatNoNorm.T)
                epi2 = vt.T[:, -1]
                epi2 = epi2 / epi2[-1]
                print("fundMatNoNorm epipole2={0}".format(epi2))

            fundMat = findFundamentalMatBasic8Point(self.xs1, self.xs2)

            # fundamental matrix is skew symmetric for pure translational motion
            normLastOne = True
            if dataTypeStr == 'corridor123':
                normLastOne = False

            # OpenCV normalizes F so that the bottom right element = 1
            # For translational motion F=[epi] and last element=0 so normalization is impossible
            if normLastOne:
                fundMat = normLastCellToOne(fundMat)
                print("Fundamental matrix=\n{0}".format(fundMat))

            if dataTypeStr == 'corridor123':
                fundMat = makeSkewSymmetric(fundMat)
                print("Fundamental matrix skew symmetric=\n{0}".format(fundMat))
            #fundMat = fundMatCV # !!
            #fundMat = fundMatFromEss

            # find epipoles (epi1=null(F), epi2=null(F.transpose))
            (dVec, u, vt) = cv2.SVDecomp(fundMat)
            epi1 = vt.T[:, -1]
            epi1 = epi1 / epi1[-1]
            print("epipole1={0}".format(epi1))
            cv2.circle(image1, (int(epi1[0]), int(epi1[1])), 5, (0, 0, 255))

            (dVec, u, vt) = cv2.SVDecomp(fundMat.T)
            epi2 = vt.T[:, -1]
            epi2 = epi2 / epi2[-1]
            print("epipole2={0}".format(epi2))
            cv2.circle(image2, (int(epi2[0]), int(epi2[1])), 5, (0, 0, 255))

            # MVGCV Result 9.14 page 256
            # MASKS formula 6.32 page 189
            p1Proj = np.hstack((np.eye(3), np.zeros((3, 1))))
            p2Proj = np.hstack((np.dot(skewSymmeticMat(epi2), fundMat), epi2.reshape(3, 1)))
            p2ProjMasks = np.hstack((np.dot(skewSymmeticMat(epi2).T, fundMat), epi2.reshape(3, 1)))
            print("p1Proj=\n{0}".format(p1Proj))
            print("p2Proj=\n{0}".format(p2Proj))
            print("p2ProjMASKS=\n{0}".format(p2ProjMasks))

            # check that projection matrices P1 and P2 create expected fundamental matrix F
            checkProjMat = True
            if checkProjMat:
                # F = skew(T)*P2[:,0:3]
                fundMatPM = np.dot(skewSymmeticMat(p2Proj[:, -1]), p2Proj[:, 0:3])
                fundMatPM = fundMatPM / fundMatPM[-1,-1]
                err = LA.norm(fundMat - fundMatPM)
                #v1 = fundMat.ravel();
                #v2 = fundMatPM.ravel();
                #err2=np.dot(v1,v2)/LA.norm(v1)**2 - 1
                print("check ProjMat->FundMat error={0}".format(err))

            # fix image points correspondence in compliance to fundamental matrix
            fixXs1, fixXs2 = correctPointCorrespondencePoly6(fundMat, self.xs1, self.xs2)
            self.xs1 = fixXs1
            self.xs2 = fixXs2
            #fixXs1 = self.xs1
            #fixXs2 = self.xs2

            xs_per_image_meter = []
            if not cam_mat is None and False:
                convertPixelToMeterPoints(cam_mat, [fixXs1, fixXs2, self.xs_per_image[2]], xs_per_image_meter)

                experimentWithEssentialMat(cam_mat, xs_per_image_meter) # TODO: crashes!!!

                #
                suc, frames_R, frames_T, world_pnts = calcCameras(xs_per_image_meter)
                if suc:
                    points_num = len(xs_per_image_meter[0])
                    self.xs3d = [world_pnts[i, :] for i in range(0, points_num)]
                    frames_num = len(xs_per_image_meter)
                    self.world_to_cam_R = [frames_R[i, :, :] for i in range(0, frames_num)]
                    self.world_to_cam_T = [frames_T[i, :] for i in range(0, frames_num)]

            # for i in range(0, len(self.xs1)):
            #     m1 = abs(self.xs1[i] - fixXs1[i]).max()
            #     m2 = abs(self.xs2[i] - fixXs2[i]).max()
            #     print("i={0} xs1={1} fix1={2} dif1={3} xs2={4} fix2={5} dif2={6}".format(i, self.xs1[i], fixXs1[i], m1, self.xs2[i], fixXs2[i], m2))
            # pixDiffMax=abs(np.vstack((self.xs1 - fixXs1, self.xs2-fixXs2)).ravel()).max()
            # print("pixDiffMax={0}".format(pixDiffMax))
            # self.draw2DCorners(image1, fixXs1, (255, 0, 0), None)
            # self.draw2DCorners(image2, fixXs2, (255, 0, 0), None)

            cv2.imshow("1", image1)
            cv2.imshow("2", image2)

            # calcProjHomog = True
            # if calcProjHomog:
            #     # homography
            #     homogCV,_ = cv2.findHomography(self.xs1, self.xs2) # works badly
            #     print("homogCV=\n{0}".format(homogCV))
            #     homogMan = findHomogDlt(self.xs1, self.xs2) # works badly
            #     print("homogMan=\n{0}".format(homogMan))
            #
            #     # Hinf TODO: doesn't work for projective cameras?
            #     # suc,res1=cv2.invert(p1Proj[:,0:3])
            #     # homogInfProj=np.dot(p2Proj[:,0:3], res1) # TODO: doesn't work?
            #     # homogInfProj=homogInfProj / homogInfProj[-1,-1]
            #     # print("homogInfProj=\n{0}".format(homogInfProj))
            #
            #     drawHomog = True
            #     if drawHomog:
            #         reprojXs2 = mapHomography(self.xs1, homogMan, normHomog=True)
            #         for i in range(0, len(self.xs1)):
            #             print("xs2={0} reXs2={1}".format(self.xs2[i], reprojXs2[i]))
            #         imgRe2 = self.imgRight.copy()
            #         self.draw2DCorners(imgRe2, self.xs2, (0, 255, 0), (0, 255, 0))
            #         self.draw2DCorners(imgRe2, reprojXs2, (0, 0, 255), textColor)
            #         cv2.imshow("homogProj", imgRe2)


            if 'p1M' in locals() and hasattr(self, 'xs1') and False:
                # TODO: CAMERA-CHAIN cameras are connected so that p2 = p1*H is wrong?????????
                suc, h123 = computeP1P2Transform(p1Proj, p2Proj, p1M, p2M)

        # affine reconstruction
        if reconstruct >= 2:
            p1Affine = None
            p2Affine = None
            HomogInf = None
            if dataTypeStr == 'corridor':
                # corridor
                # vx = (1.0, 0, 0)
                # vy = (0, 1.0, 0)
                vz1=(274.0, 187, 1) # screen center
                # vertical line
                # line1 (160,145,1)x(163,432,1)=line(-287,3,45485)
                # line2 (199,168,1)x(199,452,1)=line(-284,0,56516)
                vy1=(199,3876,1) # vertical inf, bottom
                # 2nd image
                vz2=(281.0, 190, 1)

                # vanish1 = np.array([vz1])
                # vanish2 = np.array([vz2])
                # p1Affine, p2Affine = affineReconstruct3V(vanish1, vanish2)

                # translational motion
                p1Affine = np.hstack((np.eye(3), np.zeros((3,1))))
                p2Affine = np.hstack((np.eye(3), epi2.reshape((3,1))))

            elif dataTypeStr == 'roshen':
                # roshen
                vx1 = (46.0, -302, 1) # rig X
                vy1 = (554.0, -113, 1) # box base, top-right direction
                vz1 = (218.0, 892, 1) # box base, bottom direction
                # (397,306) rig bot left
                # (618,216) rig bot right
                # line (397,306,1)x(618,216,1)=(90,221,-103356)
                # (295,128) rig top left
                # (457,70) rig top right
                # line (295,128,1)x(457,70,1)=(90,221,-103356)
                # rig Y =crossing of two lines = (4755,-1470)
                # 2nd image
                vx2 = (461.0, -218, 1) # rig X
                vy2 = (965.0, 32, 1) # box base, top-right direction
                vz2 = (164.0, 773, 1) # box base, bottom direction
                #vrigY = (-2307.0, 47, 1) # rig Y; has no pair!
                vanish1Meas = np.array([vx1, vy1, vz1])
                vanish2Meas = np.array([vx2, vy2, vz2])
            elif dataTypeStr == 'shed':
                # 1st image
                # imageTopLeft={1129,2866}
                vx1 = (-1041.0, 381, 1) # shed to the left, {85, 3247}-imageTopLeft
                vy1 = (1315.0, 375, 1) # shed to the right, {2440, 3226}-imageTopLeft
                vz1 = (0.0, -1, 0) # inf up-down
                #vz = (512.0, -12260, 1) # inf up
                vr1 = (1295.0, -588, 1) # left side of roof, {2424, 2278}-imageTopLeft
                #vm1 = (1495.0, -2675, 1) # across the middle of the building, {2624, 191}-imageTopLeft
                # dim=(1024,768) center=(512, 384)
                # vx = (-1554.0, 3, 1) # shed to the left
                # vy = (588.0, 9, 1) # shed to the right
                # vz = (0.0, 1226, 1) # inf up-down
                # vx = (2.0, 0, 0, 1) # shed to the left
                # vy = (0.0, 3, 0, 1) # shed to the right
                # vz = (0.0, 0, 5, 1) # inf up-down
                #
                # vx = (2.0, 0, 0) # shed to the left
                # vy = (0.0, 3, 0) # shed to the right
                # vz = (0.0, 0, 5) # inf up-down

                # 2nd image
                vx2 = (-70.0, 360, 1) # shed to the left
                vy2 = (2441.0, 343, 1) # shed to the right
                vz2 = (0.0, -1, 0) # inf up-down
                #vz = (640.0, -3500, 1) # inf up
                vr2 = (2669.0, -1383, 1) # left side of roof
                # center=(512,384)
                # vx = (1929.0, 37, 1) # to the right
                # vy = (-582.0, 24, 1) # to the left
                # vz = (128.0, 734, 1) # inf up-down

                vanish1Meas = np.array([vx1, vy1, vr1])
                vanish2Meas = np.array([vx2, vy2, vr2])
                # vanish1 = np.array([vx1, vy1, vz1]) # inf vanishing points
                # vanish2 = np.array([vx2, vy2, vz2])
            elif dataTypeStr == 'chapel':
                # AutoCad
                vx1 = np.float64([914.3, 162.3, 1]) # chapel to the right
                vy1 = np.float64([-397.3, 219.4, 1]) # chapel to the left
                vr1 = np.float64([-471.9, -695.3, 1])# slant roof directed to the left
                vx2 = np.float64([812.8, 172.1, 1])
                vy2 = np.float64([-520.9, 240.3, 1])
                vr2 = np.float64([-1216.8, -1372.8, 1])
                vanish1Meas = np.array([vx1, vy1, vr1]) # NOTE: vz is unused
                vanish2Meas = np.array([vx2, vy2, vr2])
                # MsPaint
                # vx1 = np.float64([921, 162, 1]) # chapel to the right
                # vy1 = np.float64([-512, 222, 1]) # chapel to the left
                # vr1 = np.float64([-475, -707, 1]) # slant roof directed to the left
                # vz1 = np.float64([12, -2466, 1]) # chapel to the top
                # vx2 = np.float64([817, 170, 1])
                # vy2 = np.float64([-624, 247, 1])
                # vr2 = np.float64([-641, -790, 1])
                # vz2 = np.float64([-17, -2694, 1])
                # vanish1Meas = np.array([vx1, vy1, vr1, vz1])
                # vanish2Meas = np.array([vx2, vy2, vr2, vz2])

            if p1Affine is None:
                vanish1Fix,vanish2Fix = correctPointCorrespondencePoly6(fundMat, vanish1Meas, vanish2Meas)
                p1Affine, p2Affine, HomogInf = affineReconstruct3V(fundMat, p1Proj, p2Proj, vanish1Fix, vanish2Fix, epi1, epi2)

                ## Fa=epi2 x H
                # fundMatAffine=np.dot(skewSymmeticMat(epi2), HomogInf)
                # print("fundMatAffine=\n{0}".format(fundMatAffine))

            print("p1Affine=\n{0}".format(p1Affine))
            print("p2Affine=\n{0}".format(p2Affine))

            # check that projection matrices P1 and P2 create expected fundamental matrix F
            checkAffMat = True
            if checkAffMat:
                # F = skew(T)*P2[:,0:3]
                fundMatAM = np.dot(skewSymmeticMat(p2Affine[:, -1]), p2Affine[:, 0:3])
                fundMatAM = fundMatAM / fundMatAM[-1, -1]
                err = LA.norm(fundMat - fundMatAM)
                # v1 = fundMat.ravel();
                # v2 = fundMatPM.ravel();
                # err2=np.dot(v1,v2)/LA.norm(v1)**2 - 1
                print("check AffMat->FundMat error={0}".format(err))


                # # Hinf on affine cameras
            # suc,res1=cv2.invert(p1Affine[:,0:3])
            # hinfAffine=np.dot(p2Affine[:,0:3], res1)
            # hinfAffine=hinfAffine / hinfAffine[-1,-1]
            #
            # drawHomog = True
            # if drawHomog:
            #     reprojXs2 = mapHomography(self.xs1, hinfAffine, normHomog=True)
            #     for i in range(0, len(self.xs1)):
            #         print("i={0} xs2={1} reXs2={2}".format(i, self.xs2[i], reprojXs2[i]))
            #     imgRe2 = self.imgRight.copy()
            #     self.draw2DCorners(imgRe2, self.xs2, (0, 255, 0), (0, 255, 0))
            #     self.draw2DCorners(imgRe2, reprojXs2, (0, 0, 255), textColor)
            #     cv2.imshow("hinfAffine", imgRe2)

        # reconstruct the camera internal matrix
        # NOTE: if we do inversion at first and then Cholesky decomposition, the things don't work (get invalid camera matrix)
        def imgAbsConicDecompose(w):
            if w is None: return None
            KInv=LA.cholesky(w)
            K=LA.inv(KInv).T
            K=K/K[-1,-1]
            return K

        # metric reconstruction
        if reconstruct >= 3:
            height, width = self.imgLeft.shape[0:2]

            halfWidth=self.imgLeft.shape[1] / 2.0 # half width
            halfHeight=self.imgLeft.shape[0] / 2.0
            defFovX, defFovY = 60, 50

            wAllDef = imageOfAbsConic_AllDefault(width, height, defFovX, defFovY)
            print("IAC all default w=\n{0}".format(wAllDef))
            print("KAllDef=\n{0}".format(imgAbsConicDecompose(wAllDef)))
            w = wAllDef

            if dataTypeStr == 'corridor':
                w = imageOfAbsConic_DefaultCenterSquarePixels(width, height, vz1, vy1)
                print("wDef=\n{0}".format(w))
                w = wAllDef
            if dataTypeStr == 'roshen' and False:
                p1Metric = p1M
                p2Metric = p2M
            if dataTypeStr == 'roshen':
                # from known calibration matrix
                assert not cam_mat is None
                # image of absolute conic (IAC)
                suc, camMatInv = cv2.invert(cam_mat)
                wKK = np.dot(camMatInv.T, camMatInv)
                print("IAC from KK w=\n{0}".format(wKK))

                #
                wYZ = imageOfAbsConic_DefaultCenterSquarePixels(width, height, vy1, vz1)
                print("wYZ1=\n{0}".format(wYZ))
                wYZ2 = imageOfAbsConic_DefaultCenterSquarePixels(width, height, vy2, vz2)
                print("wYZ2=\n{0}".format(wYZ2))
                w = wAllDef
            if dataTypeStr == "shed":
                wXY = imageOfAbsConic_DefaultCenterSquarePixels(width, height, vx1, vy1)
                print("wXY1=\n{0}".format(wXY))
                wXY2 = imageOfAbsConic_DefaultCenterSquarePixels(width, height, vx2, vy2)
                print("wXY2=\n{0}".format(wXY))

                lineInf1 = np.cross(vy1, vr1)
                #lineInf1 = lineInf1 / lineInf1[-1] # NOTE: mandatory normalization!
                wVL = imageOfAbsConic_OrthoLineAndPlane(width, height, [(vx1,vy1), (vx1,vr1)], [(vx1,lineInf1)])
                print("wVL=\n{0}".format(wVL))
                w = wXY
            if dataTypeStr == "chapel":
                vx1 = vanish1Fix[0]
                vx2 = vanish2Fix[0]
                vy1 = vanish1Fix[1]
                vy2 = vanish2Fix[1]
                vr1 = vanish1Fix[2]
                vr2 = vanish2Fix[2]
                #vz1 = vanish1Fix[3]
                #vz2 = vanish2Fix[3]
                wXY = imageOfAbsConic_DefaultCenterSquarePixels(width, height, vx1, vy1)
                print("wXY1=\n{0}".format(wXY))
                print("KXY1=\n{0}".format(imgAbsConicDecompose(wXY)))

                wAxAyXR1 = imageOfAbsConic_DefaultCenter2(width, height, [(vx1, vy1),(vx1, vr1)])
                print("wAxAyXR1=\n{0}".format(wAxAyXR1))
                print("KAxAyXR1=\n{0}".format(imgAbsConicDecompose(wAxAyXR1)))
                # wAxAy = imageOfAbsConic_DefaultCenter2(width, height, [(vx1, vy1),(vx1, vz1)])
                # print("wAxAyXZ1=\n{0}".format(wAxAy))
                # print("KAxAyXZ1=\n{0}".format(imgAbsConicDecompose(wAxAy)))
                wAxAyXR2 = imageOfAbsConic_DefaultCenter2(width, height, [(vx2, vy2),(vx2, vr2)])
                print("wAxAyXR2=\n{0}".format(wAxAyXR2))
                print("KAxAyXR2=\n{0}".format(imgAbsConicDecompose(wAxAyXR2)))
                # wAxAy = imageOfAbsConic_DefaultCenter2(width, height, [(vx2, vy2),(vx2, vz2)])
                # print("wAxAyXZ2=\n{0}".format(wAxAy))
                # print("KAxAyXZ2=\n{0}".format(imgAbsConicDecompose(wAxAy)))

                lineInf1 = np.cross(vy1, vr1)
                #lineInf1 = lineInf1 / lineInf1[-1] # NOTE: mandatory normalization!
                lineInf2 = np.cross(vy2, vr2)
                #lineInf2 = lineInf2 / lineInf2[-1
                orthoPnts = [(vx1, vy1),(vx1, vr1),(vx2, vy2), (vx2, vr2)]
                #orthoPnts = [(vx1, vy1),(vx1, vr1),(vx1, vz1),(vx2, vy2), (vx2, vr2), (vx2, vz2)]
                orthoVanishLines = [(vx1,lineInf1), (vx2, lineInf2)]
                #orthoVanishLines = []
                wVV = imageOfAbsConic_OrthoLineAndPlane(width, height, orthoPnts, orthoVanishLines)
                print("wVV=\n{0}".format(wVV))
                print("KVV=\n{0}".format(imgAbsConicDecompose(wVV)))

                # same camera
                #wSameCam = imageOfAbsConic_SameCameraOrthoVanish(HomogInf, [(vx1, vy1)])
                wSameCam = imageOfAbsConic_SameCameraOrthoVanish(HomogInf, [(vx1, vy1),(vx2, vy2)])
                #wSameCam = imageOfAbsConic_SameCameraOrthoVanish(HomogInf, [(vx1, vy1),(vx1, vr1),(vx2, vy2), (vx2, vr2)])
                #wSameCam = imageOfAbsConic_SameCameraOrthoVanish(HomogInf, [(vx1, vy1),(vx1, vr1),(vx1, vz1),(vx2, vy2), (vx2, vr2), (vx2, vz2)])
                print("wSameCam=\n{0}".format(wSameCam))
                ###print("KSameCam=\n{0}".format(imgAbsConicDecompose(wSameCam))) Error: Matrix is not positive definite

                #w = wVV
                w = wXY
                #w = wSameCam

            # dump camera calibration matrix
            K=imgAbsConicDecompose(w)
            print("K=\n{0}".format(K))

            h1MetricInv = calcAffineToMetricTransform(p1Affine, w)
            h2MetricInv = calcAffineToMetricTransform(p2Affine, w)
            p1Metric = np.dot(p1Affine, h1MetricInv)
            p2Metric = np.dot(p2Affine, h2MetricInv)
            print("p1Metric=\n{0}".format(p1Metric))
            print("p2Metric=\n{0}".format(p2Metric))
            princip1 = p1Metric[-1,0:3]
            princip1 = princip1 / LA.norm(princip1)
            princip2 = p2Metric[-1,0:3]
            princip2 = princip2 / LA.norm(princip2)
            print("princip1={0} princip2={1}".format(princip1,princip2))

            # test pProj*HAffineInv = pMetric*HMetric
            # define pAffine indirectly from known pMetric, HMetric, pProj
            if 'p1M' in locals():
                h2Metric = LA.inv(h2MetricInv)
                pRight2 = np.dot(p2M, h2Metric)
                c1 = pRight2[0,3] / p2Affine[0,3]
                c2 = pRight2[1,3] / p2Affine[1,3]
                c3 = pRight2[2,3] / p2Affine[2,3]
                # NOTE: c1,c2,c3 should be the same factor (now: -85,-717,397)
                # TODO: CAMERA-CHAIN cameras are connected so that p2 = p1*H is wrong?????????
                pass

        xs1 = self.xs1
        xs2 = self.xs2

        if reconstruct == 1:
            p1 = p1Proj
            p2 = p2Proj
        elif reconstruct == 2:
            p1 = p1Affine
            p2 = p2Affine
        elif reconstruct == 3:
            p1 = p1Metric
            p2 = p2Metric

        if reconstruct >= 1:
            self.xs3d,reprojXs1,reprojXs2 = triangAndReprojectCorrespondences(p1, p2, self.xs1, self.xs2, False)
            # draw reprojected correspondences
            imgRe1 = self.imgLeft.copy()
            imgRe2 = self.imgRight.copy()
            self.draw2DCorners(imgRe1, self.xs1, (0, 255, 0), None)
            self.draw2DCorners(imgRe1, reprojXs1, (0, 0, 255), textColor)
            self.draw2DCorners(imgRe2, self.xs2, (0, 255, 0), None)
            self.draw2DCorners(imgRe2, reprojXs2, (0, 0, 255), textColor)
            #cv2.imshow("1: 3Dback", imgRe1)
            #cv2.imshow("2: 3Dback", imgRe2)
        elif reconstruct == -4:
            pass

        #
        if reconstruct > 0:
            points = np.array(self.xs1)
            tri = Delaunay(points)
            self.simplices = tri.simplices.copy()
            # plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
            # plt.plot(points[:,0], points[:,1], 'o')
            # plt.show()

            for i in range(0, tri.simplices.shape[0]):
                simpl = tri.simplices[i, :]
                pt1 = self.xs1[simpl[0]]
                pt2 = self.xs1[simpl[1]]
                pt3 = self.xs1[simpl[2]]
                cv2.line(image1, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 0, 255))
                cv2.line(image1, (int(pt1[0]), int(pt1[1])), (int(pt3[0]), int(pt3[1])), (0, 0, 255))
                cv2.line(image1, (int(pt2[0]), int(pt2[1])), (int(pt3[0]), int(pt3[1])), (0, 0, 255))
                print("simplexInd={0} simplex={1} x1={2} x2={3} x3={4}".format(i, simpl, pt1, pt2, pt3))
                #cv2.imshow("1", image1)
                #cv2.waitKey(1000)
            cv2.imshow("1", image1)
            cv2.waitKey(1000)

        #
        def mouseHandler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print("point={0}".format((x, y)))

                line2 = np.dot(fundMat, (x, y, 1))
                print("line2={0}".format(line2))

                # L1=F'*[e2]*L2 TODO why???
                line1 = np.dot(np.dot(fundMat.T, skewSymmeticMat(epi2)), line2)

                p1, p2 = lineSegment(line1, imgWidth=image1.shape[1])
                cv2.line(image1, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0))

                p1, p2 = lineSegment(line2, imgWidth=image2.shape[1])
                cv2.line(image2, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0))

                cv2.imshow("1", image1)
                cv2.imshow("2", image2)

        cv2.setMouseCallback("1", mouseHandler, param=None)

        if not self.xs1 is None:
            cv2.waitKey(0)

    def DrawPhysicalCamera(self, camR, camT, cam_to_world):
        # transform to the camera frame
        # cam_to_world=inv(world_to_cam)=[Rt,-Rt.T]
        cam_to_world.fill(0)
        cam_to_world[0:3, 0:3] = camR.T
        cam_to_world[0:3, 3] = -camR.T.dot(camT)
        cam_to_world[-1, -1] = 1

        glPushMatrix()
        glMultMatrixf(cam_to_world.ravel('F'))

        # draw axes in the local coordinates
        ax = 0.2
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(ax, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, ax, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, ax)
        glEnd()

        # draw camera in the local coordinates
        hw = ax / 3  # halfwidth
        cam_skel = np.float32([
            [0, 0, 0],
            [hw, hw, ax],  # left top
            [-hw, hw, ax],  # right top
            [-hw, -hw, ax],  # right bot
            [hw, -hw, ax],  # left bot
        ])
        glLineWidth(1)
        glColor3f(1, 1, 1)
        glBegin(GL_LINE_LOOP)  # left top of the front plane of the camera
        glVertex3fv(cam_skel[1])
        glVertex3fv(cam_skel[2])
        glVertex3fv(cam_skel[3])
        glVertex3fv(cam_skel[4])
        glEnd()
        glBegin(GL_LINES)  # edges from center to the front plane
        glVertex3fv(cam_skel[0])
        glVertex3fv(cam_skel[1])
        glVertex3fv(cam_skel[0])
        glVertex3fv(cam_skel[2])
        glVertex3fv(cam_skel[0])
        glVertex3fv(cam_skel[3])
        glVertex3fv(cam_skel[0])
        glVertex3fv(cam_skel[4])
        glEnd()
        glPopMatrix()

    def DrawCameras(self, draw_camera_each_frame=True):
        cam_to_world = np.eye(4, 4, dtype=np.float32)
        cam_pos_world = np.zeros(4, np.float32)

        with self.cameras_lock:
            # process two alternative R,T/d decompositions of planar homography H
            two_decomps = [(self.world_to_cam_R, self.world_to_cam_T), (self.world_to_cam_R_alt, self.world_to_cam_T_alt)]
            for decomp_ind, (camR_list, camT_list) in enumerate(two_decomps):
                cam_pos_world_prev = np.zeros(4, np.float32)
                cam_pos_world_prev_inited = False

                # draw the center of the axes
                track_col = 0.0 if decomp_ind == 0 else 1.0
                glColor3f(track_col, track_col, track_col)

                for i in range(0, len(camR_list)):
                    camR = camR_list[i]
                    camT = camT_list[i]
                    if camR is None:
                        continue

                    # get position of the camera in the world: cam_to_world*(0,0,0,1)=cam_pos
                    cam_pos_world_tmp = -camR.T.dot(camT)
                    #assert np.isclose(1, cam_pos_world_tmp[-1]), "Expect cam_to_world(3,3)==1"
                    cam_pos_world[0:3] = cam_pos_world_tmp[0:3]

                    # draw trajectory of the camera
                    if cam_pos_world_prev_inited:
                        glBegin(GL_LINES)
                        glColor3f(0, 0, 1)
                        glVertex3fv(cam_pos_world_prev[0:3])
                        glVertex3fv(cam_pos_world[0:3])
                        glEnd()

                    glBegin(GL_POINTS)
                    glVertex3fv(cam_pos_world[0:3])
                    glEnd()

                    if draw_camera_each_frame:
                        self.DrawPhysicalCamera(camR, camT, cam_to_world)

                    cam_pos_world_prev, cam_pos_world = cam_pos_world, cam_pos_world_prev
                    cam_pos_world_prev_inited = True
                pass # frame poses

                # draw head camera at the latest frame position
                # find the latest frame
                cam_ind = len(camR_list) - 1
                while cam_ind >= 0 and camR_list[cam_ind] is None:
                    cam_ind -= 1

                # current <- the latest frame position
                if cam_ind >= 0:
                    camR = camR_list[cam_ind]
                    camT = camT_list[cam_ind]
                    assert not camR_list[cam_ind] is None

                    self.DrawPhysicalCamera(camR, camT, cam_to_world)
                pass # process head camera
            pass # two H decompositions
        pass # lock

    def setup(self, width, height, debug=0):
        glutInit()

        """ Setup window and pygame environment. """
        pygame.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption('OpenGL AR demo')
        #pygame.key.set_repeat(10, 10)  # allow multiple keyDown events per single key press
        pygame.key.set_repeat(5, 1)  # allow multiple keyDown events per single key press

        gap = 0
        glViewport(gap, gap, width - 2 * gap, height - 2 * gap)

        glPointSize(3)

        # glEnable(GL_LIGHTING)
        #glDisable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        #glCullFace(GL_BACK)
        glCullFace(GL_FRONT_AND_BACK)
        #glEnable(GL_COLOR_MATERIAL)
        #glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        #glEnable(GL_NORMALIZE)
        glClearColor(0, 1, 1, 0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        r = 2
        glOrtho(-r, r, -r, r, 0, 100)
        self.orthoRadius = r

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.scene_scale = 1.0 / 2
        s = self.scene_scale
        glScale(s, s, s)
        # gluLookAt(1, 1, 5, 0, 0, 0, 0, 1, 0)
        # gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

        if len(self.xs3d) > 0:
            cent = np.average(self.xs3d, axis=0)
        else:
            cent = np.array([0,0,0])
        if debug >= 3: print("model average={0}".format(cent))
        self.eye = np.array([0, 0, -2], dtype=np.float32)
        self.center = cent
        self.up = np.array([0, -1, 0], dtype=np.float32)
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], self.center[0], self.center[1], self.center[2], self.up[0],
                  self.up[1], self.up[2])

    def drawData(self, width, height):
        draw_skin = False
        if draw_skin:
            imgAdorn = self.imgLeft
            #imgAdorn = self.imgRight
            origHeight = imgAdorn.shape[0]
            origWidth = imgAdorn.shape[1]

            imgLeftRgb = cv2.cvtColor(imgAdorn, cv2.COLOR_BGR2RGB)

            # texture must have size of power 2 (eg 128, 256, 512...)
            # image is padded with zeros up to power 2 size
            texHeight = int(2 ** math.ceil(math.log(origHeight, 2)))
            texWidth = int(2 ** math.ceil(math.log(origWidth, 2)))
            fracX = origWidth  / float(texWidth)
            fracY = origHeight / float(texHeight)

            texImage = np.zeros((texHeight, texWidth, 3), dtype=np.uint8)
            texImage[0:origHeight, 0:origWidth, :] = imgLeftRgb
            texImage = np.flipud(texImage)  # NOTE: _PyOpenGL require y-flipped image


            # bind the texture
            glEnable(GL_TEXTURE_2D)
            texId = int(glGenTextures(1))  # NOTE: need to cast to int (with uint it doesn't work)
            glBindTexture(GL_TEXTURE_2D, texId)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, texImage)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            glColor4f(1.0, 1.0, 1.0, 1.0)

            #
            for i in range(0, self.simplices.shape[0]):
                simpl = self.simplices[i, :]

                pt1 = self.xs1[simpl[0]]
                pt2 = self.xs1[simpl[1]]
                pt3 = self.xs1[simpl[2]]
                # pt1 = self.xs2[simpl[0]]
                # pt2 = self.xs2[simpl[1]]
                # pt3 = self.xs2[simpl[2]]

                w1 = self.xs3d[simpl[0]]
                w2 = self.xs3d[simpl[1]]
                w3 = self.xs3d[simpl[2]]

                # NOTE: (1- fraction) is because texture coordinates go bottom-top
                # NOTE: fracXY is multiplied to convert from padded to original texture coordinates

                glBegin(GL_TRIANGLES)
                # glNormal3f(1, 1, 1)
                t1x, t1y = fracX * (pt1[0] / float(origWidth)), 1 - fracY * (pt1[1] / float(origHeight))
                glTexCoord2f(t1x, t1y)
                glVertex3f(w1[0], w1[1], w1[2])

                # glNormal3f(1, 1, 1)
                t1x, t1y = fracX * (pt2[0] / float(origWidth)), 1 - fracY * (pt2[1] / float(origHeight))
                glTexCoord2f(t1x, t1y)
                glVertex3f(w2[0], w2[1], w2[2])

                # glNormal3f(1, 1, 1)
                t1x, t1y = fracX * (pt3[0] / float(origWidth)), 1 - fracY * (pt3[1] / float(origHeight))
                glTexCoord2f(t1x, t1y)
                glVertex3f(w3[0], w3[1], w3[2])
                glEnd()

            # clear the texture
            glDeleteTextures(texId)

            #glDisable(GL_TEXTURE_2D)

        draw_camers = True
        if draw_camers:
            self.DrawCameras()

        draw_world_coords = True
        if draw_world_coords:
            with self.cameras_lock:
                glBegin(GL_POINTS)
                for pt in self.xs3d:
                    glVertex3f(pt[0],pt[1],pt[2])
                glEnd()

    def handlePyGameEvent(self, event):
        moved = False
        if event.type == pygame.KEYDOWN:
            # keysPressed = pygame.key.get_pressed()

            sc = 0.01
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                sc = sc*20

            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                dir = self.center - self.eye
                upNorm = self.up / LA.norm(self.up)
                dirEyeNormal = np.cross(dir, upNorm)

                if event.key == pygame.K_UP:
                    self.eye += upNorm * sc
                elif event.key == pygame.K_DOWN:
                    self.eye -= upNorm * sc

                newUp = np.cross(dirEyeNormal, dir)
                newUp = newUp / LA.norm(newUp)
                self.up = newUp
                moved = True

            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                dir = self.center - self.eye
                right = np.cross(dir, self.up)
                right = right / LA.norm(right)

                if event.key == pygame.K_LEFT:
                    self.eye -= right * sc
                elif event.key == pygame.K_RIGHT:
                    self.eye += right * sc
                moved = True

            if event.key == pygame.K_KP_PLUS or event.key == pygame.K_KP_MINUS:
                # dir = self.center - self.eye
                # dirNorm = dir / LA.norm(dir)
                #
                # if event.key == pygame.K_KP_PLUS: # zoom in
                #     self.eye += dirNorm
                # elif event.key == pygame.K_KP_MINUS:
                #     self.eye -= dirNorm

                r = self.orthoRadius
                if event.key == pygame.K_KP_PLUS:  # zoom in
                    r *= 0.95
                elif event.key == pygame.K_KP_MINUS:
                    r *= 1.05
                self.orthoRadius = r

                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                glOrtho(-r, r, -r, r, -99999, 99999)
                glMatrixMode(GL_MODELVIEW)
                moved = True

            if moved:
                glLoadIdentity()

                s = self.scene_scale
                glScale(s, s, s)

                gluLookAt(self.eye[0], self.eye[1], self.eye[2], self.center[0], self.center[1], self.center[2],
                          self.up[0], self.up[1], self.up[2])
                #print("eye={0} up={1} center={2} orthoRadius={3}".format(self.eye, self.up, self.center, self.orthoRadius))

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pass
        return moved

    # Dummy generation thread
    def WorkerGenerateDummyCameraPositions(self, sleep_time_sec, continue_lock, cameras_lock, main_args):
        debug = main_args.debug
        yaw = 0
        stroke = 0
        for it in range(0,10000):
            # check if cancel is requested
            with continue_lock:
                cont = self.do_computation_flag
            if not cont:
                if debug >= 3: print("got computation cancel request")
                break

            time.sleep(sleep_time_sec)
            print("it:{0} Generating [R,T]".format(it))

            # r1 = [random.uniform(0, 1) for _ in range(0,3)]
            # r1 = r1 / LA.norm(r1)
            # r2 = [random.uniform(0, 1) for _ in range(0,3)]
            # r2 = r2 / LA.norm(r2)
            # r3 = skewSymmeticMat(r1).dot(r2)
            # cam_R = np.vstack((r1, r2, r3)).T
            # cam_T = np.array([random.uniform(0, 1) for _ in range(0, 3)])

            Toff = FillRT4x4(np.eye(3,3), [0,0,2])
            rot = FillRT4x4(rotMat([1,0,0], yaw), [stroke,0,0]) # TODO: zero translation!!!!!!!!!!!!!!
            cam_pos4x4 = Toff.dot(rot).dot(LA.inv(Toff))

            cam_R = cam_pos4x4[0:3, 0:3]
            cam_T = cam_pos4x4[0:3, 3]

            with cameras_lock:
                self.world_to_cam_R.append(cam_R)
                self.world_to_cam_T.append(cam_T)

            yaw += math.radians(10)
            stroke += 0.01

    def WorkerRunCornerMatcherAndFindCameraPos(self, continue_lock, cameras_lock, main_args):
        debug = main_args.debug
        cam_mat_pixel_from_meter = np.array([
            [5.7231451642124046e+02, 0., 3.2393613004134221e+02],
            [0., 5.7231451642124046e+02, 2.8464798761067397e+02],
            [0., 0., 1.]])

        #img_dir_path = "/home/mmore/Pictures/roshensweets_201704101700/is"
        img_dir_path = "/home/mmore/Pictures/blue_tapis4_640x480_n45deg/is_always_shift"
        img_abs_pathes = [os.path.join(img_dir_path, file_name) for file_name in sorted(os.listdir(img_dir_path))]
        # image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_57.png")
        # image1 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_6.png")
        # image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_13.png")

        track_sys = PointsWorld()
        track_sys.visual_host = self
        track_sys.SetCamMat(cam_mat_pixel_from_meter)

        for img_ind in range(0, len(img_abs_pathes)):
            # check if cancel is requested
            with continue_lock:
                cont = self.do_computation_flag
            if not cont:
                if debug >= 3: print("got computation cancel request")
                break

            img2_path = img_abs_pathes[img_ind]
            img2_bgr = cv2.imread(img2_path)
            assert not img2_bgr is None, "can't read image {}".format(img2_path)

            track_sys.ProcessNextImage(img2_bgr, debug=debug)
        #
        track_sys.PrintStat()

        print("num_tracked_points={0}".format(track_sys.num_tracked_points_per_frame))
        with cameras_lock:
            print("cam_poses_R={0}".format(self.world_to_cam_R))
            print("cam_poses_T={0}".format(self.world_to_cam_T))

        print("Exiting WorkerRunCornerMatcherAndFindCameraPos")

    def WorkerWalkThroughVirtualWorld(self, continue_lock, cameras_lock, main_args):
        debug = main_args.debug
        xs3D, cell_width, side_mask = GenerateTwoOrthoChess(debug=debug)
        xs3Dold = xs3D.copy()
        #xs3D = xs3D[side_mask >= 2]

        provide_ground_truth = True
        ground_truth_R_per_frame = []
        ground_truth_T_per_frame = []

        # returns [R,T], such that X2=[R,T]*X1
        def GroundTruthRelativeMotion(img_ind1, img_ind2):
            # ri from world
            r1 = ground_truth_R_per_frame[img_ind1]
            t1 = ground_truth_T_per_frame[img_ind1]
            r2 = ground_truth_R_per_frame[img_ind2]
            t2 = ground_truth_T_per_frame[img_ind2]

            r2_from1 = r2.dot(r1.T)
            t2_from1 = -r2_from1.dot(t1)+t2
            return (r2_from1,t2_from1)

        track_sys = PointsWorld()
        track_sys.visual_host = self
        if provide_ground_truth:
            track_sys.ground_truth_relative_motion = GroundTruthRelativeMotion

        pnt_ids = None
        R2 = None
        T2 = None
        img_ind = 0
        cell_width = None
        targ_ang_deg_list = np.arange(30, 360+180, 1)
        #targ_ang_deg_list = np.arange(-10, 360+180, 0.01)
        #targ_ang_deg_list = np.arange(303, 360+180, 1)
        for targ_ang_deg in targ_ang_deg_list:
            # check if cancel is requested
            with continue_lock:
                cont = self.do_computation_flag
            if not cont:
                if debug >= 3: print("got computation cancel request")
                break

            targ_ang = math.radians(targ_ang_deg)

            # cam3
            R3 = rotMat([0, 1, 0], targ_ang).T
            T3 = np.array([0, 0, 0.4])

            if provide_ground_truth:
                ground_truth_R_per_frame.append(R3)
                ground_truth_T_per_frame.append(T3)

            xs3D_cam3 = np.dot(R3, xs3D.T).T + T3

            corrupt_with_noise = False
            if corrupt_with_noise:
                np.random.seed(124)
                if cell_width is None:
                    cell_width = LA.norm(xs3D_cam3[0] - xs3D_cam3[1])
                    print("2Dcell_width={0}".format(cell_width))

                noise_perc = 0.02
                proj_err_pix = noise_perc * cell_width # 'radius' of an error
                print("proj_err_pix={0}".format(proj_err_pix))
                n3 = np.random.rand(len(xs3D), 3) * 2*proj_err_pix - proj_err_pix
                xs3D_cam3 += n3

            # perform general projection 3D->2D
            xs_img3 = xs3D_cam3.copy()
            for i in range(0, len(xs_img3)):
                xs_img3[i, :] /= xs_img3[i, -1]

            if debug >= 3: print("xs2D=\{0}".format(np.hstack((xs3D_cam3, xs_img3))))

            # expected transformation
            if debug >= 3 and not R2 is None:
                R23 = np.dot(R3, R2.T)
                T23 = -np.dot(np.dot(R3, R2.T), T2) + T3
                n23, ang23 = logSO3(R23)
                print("exact R23: n={0} ang={1}deg T23={2}\n{3}".format(n23, math.degrees(ang23), T23, R23))
                print("expect unity T:{}".format(T23 / LA.norm(T23)))
                ess_mat = skewSymmeticMat(T23 / LA.norm(T23)).dot(R23)
                ess_mat = ess_mat / LA.norm(ess_mat)
                print("expect E (for img_ind={}):\n{}".format(img_ind, ess_mat))
                print("detE==0, actually {}".format(LA.det(ess_mat)))
                c1 = LA.norm(2 * ess_mat.dot(ess_mat.T).dot(ess_mat) - np.trace(ess_mat.dot(ess_mat.T)) * ess_mat)
                print("2E*Et*E-trace(E*Et)E==0, actually {}".format(c1))

            track_sys.StartNewFrame()
            if R2 is None:
                pnt_ids = track_sys.PutNewPoints2D(xs_img3, None) # initial points
            else:
                track_sys.PutMatchedPoints2D(pnt_ids, xs_img3, None)
            track_sys.Process(debug=debug)

            R2 = R3
            T2 = T3
            img_ind += 1
        if debug >= 3: print("exiting worker thread")



    def WorkerWalkThroughVirtualWorldCrystalGrid(self, continue_lock, cameras_lock, main_args):
        debug = main_args.debug
        slam_impl = main_args.slam
        cam_mat_pixel_from_meter = None

        el_type = np.float32
        if main_args.float == "f32":
            el_type = np.float32
        elif main_args.float == "f64":
            el_type = np.float64
        elif main_args.float == "f128":
            el_type = np.longfloat
        print("config float={}".format(el_type))

        provide_ground_truth = True
        img_width, img_height = 640, 480  # target image to project 3D points to

        test_data_gen = suriko.test_data_builder.CrystallGridDataSet(el_type, img_width, img_height, provide_ground_truth=provide_ground_truth)

        track_sys = PointsWorld()
        track_sys.elem_type = el_type
        track_sys.use_mpmath = main_args.mpmath
        track_sys.visual_host = self
        if provide_ground_truth:
            track_sys.ground_truth_relative_motion = lambda img_ind1, img_ind2: test_data_gen.GroundTruthRelativeMotion(img_ind1, img_ind2)
            track_sys.ground_truth_map_pnt_pos = lambda img_ind1, map_point_id: test_data_gen.GroundTruthMapPointPos(img_ind1, map_point_id)
        min_num_3Dpoints = 9120
        track_sys.min_num_3Dpoints = min_num_3Dpoints
        track_sys.slam_impl = slam_impl

        test_data_gen.CamMatChanged(lambda cam_mat_pixel_from_meter: track_sys.SetCamMat(cam_mat_pixel_from_meter))

        virt_feat_tracker = VirtualImageFeatureTracker(min_num_3Dpoints)
        track_sys.img_feat_tracker = virt_feat_tracker

        img_gray = np.zeros((img_height, img_width), np.uint8)
        img_bgr = np.zeros((img_height, img_width, 3), np.uint8)
        img_bgr_prev = np.zeros((img_height, img_width, 3), np.uint8)
        xs_objs_clipped_prev = None

        img_ind = 0
        R2,T2 = None,None
        np.random.seed(124)

        obs_data = test_data_gen.Generate()
        for frame_ind, (R3,T3), xs_objs_clipped in obs_data:
            # check if cancel is requested
            with continue_lock:
                cont = self.do_computation_flag
            if not cont:
                if debug >= 3: print("got computation cancel request")
                break

            virt_feat_tracker.SetImage2DPoints(img_ind, xs_objs_clipped)

            img_gray.fill(0)
            for virt_id, (xpix,ypix) in xs_objs_clipped:
                img_gray[int(ypix),int(xpix)]=255
            img_bgr[:,:,0] = img_gray
            img_bgr[:,:,1] = img_gray
            img_bgr[:,:,2] = img_gray

            cv2.imshow("frontal_camera", img_bgr)
            cv2.waitKey(1)
            if False and not xs_objs_clipped_prev is None:
                pnt_dic = {} # map pntid->(pntid,x1,x2)
                for virt_id, (xpix, ypix) in xs_objs_clipped_prev:
                    pnt_dic[virt_id] = [virt_id, (xpix,ypix), None]
                for virt_id, (xpix, ypix) in xs_objs_clipped:
                    pnt_match = pnt_dic.get(virt_id, [virt_id, None, None])
                    pnt_match[2] = (xpix,ypix)
                    pnt_dic[virt_id] = pnt_match
                pts1 = []
                pts2 = []
                print("pnt_dic=\n{}".format(pnt_dic))
                for item in pnt_dic.values():
                    pts1.append(item[1])
                    pts2.append(item[2])
                ShowMatches("camera_diff", img_bgr_prev, img_bgr, pts1, pts2, transit=1, print_transit=True)

            # expected transformation
            if debug >= 3 and not R2 is None:
                R23 = np.dot(R3, R2.T)
                T23 = -np.dot(np.dot(R3, R2.T), T2) + T3
                n23, ang23 = logSO3(R23)
                print("exact R23: n={0} ang={1}deg T23={2}\n{3}".format(n23, math.degrees(ang23), T23, R23))
                print("expect unity T:{}".format(T23 / LA.norm(T23)))
                ess_mat = skewSymmeticMat(T23 / LA.norm(T23)).dot(R23)
                ess_mat = ess_mat / LA.norm(ess_mat)
                print("expect E (for img_ind={}):\n{}".format(img_ind, ess_mat))
                print("detE==0, actually {}".format(scipy.linalg.det(ess_mat)))
                c1 = LA.norm(2 * ess_mat.dot(ess_mat.T).dot(ess_mat) - np.trace(ess_mat.dot(ess_mat.T)) * ess_mat)
                print("2E*Et*E-trace(E*Et)E==0, actually {}".format(c1))

            track_sys.ProcessNextImage(img_ind, img_bgr, debug=debug)

            R2 = R3
            T2 = T3
            img_bgr_prev, img_bgr = img_bgr, img_bgr_prev
            xs_objs_clipped_prev = xs_objs_clipped
            img_ind += 1

            wait_key_press_on_next_frame = False
            if wait_key_press_on_next_frame:
                cv2.waitKey(0)
        if debug >= 3: print("exiting worker thread")

    def WorkerRunPlanarHomographyReconstruction(self, continue_lock, cameras_lock, main_args):
        debug = main_args.debug
        cam_mat = np.array([
            [5.7231451642124046e+02, 0., 3.2393613004134221e+02],
            [0., 5.7231451642124046e+02, 2.8464798761067397e+02],
            [0., 0., 1.]])

        kpd = cv2.xfeatures2d.SIFT_create(nfeatures=50)  # OpenCV-contrib-3

        # img_path_str = "/home/mmore/Pictures/blue_tapis1_640x480/is_always_shift/scene{0:05}.png"
        # img_path_str = "/home/mmore/Pictures/blue_tapis2_640x480/is_always_shift/scene{0:05}.png"
        img_path_str = "/home/mmore/Pictures/blue_tapis4_640x480_n45deg/is_always_shift/image-{0:04}.png"
        image1 = cv2.imread(img_path_str.format(1))  # tapis1=122, tapis2=1
        assert not image1 is None
        img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        kp1 = kpd.detect(img1_gray, None)

        # init
        points = np.array([kp.pt for kp in kp1], np.float32)
        points_life = [None] * len(points)
        for i, x1 in enumerate(points):
            points_list = [x1]
            points_life[i] = points_list

        num_tracked_points = []
        cam_poses_R = [np.eye(3, 3)]  # world
        cam_poses_T = [np.zeros(3)]  # world
        last_reg_frame_ind = 0
        cam2_from_world_pair = [np.eye(4)] * 2
        homog_plane_pair_prev = [None] * 2

        # gather points to search for a partner
        head_pixels = []
        head_to_thread_inds = []
        for i, points_list in enumerate(points_life):
            pnt = points_list[last_reg_frame_ind]
            if not pnt is None:
                head_pixels.append(pnt)
                head_to_thread_inds.append(i)
        num_tracked_points.append(len(head_pixels))

        with continue_lock:
            self.processed_images_counter = 0

        # img_inds = range(122+1, 451)
        img_inds = range(2, 848 + 1)
        ind2 = 1
        for img2_id in img_inds:
            with continue_lock:
                cont = self.do_computation_flag
            if not cont:
                print("got request to cancel computation")
                break

            if debug >= 3: print("img2_id={}".format(img2_id))

            img2_path = img_path_str.format(img2_id)
            image2 = cv2.imread(img2_path)
            if image2 is None:  # skip gaps in numbering
                continue

            img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            # try to find the corner in the next image
            head_pixels_array = np.array(head_pixels)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, head_pixels_array, None,
                                                             winSize=(45, 45))

            status = status.ravel()
            xs1_pixels = np.array([p for s, p in zip(status, head_pixels) if s])
            xs2_pixels = np.array([p for s, p in zip(status, next_pts) if s])

            if debug >= 4: ShowMatches("opticalFlow", image1, image2, xs1_pixels, xs2_pixels)

            clean_files_without_shift = False
            if clean_files_without_shift:
                avg_pix_shift = LA.norm(xs1_pixels - xs2_pixels)
                print("avg_pix_shift={0}".format(avg_pix_shift))
                if avg_pix_shift < 1:
                    # print("removing {0}".format(img2_path))
                    os.remove(img2_path)
                    continue

            num_pnts = len(head_pixels)
            next_num_pnts = len(xs1_pixels)
            num_tracked_points.append(next_num_pnts)
            if debug >= 3: print("Number of active points {0}->{1}".format(num_pnts, next_num_pnts))

            # reserve space
            for points_list in points_life: points_list.append(None)

            next_to_thread_inds = [0] * len(next_pts)
            for i, s in enumerate(status):
                if s:
                    parent_ind = head_to_thread_inds[i]  # index in the point_life
                    pnt2 = next_pts[i]
                    points_life[parent_ind][ind2] = pnt2
                    next_to_thread_inds[i] = parent_ind

            # convert pixel to image coordinates (pixel -> meters)
            xs1_meters = []
            xs2_meters = []
            convertPixelToMeterPoints(cam_mat, [xs1_pixels], xs1_meters)
            convertPixelToMeterPoints(cam_mat, [xs2_pixels], xs2_meters)
            xs1_meters = xs1_meters[0]
            xs2_meters = xs2_meters[0]

            def MeasureConsensusFun(items_count, samp_group_inds, cons_set_mask):
                if debug >= 94:
                    samp_xs1_pixels = xs1_pixels[samp_group_inds]
                    samp_xs2_pixels = xs2_pixels[samp_group_inds]
                    ShowMatches("sample", image1, image2, samp_xs1_pixels, samp_xs2_pixels)

                samp_xs1_meters = xs1_meters[samp_group_inds]
                samp_xs2_meters = xs2_meters[samp_group_inds]

                H2 = findHomogDltMasks(samp_xs1_meters, samp_xs2_meters)

                homog_thr = 1.5 ** 2  # number of pixels between point x2 and one, projected by homography H*x1
                cons_set_card = 0
                for i in range(0, items_count):
                    pt1 = xs1_meters[i]
                    pt2 = xs2_meters[i]

                    err = HomogErrSqrOneWay(H2, pt1, pt2)
                    cons = err < homog_thr
                    # if debug >= 3: print("err={0} include={1}".format(err, cons))
                    if cons:
                        cons_set_card += 1
                    cons_set_mask[i] = cons

                return cons_set_card

            points_count = len(xs1_meters)
            consens_mask = np.zeros(points_count, np.uint8)
            suc_prob = 0.99
            outlier_ratio = 0.3
            cons_set_card = GetMaxSubsetInConsensus(points_count, 4, outlier_ratio, suc_prob, MeasureConsensusFun, consens_mask)
            assert cons_set_card >= 4, "need >= 4 points to compute homography"
            if debug >= 3: print("cons_set_card={0}".format(cons_set_card))

            cons_xs1_meter = np.array([p for i, p in enumerate(xs1_meters) if consens_mask[i]])
            cons_xs2_meter = np.array([p for i, p in enumerate(xs2_meters) if consens_mask[i]])

            if debug >= 4:
                cons_xs1_pixels = np.array([p for i, p in enumerate(xs1_pixels) if consens_mask[i]])
                cons_xs2_pixels = np.array([p for i, p in enumerate(xs2_pixels) if consens_mask[i]])
                ShowMatches("cons", image1, image2, cons_xs1_pixels, cons_xs2_pixels)

            # now we have - the set of points which agree on homography
            H = findHomogDltMasks(cons_xs1_meter, cons_xs2_meter)
            err2 = CalculateHomogError(cons_xs1_meter, cons_xs2_meter, HomogErrSqrOneWay)
            if debug >= 3: print("H=\n{0} err={1}".format(H, err2))

            frame_poses = ExtractRotTransFromPlanarHomography(H, cons_xs1_meter, cons_xs2_meter, debug=debug)

            if debug >= 3:
                print("frame_poses={0}".format(frame_poses))

            # contains indices of matched normal N in previous frame
            # [A,B] means: match 0th normal N in current frame to normal A in previous frame, 1st to Bth
            decomp_track_parent_ind = [0, 1]
            decomp_track_price = [0, 0]
            if not "decomp_track_price_accum" in locals():
                decomp_track_price_accum = [0, 0]
                decomp_track_flip_count = [0, 0]
                primary_decomp_track_ind = 0
                decomp_track_n_world = []

            # find angles between normals in previous and current frames
            # match the closest normals
            if not homog_plane_pair_prev[0] is None:
                cosangs = np.zeros((2, 2))
                inds = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for i1, i2 in inds:
                    N_prev = homog_plane_pair_prev[i1]
                    N_cur = frame_poses[i2][2]
                    ca = np.dot(N_prev[0:3], N_cur)
                    cosangs[i1, i2] = ca
                # min(ang)=max(cosang)
                if debug >= 3: print("cosangs={0}".format(cosangs))
                # match_inds = max(inds, key=lambda ind: cosangs[ind[0],ind[1]])

                # Way2 find best price
                prices = np.zeros(2)
                for i, (i1, i2) in enumerate(inds[0:2]):
                    ca1 = cosangs[i1, i2]
                    ca2 = cosangs[1 - i1, 1 - i2]
                    price = math.degrees(math.acos(ca1)) + math.degrees(math.acos(ca2))
                    prices[i] = price
                min_price_ind = min([0, 1], key=lambda i: prices[i])
                match_inds = inds[min_price_ind]

                i1, i2 = match_inds
                decomp_track_parent_ind[i2] = i1
                decomp_track_parent_ind[1 - i2] = 1 - i1  # invert 0 or 1 (1-0=1 and 1-1=0)
                price1 = math.degrees(math.acos(cosangs[i1, i2]))
                price2 = math.degrees(math.acos(cosangs[1 - i1, 1 - i2]))
                decomp_track_price[i2] = price1
                decomp_track_price[1 - i2] = price2
                decomp_track_price_accum[i2] += price1
                decomp_track_price_accum[1 - i2] += price2
                decomp_track_flip_count[i2] += (price1 > 90)
                decomp_track_flip_count[1 - i2] += (price2 > 90)

                if primary_decomp_track_ind == i1:
                    primary_decomp_track_ind = i2
                else:
                    primary_decomp_track_ind = 1-i2

            assert decomp_track_parent_ind[0] != decomp_track_parent_ind[1], "Both decomposition must be associated with parent"

            cam3_from_world_pair = [None, None]
            candN_world_pair = [None, None]
            for i, (candR, candT_div_d, candN) in enumerate(frame_poses):
                cam3_from2 = SE3Mat(r=candR, t=candT_div_d)
                if debug >= 3:
                    n, ang = logSO3(candR)
                    print("R n={0} ang={1}deg\n{2}".format(n, math.degrees(ang), candR))
                    print("T/d={0}".format(candT_div_d))
                    print("N_={0}".format(candN))

                match_ind = decomp_track_parent_ind[i]

                # transform N from cam2 to world
                candN_world = np.dot(cam2_from_world_pair[match_ind][0:3,0:3].T, candN)
                if debug >= 3:
                    print("Nw={0} match_ind={1} Nwp={2} price={3}(d={4} flip={5})".format(candN_world, match_ind, homog_plane_pair_prev[match_ind], decomp_track_price_accum[match_ind], decomp_track_price[match_ind],decomp_track_flip_count[match_ind]))

                cam3_from_world = cam3_from2.dot(cam2_from_world_pair[i])
                cam3_from_world_pair[i] = cam3_from_world
                homog_plane_pair_prev[match_ind] = candN_world
                candN_world_pair[i] = candN_world

            #
            decomp_track_n_world.append(candN_world_pair)

            with cameras_lock:
                mat4x4 = cam3_from_world_pair[primary_decomp_track_ind]
                self.world_to_cam_R.append(mat4x4[0:3,0:3]) # world
                self.world_to_cam_T.append(mat4x4[0:3,3])  # world
                mat4x4 = cam3_from_world_pair[1-primary_decomp_track_ind]
                self.world_to_cam_R_alt.append(mat4x4[0:3,0:3]) # world
                self.world_to_cam_T_alt.append(mat4x4[0:3,3])  # world

            with continue_lock: # notify model is changed, redraw is required
                self.world_map_changed_flag = True
                self.processed_images_counter += 1

            cam2_from_world_pair[:] = cam3_from_world_pair[:]

            img1_gray = img2_gray
            ind2 += 1
            #time.sleep(0.1)

        print("num_tracked_points={0}".format(num_tracked_points))
        print("cam_poses_R={0}".format(cam_poses_R))
        print("cam_poses_T={0}".format(cam_poses_T))

        thr = 0.1 # norm(N-candN)<thr
        ChooseSingleHomographyDecomposition(decomp_track_n_world, thr, debug)

        # [Nx6]
        dump_decomp = False
        if dump_decomp: np.savetxt("homog_two.txt", np.array([np.hstack((p[0], p[1])) for p in decomp_track_n_world]))

        return cam_poses_R, cam_poses_T

    def mainRun(self, main_args):
        debug = main_args.debug
        #self.run()

        width, height = 1024, 768
        self.setup(width, height, debug=debug)

        job_id = main_args.job or 1
        if job_id == 1:
            compute_thread = threading.Thread(name="comput", target=self.WorkerGenerateDummyCameraPositions,\
                args=(0.033, self.continue_computation_lock, self.cameras_lock, main_args))
        elif job_id == 20:
            compute_thread = threading.Thread(name="comput", target=self.WorkerWalkThroughVirtualWorld,\
                args=(self.continue_computation_lock, self.cameras_lock, main_args))
        elif job_id == 2:
            compute_thread = threading.Thread(name="comput", target=self.WorkerRunCornerMatcherAndFindCameraPos,\
                args=(self.continue_computation_lock, self.cameras_lock, main_args))
        elif job_id == 3:
            compute_thread = threading.Thread(name="comput", target=self.WorkerRunPlanarHomographyReconstruction,\
                args=(self.continue_computation_lock, self.cameras_lock, main_args))
        elif job_id == 61:
            compute_thread = threading.Thread(name="comput", target=self.WorkerWalkThroughVirtualWorldCrystalGrid,\
                args=(self.continue_computation_lock, self.cameras_lock, main_args))
        compute_thread.start()


        require_redraw = False
        t1 = time.time()
        processed_images_counter_prev = -1
        while True:
            if pygame.event.peek():
                event = pygame.event.poll()
                #pygame.event.clear()  # clear all other events; try if got transparent window which is not redrawen
                if event.type == pygame.QUIT:
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    break
                moved = self.handlePyGameEvent(event)
                if moved:
                    require_redraw = True

            # query if model has changed
            processed_images_counter = -1
            with self.continue_computation_lock: # notify model is changed, redraw is required
                if self.world_map_changed_flag:
                    require_redraw = True
                    self.world_map_changed_flag = False # reset the flag to avoid further redraw
                processed_images_counter = self.processed_images_counter

            #require_redraw = True # forcebly, exhaustively redraw a scene
            if require_redraw:
                t2 = time.time()
                #print("redraw")
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                # self.drawBackground(width,height,None)
                self.drawData(width, height)

                # images per sec
                if t2 - t1 > 0.5: # display images per sec every X seconds
                    if processed_images_counter_prev != -1:
                        ips = (processed_images_counter - processed_images_counter_prev) / (t2 - t1)
                        title_str = "images_per_sec={0:05.2f}".format(ips)
                        pygame.display.set_caption(title_str)
                    processed_images_counter_prev = processed_images_counter
                    t1 = t2

                pygame.display.flip() # flip after ALL updates have been made
                require_redraw = False
            pass # require redraw

            # give other threads an opportunity to progress;
            # without yielding, the pygame thread consumes the majority of cpu
            time.sleep(0.03)
        pass # game loop

        # do cancel request
        with self.continue_computation_lock:
            self.do_computation_flag = False

        compute_thread.join()
        if debug >= 3: print("exiting mainRun")


def testCorrespondencePoly6():
    #a = 2; b = 3; c = 3; d = 4; f1 = 1; f2 = 1; # tmin=-0.019 MVGCV Fig 12.4a
    a = 2; b = -1; c = 1; d = 0; f1 = 1; f2 = 1; # tmin=0 MVGCV Fig 12.4b
    x1Perp, x2Perp = correctPointCorrespondencePoly6_Transformed(a, b, c, d, f1, f2)

def TestHomography():
    debug = 3
    xs3D_orig, cell_width, side_mask = GenerateTwoOrthoChess()

    # select points on one side
    side = 2 # 1 or 2
    xs3D = [p for i,p in enumerate(xs3D_orig) if side_mask[i] == side or side_mask[i] == 3] # 3=1or2
    xs3D = np.array(xs3D)

    R2 = None
    #angs = [30]
    #angs = np.arange(0,45,5)
    angs = np.arange(45,85,15) # Movement3
    world_to_cam_prev = None
    for ang_ind, targ_ang_deg in enumerate(angs):
        targ_ang = math.radians(targ_ang_deg)

        d=0.4 # distance to plane
        # cam3
        # Movement1: rotate around OY by 45deg
        # R3 = rotMat([0,1,0], math.radians(15)).T # 15
        # Movement2: move camera parallel to plane with distance d
        # T3 = np.array([-0.4-0.1*ang_ind, 0, 0.4])
        # R3 = rotMat([0, 1, 0], math.radians(targ_ang_deg))
        # world_to_cam = np.dot(SE3Mat(r=R3), SE3Mat(t=T3))
        # Movement3: rotate and keep by same distance from plane
        R3 = rotMat([0, 1, 0], math.radians(targ_ang_deg))
        T3 = np.array([0, 0, d/math.cos(targ_ang)])
        world_to_cam = np.dot(SE3Mat(t=T3),SE3Mat(r=R3))

        xs3D_cam3 = np.array([np.dot(world_to_cam, (p[0], p[1], p[2], 1))[0:3] for p in xs3D])
        #print("xs3D_cam3={0}".format(xs3D_cam3))

        if not world_to_cam_prev is None:
            cam32 = world_to_cam.dot(LA.inv(world_to_cam_prev))
            print("cam32=\n{0}".format(cam32))

            actual_transl = cam32[0:3,-1]
            print("actual_transl={0}".format(actual_transl))
            print("expect T/d={0}".format(actual_transl/d))
            # points are in XOY plane, hence plane's norm=OZ
            norm = np.dot(world_to_cam_prev, [0, 0, 1, 0])
            print("expect N={0}".format(norm))


        if not R2 is None:
            # perform general projection 3D->2D
            xs_img2 = xs3D_cam2.copy()
            for i in range(0, len(xs_img2)):
                xs_img2[i, :] /= xs_img2[i, -1]
            xs_img3 = xs3D_cam3.copy()
            for i in range(0, len(xs_img3)):
                xs_img3[i, :] /= xs_img3[i, -1]

            H1 = findHomogDlt(xs_img2[:,0:2], xs_img3[:,0:2])
            err1 = CalculateHomogError(H1, xs_img2, xs_img3, HomogErrSqrOneWay)
            #print("H=\n{0} err={1}".format(H1, err1))

            H2 = findHomogDltMasks(xs_img2, xs_img3)
            H2 = H2 / H2[-1,-1]
            err2 = CalculateHomogError(H2, xs_img2, xs_img3, HomogErrSqrOneWay)
            #print("H=\n{0} err={1}".format(H2, err2))

            E = FindEssentialMat5PointStewenius(xs_img2, xs_img3, True, debug)

            frame_poses = ExtractRotTransFromPlanarHomography(H2, xs_img2, xs_img3, debug = debug)

            for cands_R, cands_T_div_d, cands_N in frame_poses:
                pass

        R2,T2 = R3,T3
        xs3D_cam2 = xs3D_cam3
        world_to_cam_prev = world_to_cam

    print("")

def TestEssMat(main_args):
    debug = main_args.debug
    xs3D, cell_width, side_mask = GenerateTwoOrthoChess()

    # choose 5 points
    #xs3D = np.array([xs3D[0],xs3D[3],xs3D[11],xs3D[13],xs3D[19],xs3D[22]]) # 6 points on two planes
    xs3D = np.array([xs3D[0],xs3D[3],xs3D[11],xs3D[13],xs3D[22]]) # 5 points on two planes
    #xs3D = np.array([xs3D[3],xs3D[7],xs3D[17],xs3D[20],xs3D[22]]) # 5 points on the same plane
    #xs3D = np.array([p for i,p in enumerate(xs3D) if side_mask[i] >= 2]) # >5 points on the same plane

    targ_ang_deg = 30
    targ_ang = math.radians(targ_ang_deg)

    # rotate around OY by 45deg
    # NOTE: R1 rotates the points => to rotate the frame we take the transpose (inverse)
    R2 = rotMat([0, 1, 0], math.radians(45)).T  # 45
    T2 = np.array([0, 0, 0.25])
    xs3D_cam2 = np.dot(R2, xs3D.T).T + T2
    print("xs3D_cam2={0}".format(xs3D_cam2))

    # cam3
    # R3 = rotMat([0,1,0], math.radians(15)).T # 15
    R3 = rotMat([0, 1, 0], math.radians(45 - targ_ang_deg)).T
    T3 = np.array([0, 0, 0.4])
    xs3D_cam3 = np.dot(R3, xs3D.T).T + T3
    print("xs3D_cam3={0}".format(xs3D_cam3))

    corrupt_with_noise = False
    if corrupt_with_noise:
        noise_perc = 0.01
        np.random.seed(124)

        cell_width = LA.norm(xs3D_cam2[0] - xs3D_cam2[1])
        print("2Dcell_width={0}".format(cell_width))

        proj_err_pix = noise_perc * cell_width  # 'radius' of an error
        print("proj_err_pix={0}".format(proj_err_pix))
        n2 = np.random.rand(len(xs3D), 3) * 2 * proj_err_pix - proj_err_pix
        n3 = np.random.rand(len(xs3D), 3) * 2 * proj_err_pix - proj_err_pix
        xs3D_cam2 += n2
        xs3D_cam3 += n3

    # perform general projection 3D->2D
    xs_img2 = xs3D_cam2.copy()
    for i in range(0, len(xs_img2)):
        xs_img2[i, :] /= xs_img2[i, -1]
    xs_img3 = xs3D_cam3.copy()
    for i in range(0, len(xs_img3)):
        xs_img3[i, :] /= xs_img3[i, -1]

    print(np.hstack((xs3D_cam2, xs_img2, xs3D_cam3, xs_img3)))

    # expected transformation
    R23 = np.dot(R3, R2.T)
    T23 = -np.dot(np.dot(R3, R2.T), T2) + T3
    n23, ang23 = logSO3(R23)
    print("exact R23: n={0} ang={1}deg T23={2}\n{3}".format(n23, math.degrees(ang23), T23, R23))
    print("expect unity T:{}".format(T23/LA.norm(T23)))
    print("expect E:\n{}".format(skewSymmeticMat(T23/LA.norm(T23)).dot(R23)))

    calc_R0 = True # calculate R0 from MASKS Theorem 5.5 page 114
    if calc_R0:
        tvec= T23
        tlen = LA.norm(tvec)
        tlen2 = tlen**2
        t1,t2,t3 = tvec[:]
        b1 = np.array([t2**2+t3**2, -t1*t2, -t1*t3]) / math.sqrt((t2**2+t3**2)*tlen2)
        b2 = np.array([0, t3, -t2]) / math.sqrt(t2**2+t3**2)
        b3 = np.array([t1, t2, t3]) / tlen
        B = np.vstack((b1, b2, b3)).T
        r0_fromT =  LA.inv(B)
        print("R0_fromT=\n{}".format(r0_fromT))


    # recover essential matrix
    plinear_sys_err = [0.0]
    perr_msg = [""]
    suc, ess_mat8 = FindEssentialMat8Point(xs_img2, xs_img3, unity_translation=True, plinear_sys_err=plinear_sys_err, debug=debug, perr_msg=perr_msg)
    if suc:
        print("8point ess_mat=\n{0} linear_sys_err={1}".format(ess_mat8, plinear_sys_err[0]))
    else:
        print("8point failed: {0}".format(perr_msg))

    suc, ess_mat_list, real_roots = FindEssentialMat7Point(xs_img2, xs_img3, unity_translation=True, debug=debug, perr_msg=perr_msg)
    if suc:
        print("7point ess_mat_list=\n{0}".format(ess_mat_list))
    else:
        print("7point failed: {0}".format(perr_msg))

    suc,ess_mat_list=FindEssentialMat5PointStewenius(xs_img2, xs_img3, True, True, debug, expected_ess_mat=ess_mat8, perr_msg=perr_msg)
    if suc:
        print("5point ess_mat_list=\n{}".format(ess_mat_list))
    else:
        print("5point failed: {0}".format(perr_msg))

    #em_cands = [ess_mat8]
    #em_cands = [skewSymmeticMat(-T23/LA.norm(T23)).dot(-R23)]

    rz90p = rotMat([0, 0, 1],  math.pi / 2)
    rz90m = rotMat([0, 0, 1], -math.pi / 2)
    u = np.dot(r0_fromT.T, rz90m)
    sig = np.array([1,1,0])
    vt = np.dot(r0_fromT, R23)
    #svd_ess_mat = (sig, u, vt)
    svd_ess_mat = None
    #em_cands = [u.dot(np.diag(sig)).dot(vt)]

    em_cands = ess_mat_list
    for ess_mat in em_cands:
        ang_err_pix1 = -1
        ang23_actual1 = -1
        if True:
            check_is_ess = IsEssentialMat(ess_mat)
            if check_is_ess:
                suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat, xs_img2, xs_img3, validate_ess_mat=True, svd_ess_mat=svd_ess_mat)
                if suc:
                    n23a, ang23a = logSO3(ess_R)
                    print("R23a: n={0} ang={1}deg T23a={2}\n{3}".format(n23a, math.degrees(ang23a), ess_Tvec, ess_R))
                    ang_err_pix1 = math.degrees(math.fabs(targ_ang - ang23a))
                    ang23_actual1 = math.degrees(ang23a)

        #
        ang_err_pix2 = -1
        ang23_actual2 = -1
        # TODO: why refinement may corrupt essential mat structure
        suc, ess_mat_refined = RefineFundMat(ess_mat, xs_img2, xs_img3)
        if suc:
            check_is_ess = IsEssentialMat(ess_mat_refined)
            print("check_is_ess={}".format(check_is_ess))
            check_is_ess = IsEssentialMat(ess_mat_refined)
            if check_is_ess:
                suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat_refined, xs_img2, xs_img3)
                if suc:
                    n23a, ang23a = logSO3(ess_R)
                    print("R23a: n={0} ang={1}deg T23a={2}\n{3}".format(n23a, math.degrees(ang23a), ess_Tvec, ess_R))
                    ang_err_pix2 = math.degrees(math.fabs(targ_ang - ang23a))
                    ang23_actual2 = math.degrees(ang23a)
        print()
    print()

def RunGenerateVirtualPointsProjectAndReconstruct():
    debug = 3
    xs3D, cell_width = GenerateTwoOrthoChess()

    #for targ_ang_deg in np.arange(30, 0, -1):
    #for targ_ang_deg in np.linspace(3,0,50):
    for targ_ang_deg in [30]:
        print("targ_ang={0}deg".format(targ_ang_deg))
        targ_ang = math.radians(targ_ang_deg)

        # rotate around OY by 45deg
        # NOTE: R1 rotates the points => to rotate the frame we take the transpose (inverse)
        R2 = rotMat([0,1,0], math.radians(45)).T # 45
        T2 = np.array([0,0,0.25])
        xs3D_cam2 = np.dot(R2, xs3D.T).T+T2
        print("xs3D_cam2={0}".format(xs3D_cam2))

        # cam3
        #R3 = rotMat([0,1,0], math.radians(15)).T # 15
        R3 = rotMat([0,1,0], math.radians(45-targ_ang_deg)).T
        T3 = np.array([0,0,0.4])
        xs3D_cam3 = np.dot(R3, xs3D.T).T+T3
        print("xs3D_cam3={0}".format(xs3D_cam3))

        cell_width=LA.norm(xs3D_cam2[0]-xs3D_cam2[1])
        print("2Dcell_width={0}".format(cell_width))

        noise_to_pix_error = []
        real_root_to_ang = []
        # add noise to the projected values
        # noise = uniform(0, portion of cell_width)
        for noise_perc in np.arange(0,1, 0.1):
        #for noise_perc in np.linspace(0,3, 200):
        #for noise_perc in [0.0457627118644]:
        #for noise_perc in [1]:

            corrupt_with_noise = True
            if corrupt_with_noise:
                np.random.seed(124)
                proj_err_pix = noise_perc * cell_width # 'radius' of an error
                print("proj_err_pix={0}".format(proj_err_pix))
                n2 = np.random.rand(len(xs3D), 3) * 2*proj_err_pix - proj_err_pix
                n3 = np.random.rand(len(xs3D), 3) * 2*proj_err_pix - proj_err_pix
                xs3D_cam2 += n2
                xs3D_cam3 += n3


            # perform general projection 3D->2D
            xs_img2 = xs3D_cam2.copy()
            for i in range(0, len(xs_img2)):
                xs_img2[i,:] /= xs_img2[i,-1]
            xs_img3 = xs3D_cam3.copy()
            for i in range(0, len(xs_img3)):
                xs_img3[i,:] /= xs_img3[i,-1]

            print(np.hstack((xs3D_cam2, xs_img2, xs3D_cam3, xs_img3)))

            # expected transformation
            R23 = np.dot(R3, R2.T)
            T23 = -np.dot(np.dot(R3, R2.T), T2) + T3
            n23, ang23 = logSO3(R23)
            print("R23: n={0} ang={1}deg T23={2}\n{3}".format(n23, math.degrees(ang23), T23, R23))

            # recover essential matrix
            perror = [0.0]
            suc, ess_mat = FindEssentialMat8Point(xs_img2, xs_img3, unity_translation=True, plinear_sys_err=perror, debug = debug)
            assert suc
            print("ess_mat=\n{0} err={1}".format(ess_mat, perror[0]))

            suc, ess_mat_list, real_roots = FindEssentialMat7Point(xs_img2, xs_img3, unity_translation=True, debug = debug)
            assert suc
            print("ess_mat_list=\n{0} err={1}".format(ess_mat_list, perror[0]))

            ess_mat_to_ang = []
            samps = SampsonDistanceCalc()
            ess_mat_cands = [ess_mat] + ess_mat_list
            for ess_mat_ind, ess_mat in enumerate(ess_mat_cands):
                calc_epi = True
                if calc_epi:
                    (dVec, u, vt) = cv2.SVDecomp(ess_mat)
                    epi1 = vt.T[:, -1]
                    epi1 = epi1 / epi1[-1]
                    print("epipole1={0}".format(epi1))

                    (dVec, u, vt) = cv2.SVDecomp(ess_mat.T)
                    epi2 = vt.T[:, -1]
                    epi2 = epi2 / epi2[-1]
                    print("epipole2={0}".format(epi2))

                samps_dist = samps.DistanceMult(ess_mat, xs_img2, xs_img3)
                print("sampson distance={0}".format(samps_dist))

                #
                ang_err_pix1 = -1
                ang23_actual1 = -1
                suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat, xs_img2, xs_img3)
                if suc:
                    n23a, ang23a = logSO3(ess_R)
                    print("R23a: n={0} ang={1}deg T23a={2}\n{3}".format(n23a, math.degrees(ang23a), ess_Tvec, ess_R))
                    ang_err_pix1 = math.degrees(math.fabs(targ_ang - ang23a))
                    ang23_actual1 = math.degrees(ang23a)

                #
                ang_err_pix2 = -1
                ang23_actual2 = -1
                suc, ess_mat_refined = RefineFundMat(ess_mat, xs_img2, xs_img3)
                if suc:
                    suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat_refined, xs_img2, xs_img3)
                    if suc:
                        n23a, ang23a = logSO3(ess_R)
                        print("R23a: n={0} ang={1}deg T23a={2}\n{3}".format(n23a, math.degrees(ang23a), ess_Tvec, ess_R))
                        ang_err_pix2 = math.degrees(math.fabs(targ_ang - ang23a))
                        ang23_actual2 = math.degrees(ang23a)

                #noise_to_pix_error.append((noise_perc, ang_err_pix1, ang_err_pix2, samps_dist))
                #print("rot angle err: {0}deg".format(ang_err_pix2))
                #assert np.isclose(targ_ang, ang23a, rtol=1.e-2), "expect: {0}deg, actual: {1}deg".format(math.degrees(targ_ang), math.degrees(ang23a))
                real_root = -1
                if ess_mat_ind >= 1:
                    real_root = real_roots[ess_mat_ind-1]

                ess_mat_to_ang.append((ess_mat_ind, real_root, samps_dist, ang23_actual1, ang23_actual2))
            pass # ess_mat enumeration
            print("ess_mat_to_ang={0}".format(ess_mat_to_ang))
            print()

            # for ess_mat
            e1_ang_err_pix = ess_mat_to_ang[0][3]
            e1_ang_err_pix_refined = ess_mat_to_ang[0][4]
            e2_ang_err_pix = -1
            e2_ang_err_pix_refined = -1
            for eind in range(1,len(ess_mat_to_ang)):
                e = ess_mat_to_ang[eind]
                if e[3] != -1 and abs(e[3]-30)<10:
                    e2_ang_err_pix = e[3]
                if e[4] != -1 and abs(e[4]-30)<10:
                    e2_ang_err_pix_refined = e[4]
                real_root_to_ang.append((e[1], e2_ang_err_pix, e2_ang_err_pix_refined))
            noise_to_pix_error.append((noise_perc, e1_ang_err_pix, e1_ang_err_pix_refined, e2_ang_err_pix, e2_ang_err_pix_refined,  samps_dist))

        print("noise_to_pix_error={0}".format(noise_to_pix_error))
        pylab.plot([p[0] for p in noise_to_pix_error], [p[1] for p in noise_to_pix_error], 'g+')
        pylab.plot([p[0] for p in noise_to_pix_error], [p[2] for p in noise_to_pix_error], 'go', fillstyle='none')
        pylab.plot([p[0] for p in noise_to_pix_error], [p[3] for p in noise_to_pix_error], 'kx')
        pylab.plot([p[0] for p in noise_to_pix_error], [p[4] for p in noise_to_pix_error], 'ks', fillstyle='none')
        pylab.plot([p[0] for p in noise_to_pix_error], [p[5] for p in noise_to_pix_error], 'b.')
        pylab.xlabel("noise%")
        pylab.ylabel("pixerr")

        pylab.figure(2)
        pylab.plot([p[0] for p in real_root_to_ang], [p[1] for p in real_root_to_ang], 'g.')
        pylab.plot([p[0] for p in real_root_to_ang], [p[2] for p in real_root_to_ang], 'r.')
        pylab.show()

    print("")


def RunReconstructionSequential():
    debug = 0
    cam_mat = np.array([
        [5.7231451642124046e+02, 0., 3.2393613004134221e+02],
        [0., 5.7231451642124046e+02, 2.8464798761067397e+02],
        [0., 0., 1.]])

    kpd = cv2.xfeatures2d.SIFT_create(nfeatures=50)  # OpenCV-contrib-3

    image1 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/is/scene00001.png")
    #image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/is/scene00002.png")
    # image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_57.png")
    # image1 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_6.png")
    # image2 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/frame_13.png")
    assert not image1 is None
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    kp1 = kpd.detect(img1_gray, None)

    # init
    points = np.array([kp.pt for kp in kp1], np.float32)
    points_life = [None] * len(points)
    for i, x1 in enumerate(points):
        points_list = [x1]
        points_life[i] = points_list

    num_tracked_points = []
    cam_poses_R = [np.eye(3,3)] # world
    cam_poses_T = [np.zeros(3)] # world
    last_reg_frame_ind = 0

    # gather points to search for a partner
    head_pixels = []
    head_to_thread_inds = []
    for i,points_list in enumerate(points_life):
        pnt = points_list[last_reg_frame_ind]
        if not pnt is None:
            head_pixels.append(pnt)
            head_to_thread_inds.append(i)
    num_tracked_points.append(len(head_pixels))

    #for ind2 in [57]:
    for ind2 in range(1, 18): # 518
        img2_id = ind2 + 1
        print("img2_id={0}".format(img2_id))

        img2_path = "/home/mmore/Pictures/roshensweets_201704101700/is/scene{0:05}.png".format(img2_id)
        image2 = cv2.imread(img2_path)
        assert not image2 is None

        img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # try to find the corner in the next image
        head_pixels_array = np.array(head_pixels)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, head_pixels_array, None)

        status = status.ravel()
        xs1_pixels = [p for s, p in zip(status, head_pixels) if s]
        xs2_pixels = [p for s, p in zip(status, next_pts) if s]

        num_pnts = len(head_pixels)
        next_num_pnts = len(xs1_pixels)
        num_tracked_points.append(next_num_pnts)
        if debug >=3: print("Number of active points {0}->{1}".format(num_pnts, next_num_pnts))

        # reserve space
        for points_list in points_life: points_list.append(None)

        next_to_thread_inds = [0] * len(next_pts)
        for i,s in enumerate(status):
            if s:
                parent_ind = head_to_thread_inds[i] # index in the point_life
                pnt2 = next_pts[i]
                points_life[parent_ind][ind2] = pnt2
                next_to_thread_inds[i] = parent_ind


        # convert pixel to image coordinates (pixel -> meters)
        xs1_meter = []
        xs2_meter = []
        convertPixelToMeterPoints(cam_mat, [xs1_pixels], xs1_meter)
        convertPixelToMeterPoints(cam_mat, [xs2_pixels], xs2_meter)
        xs1_meter = xs1_meter[0]
        xs2_meter = xs2_meter[0]

        # MatchKeypointsAndGetEssentialMatNcc(self.imgLeft, self.imgRight)
        #ess_mat, xs1_meter, xs2_meter = MatchKeypointsAndGetEssentialMatSift(8, cam_mat, image1, image2, do_norming = True, debug = debug)
        ess_mat, cons_set_inds = MatchKeypointsAndGetEssentialMatNarrowBaselineCore(8, cam_mat, xs1_meter, xs2_meter, image1, image2, xs1_pixels, xs2_pixels, debug=debug)
        if debug >= 3: print("ess_mat=\n{0}".format(ess_mat))

        cons_xs1_meter = [xs1_meter[i] for i in cons_set_inds]
        cons_xs2_meter = [xs2_meter[i] for i in cons_set_inds]

        suc, ess_mat_refined = RefineFundMat(ess_mat, cons_xs1_meter, cons_xs2_meter, debug=debug)
        if not suc:
            print("ind2={0} can't refine ess mat".format(img2_id))
        else:
            if debug >= 3: print("refined_ess_mat=\n{0}".format(ess_mat_refined))

        #
        world_R = None
        world_T = None
        suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat_refined, cons_xs1_meter, cons_xs2_meter, debug=debug)
        if not suc:
            print("Failed E->R,T")
        else:
            ess_T = skewSymmeticMat(ess_Tvec)
            ess_wvec, ess_wang = logSO3(ess_R)
            print("R|T: w={0} ang={1}deg\n{2}\n{3}".format(ess_wvec, math.degrees(ess_wang), ess_R, ess_Tvec))

            # map to the world frame
            head_R =  cam_poses_R[last_reg_frame_ind]
            head_T =  cam_poses_T[last_reg_frame_ind]

            world_R = np.dot(head_R, ess_R)
            world_T = np.dot(head_R, ess_Tvec) + head_T

        cam_poses_R.append(world_R)
        cam_poses_T.append(world_T)

        # on failure to find [R,T] the head_ind doesn't change
        if not world_R is None:
            last_reg_frame_ind = ind2
            head_pixels = xs2_pixels
            head_to_thread_inds = next_to_thread_inds

        # update cursor
        img1_gray = img2_gray

    print("num_tracked_points={0}".format(num_tracked_points))
    print("cam_poses_R={0}".format(cam_poses_R))
    print("cam_poses_T={0}".format(cam_poses_T))

    return cam_poses_R, cam_poses_T

def RunReconstructionOfIsolatedImagePairs():
    debug = 3
    cam_mat = np.array([
        [5.7231451642124046e+02, 0., 3.2393613004134221e+02],
        [0., 5.7231451642124046e+02, 2.8464798761067397e+02],
        [0., 0., 1.]])

    image1 = cv2.imread("/home/mmore/Pictures/roshensweets_201704101700/is/scene00001.png")
    assert not image1 is None

    kpd = cv2.xfeatures2d.SIFT_create(nfeatures=50)  # OpenCV-contrib-3

    #
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    kp1 = kpd.detect(img1_gray, None)
    points = np.array([kp.pt for kp in kp1], np.float32)

    head_ind = 0

    # success ind2: 2,5
    for ind2 in [5]:
    #for ind2 in range(head_ind+1, 519):
        print("ind2={0}".format(ind2))
        img2_path = "/home/mmore/Pictures/roshensweets_201704101700/is/scene{0:05}.png".format(ind2)
        image2 = cv2.imread(img2_path)
        assert not image2 is None

        img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        #
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, points, None)

        status = status.ravel()
        xs_per_image1 = [p for s, p in zip(status, points) if s]
        xs_per_image2 = [p for s, p in zip(status, next_pts) if s]

        if debug >= 94: ShowMatches("initial match", image1, image2, xs_per_image1, xs_per_image2)

        # convert matched points pixels->meters
        xs_per_image1_meter = []
        xs_per_image2_meter = []
        convertPixelToMeterPoints(cam_mat, [xs_per_image1], xs_per_image1_meter)
        convertPixelToMeterPoints(cam_mat, [xs_per_image2], xs_per_image2_meter)
        xs_per_image1_meter = xs_per_image1_meter[0]
        xs_per_image2_meter = xs_per_image2_meter[0]

        #
        ess_mat, xs1_meter, xs2_meter = MatchKeypointsAndGetEssentialMatNarrowBaseline(8, cam_mat,  image1,image2, do_norming=False, debug=debug)
        if debug >= 3: print("ess_mat=\n{0}".format(ess_mat))

        # ShowMatches("matches ess mat", self.imgLeft, self.imgRight, np.dot(xs1_meter,cam_mat.T).astype(np.int32), np.dot(xs2_meter,cam_mat.T).astype(np.int32))

        suc, ess_mat_refined = RefineFundMat(ess_mat, xs1_meter, xs2_meter, debug=debug)
        if not suc:
            print("ind2={0} can't refine ess mat".format(ind2))
            continue

        ess_mat_refined = ess_mat
        if debug >= 3: print("refined_ess_mat=\n{0}".format(ess_mat_refined))

        suc, ess_R, ess_Tvec = ExtractRotTransFromEssentialMat(ess_mat_refined, xs1_meter, xs2_meter, debug=debug)
        if not suc:
            print("Failed E->R,T")
            continue

        ess_T = skewSymmeticMat(ess_Tvec)
        ess_wvec, ess_wang = logSO3(ess_R)
        print("R|T: w={0} ang={1}deg\n{2}\n{3}".format(ess_wvec, math.degrees(ess_wang), ess_R, ess_Tvec))
    pass


def TestExtractRTdNFromPlanarHomography():
    # MASKS page 138, example 5.20
    H = np.array([[5.404, 0, 4.436], [0, 4, 0], [-1.236, 0, 3.804]])
    ExtractRotTransFromPlanarHomography(H, None, None, 3)
    print()

def RunVisualizeTwoHomogDecomp():
    data = np.loadtxt("homog_two.txt")

    mask1 = data[:,2] > 0
    plt.plot(data[mask1,0], data[mask1,1], '.')
    plt.plot(data[~ mask1,0], data[~ mask1,1], 'r.')
    plt.show()

    print("")

def EssentialMatSvd(ess_mat, u=None, vt=None, check_ess_mat=True, check_post_cond=True):
    """
    Computes SVD of the essential matrix (for which sig1==sig2, sig3==0).
    source: "An Efficient Solution to the Five-Point Relative Pose Problem", Nister, 2004, page 6
    """
    if check_ess_mat:
        perr_msg = [""]
        assert IsEssentialMat(ess_mat, perr_msg), "require valid essential mat, " + perr_msg[0]

    ea = ess_mat[0,:]
    eb = ess_mat[1,:]
    ec = ess_mat[2,:]
    vc_cand1 = np.cross(ea, eb) # ea x eb
    vc_cand2 = np.cross(ea, ec) # ea x ec
    vc_cand3 = np.cross(eb, ec) # eb x ec

    # (vc, vc_len, ea)
    vc_items = [(vc_cand1, LA.norm(vc_cand1), ea),
                (vc_cand2, LA.norm(vc_cand2), ea),
                (vc_cand3, LA.norm(vc_cand3), eb)]
    vc_items.sort(key=lambda item: item[1]) # choose longest Vc candidate
    vc_item = vc_items[1]

    # V
    vc = vc_item[0] / vc_item[1]
    va = vc_item[2] / LA.norm(vc_item[2])
    vb = np.cross(vc, va)
    if vt is None:
        vt = np.zeros_like(ess_mat)
    vt[0,:] = va[:]
    vt[1,:] = vb[:]
    vt[2,:] = vc[:]

    # U
    ua = np.dot(ess_mat, va)
    ua /= LA.norm(ua)
    ub = np.dot(ess_mat, vb)
    ub /= LA.norm(ub)
    uc = np.cross(ua, ub)
    if u is None:
        u = np.zeros_like(ess_mat)
    u[:,0] = ua[:]
    u[:,1] = ub[:]
    u[:,2] = uc[:]

    if check_post_cond:
        sig1 = LA.norm(np.dot(ess_mat, va))
        sig2 = LA.norm(np.dot(ess_mat, vb))
        sig3 = LA.norm(np.dot(ess_mat, vc))
        assert np.isclose(sig1, sig2), "two equal singular values"
        assert np.isclose(0, sig3), "two equal singular values"
        ess_tmp = np.dot(u, np.diag([sig1,sig2,sig3])).dot(vt)
        assert np.allclose(ess_mat, ess_tmp), "created invalid SVD decomposition of E"
    return u, vt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="debug level; {0: no debugging, 1: errors, 2: warnings, 3: debug, 4: interactive}", type=int, default=2)
    parser.add_argument("--job", help="specify worker thread to run", type=int, default=2)
    parser.add_argument("--slam", help="[1,3] slam impl", type=int, default=3)
    parser.add_argument("--float", help="[f32, f64, f128]", type=str, default="f32")
    parser.add_argument("--mpmath", help="[0 or 1]", type=int, default=0)
    args = parser.parse_args()

    #TestSampsonDistance()
    #testCorrespondencePoly6()
    #TestHomography()
    #TestEssMat(args)
    #TestExtractRTdNFromPlanarHomography()
    #RunVisualizeTwoHomogDecomp()
    #RunGenerateVirtualPointsProjectAndReconstruct()
    #RunReconstructionOfIsolatedImagePairs()
    #RunReconstructionSequential()
    demo = ReconstructDemo(); demo.mainRun(args)
