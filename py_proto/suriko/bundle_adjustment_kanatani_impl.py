from suriko.obs_geom import *

def FillIntrinsics3x3(intrinsic_params, same_focal_length_xy, K_result):
    """
    :param intrinsic_params: (f | fx fy) u0 v0
    """
    K_result.fill(0)
    in_ind = 0
    if same_focal_length_xy:
        f = intrinsic_params[in_ind]
        in_ind += 1
        K_result[0, 0] = f
        K_result[1, 1] = f
    else:
        fx = intrinsic_params[in_ind]
        fy = intrinsic_params[in_ind+1]
        in_ind += 2
        K_result[0, 0] = fx
        K_result[1, 1] = fy

    u0 = intrinsic_params[in_ind+0]
    v0 = intrinsic_params[in_ind+1]
    f0 = 1
    K_result[0, 2] = u0
    K_result[1, 2] = v0
    K_result[2, 2] = f0

def NormalizeOrRevertRTInternal(rt, R0, T0, world_scale, normalize_or_revert, check_back_conv=True):
    Rk, Tk = rt
    if normalize_or_revert == True:
        newRk = Rk.dot(R0.T)
        newTk = world_scale * (Tk - Rk.dot(R0.T).dot(T0))
        result_rt = (newRk, newTk)
    else:
        revertRk = Rk.dot(R0)
        Tktmp = Tk / world_scale
        revertTk = Tktmp + Rk.dot(T0)
        result_rt = (revertRk, revertTk)

    if check_back_conv:
        back_rt = NormalizeOrRevertRTInternal(result_rt, R0, T0, world_scale, not normalize_or_revert,
                                             check_back_conv=False)
        assert np.allclose(rt[0], back_rt[0], atol=1e-3), "Error in normalization or reverting"
        assert np.allclose(rt[1], back_rt[1], atol=1e-3), "Error in normalization or reverting"
    return result_rt


def NormalizeOrRevertPointInternal(X3D, R0, T0, world_scale, normalize_or_revert, check_back_conv=True):
    if normalize_or_revert == True:
        newX = R0.dot(X3D) + T0
        newX *= world_scale
        result_x = newX
    else:
        X3Dtmp = X3D / world_scale
        result_x = R0.T.dot(X3Dtmp - T0)

    if check_back_conv:
        back_x = NormalizeOrRevertPointInternal(result_x, R0, T0, world_scale, not normalize_or_revert,
                                               check_back_conv=False)
        assert np.allclose(X3D, back_x, atol=1e-3), "Error in normalization or reverting"
    return result_x

class WorldNormalizer:

    """ Performs normalization, so that (R0,T0) is the identity rotation plus zero translation and T1y=1."""
    def __init__(self, salient_points, camera_poses, bundle_pnt_ids, t1y_norm, unity_comp_ind):
        # check t1y!=0 and scene can be normalized
        get0_from1 = SE3AFromB(camera_poses[0], camera_poses[1])
        t1 = get0_from1[1]
        assert not np.isclose(0, t1[unity_comp_ind]), "can't normalize T1 with component {} (into unity)".format(t1, unity_comp_ind)

        self.bundle_pnt_ids = bundle_pnt_ids # ids of points for BA processing
        self.world_pnts = salient_points
        self.framei_from_world_RT_list = camera_poses
        # normalization state
        self.t1y_norm = t1y_norm
        self.unity_comp_ind = unity_comp_ind
        # normalization state
        self.R0 = None
        self.T0 = None
        self.world_scale = None

    def NormalizeWorldInplaceInternal(self):
        """ An optimization problem is indeterminant as is, so the boundary condition is introduced:
        R0=Identity, T0=zeros(3), t2y=1. So the input world's structure is transformed (normalized) into conformed one.
        """
        check_post_cond = True
        if check_post_cond:
            framei_from_world_RT_list_prenorm = self.framei_from_world_RT_list.copy()
            world_pnts_prenorm = self.world_pnts.copy()

        R0, T0 = self.framei_from_world_RT_list[0]
        R1, T1 = self.framei_from_world_RT_list[1]

        R0, T0 = R0.copy(), T0.copy()  # initial values of R0,T0

        # translation vector from frame0 to frame1
        get0_from1 = SE3AFromB((R0, T0), (R1, T1))
        initial_camera_shift = get0_from1[1]

        # make y-component of the first camera shift a unity (formula 27) T1y==1
        world_scale = self.t1y_norm / initial_camera_shift[self.unity_comp_ind]
        #world_scale = 1

        for pnt_id in self.bundle_pnt_ids:
            pnt_ind = pnt_id
            X3D = self.world_pnts[pnt_ind]
            newX = NormalizeOrRevertPointInternal(X3D, R0, T0, world_scale, normalize_or_revert=True)
            self.world_pnts[pnt_ind] = newX

        for frame_ind in range(0, len(self.framei_from_world_RT_list)):
            rt = self.framei_from_world_RT_list[frame_ind]
            new_rt = NormalizeOrRevertRTInternal(rt, R0, T0, world_scale, normalize_or_revert=True)
            self.framei_from_world_RT_list[frame_ind] = new_rt

        self.R0 = R0
        self.T0 = T0
        self.world_scale = world_scale

        if check_post_cond:
            pmsg = ['']
            assert CheckWorldIsNormalized(self.framei_from_world_RT_list, self.bundle_pnt_ids, self.t1y_norm, self.unity_comp_ind, pmsg), pmsg[0]

    def RevertNormalization(self):
        R0, T0, world_scale = self.R0, self.T0, self.world_scale
        # revert unity transformation
        for pnt_id in self.bundle_pnt_ids:
            pnt_ind = pnt_id
            X3D = self.world_pnts[pnt_ind]
            revertX = NormalizeOrRevertPointInternal(X3D, R0, T0, world_scale, normalize_or_revert=False)
            self.world_pnts[pnt_ind] = revertX

        for frame_ind in range(0, len(self.framei_from_world_RT_list)):
            rt = self.framei_from_world_RT_list[frame_ind]
            revert_rt = NormalizeOrRevertRTInternal(rt, R0, T0, world_scale, normalize_or_revert=False)
            self.framei_from_world_RT_list[frame_ind] = revert_rt

        # if modifications were made after normalization, the reversion process won't modify the scene into
        # the initial (pre-normalization) state


def NormalizeWorldInplace(salient_points, camera_poses, bundle_pnt_ids, t1y, unity_comp_ind):
    scene_normalizer = WorldNormalizer(salient_points, camera_poses, bundle_pnt_ids, t1y, unity_comp_ind)
    scene_normalizer.NormalizeWorldInplaceInternal()
    return scene_normalizer

def CheckWorldIsNormalized(camera_poses, bundle_pnt_ids, t1y, unity_comp_ind, msg, eps=1e-1):
    # the first frame is the identity
    rt0 = camera_poses[0]
    if not np.allclose(np.identity(3), rt0[0], atol=eps):
        msg[0] = "R0=Identity but was\n{}".format(rt0[0])
        return False
    if not np.allclose(np.zeros(3), rt0[1], atol=eps):
        msg[0] = "T0=zeros(3) but was {}".format(rt0[1])
        return False

    # the second frame has translation of unity length
    rt1 = SE3Inv(camera_poses[1])
    t1 = rt1[1]
    if not np.allclose(t1y, t1[unity_comp_ind], atol=eps):
        msg[0] = "expected T1y=1 but T1 was {}".format(t1)
        return False
    return True

def ReprojError(points_life, proj_mat_list, world_pnts):
    """ Computes reprojection error - 3D points are projected using given projection matrices into 2D pixel points.
    These estimated points are compared with provided 2D pixels.
    :param proj_mat_list: the list of projection matrices P[3x4]
    """
    err_sum = 0.0

    frames_count = len(proj_mat_list)
    for frame_ind in range(0, frames_count):
        P = proj_mat_list[frame_ind]

        for pnt_life in points_life:
            pnt_id = pnt_life.track_id
            pnt_ind = pnt_id
            pnt_life = points_life[pnt_ind]
            pnt3D_world = world_pnts[pnt_life.track_id]

            if pnt3D_world is None: continue

            corner_pix = pnt_life.points_list_pixel[frame_ind]
            if corner_pix is None: continue

            x3D_pix = np.dot(P, np.hstack( (pnt3D_world,1) ))
            assert not np.isclose(0, x3D_pix[2]), "homog 2D point can't have Z=0"
            x = x3D_pix[0] / x3D_pix[2]
            y = x3D_pix[1] / x3D_pix[2]
            one_err = (x - corner_pix[0]) ** 2 + (y - corner_pix[1]) ** 2

            err_sum += one_err

    return err_sum

class PointPatch:
    def __init__(self, pnt_id=None, pnt3D_world=None):
        self.pnt_id = pnt_id
        self.pnt3D_world = pnt3D_world

    def AddDelta(self, var_ind, addendum):
        assert var_ind < 3
        self.pnt3D_world[var_ind] += addendum

    def SetPoint(self, pnt_id, pnt3D_world):
        assert not pnt_id is None
        self.pnt_id = pnt_id
        self.pnt3D_world = pnt3D_world

    def Copy(self):
        c = PointPatch(None, None)
        c.pnt_id = self.pnt_id
        if not self.pnt3D_world is None:
            c.pnt3D_world = self.pnt3D_world.copy()
        return c


class PackedFrameOptVars:
    """ Represents a vector of optimized frame variables [[(fx fy | f)  u0 v0] direct(Tx Ty Tz) direct(Wx Wy Wz)] where W=axis angle rotation.
    (T,W)=represents direct coordinate conversion - from camera to world
    """
    def __init__(self, size, elem_type):
        self.frame_ind = None
        self.variable_intrinsics = None # True to have components from camera intrinsics K
        self.same_focal_length_xy = None # True fx=fy, that is K[0,0]=K[1,1]
        self.packed_opt_vars = np.zeros(size, dtype=elem_type) # vector of optimized variables

    def __len__(self): return len(self.packed_opt_vars)
    def __getitem__(self, i): return self.packed_opt_vars[i]
    def __setitem__(self, i, value): self.packed_opt_vars[i] = value

    def Copy(self):
        c = PackedFrameOptVars(len(self.packed_opt_vars), self.packed_opt_vars.dtype)
        c.frame_ind = self.frame_ind
        c.variable_intrinsics = self.variable_intrinsics
        c.same_focal_length_xy = self.same_focal_length_xy
        c.packed_opt_vars[:] = self.packed_opt_vars[:]
        return c


class FramePatch:
    def __init__(self, elem_type, frame_ind = None, K=None, same_focal_length_xy=None, cam_inverse_orient_rt=None):
        self.elem_type = elem_type
        self.frame_ind = frame_ind
        self.K = K  # camera intrinsics [3x3]=[fx 0 u0; 0 fy v0; 0 0 f0]
        self.same_focal_length_xy = same_focal_length_xy
        if same_focal_length_xy:
            assert np.isclose(self.K[0, 0], self.K[1, 1])

        # we can't have separate components (R and T) of camera orientation, because reprojection error calculation
        # uses inverse camera orientation (coordinates from world to camera) and Kanatani Bundle Adjustment uses
        # direct camera orientation (from camera to world). Thus if one slightly changes T_direct of some direct camera
        # orientation, we need to convert it to T_inverse to calculate an error, but it is only possible if R_direct
        # is available. Impossible: T_direct->T_inverse, possible: (R_direct,T_direct)->(R_inverse,T_inverse)
        self.cam_inverse_orient_rt = cam_inverse_orient_rt # (R_inverse, T_inverse)

    def SetK(self, frame_ind, K, same_focal_length_xy):
        if same_focal_length_xy:
            assert np.isclose(self.K[0, 0], self.K[1, 1])
        if not self.frame_ind is None:
            assert self.frame_ind == frame_ind
        self.frame_ind = frame_ind
        self.K = K
        self.same_focal_length_xy = same_focal_length_xy

    def SetCamInverseOrientRT(self, frame_ind, cam_inverse_orient_rt):
        if not self.frame_ind is None:
            assert self.frame_ind == frame_ind
        self.frame_ind = frame_ind
        self.cam_inverse_orient_rt = cam_inverse_orient_rt

    def Copy(self):
        c = FramePatch(self.elem_type)
        c.frame_ind = self.frame_ind
        c.same_focal_length_xy = self.same_focal_length_xy
        if not self.K is None:
            c.K = self.K.copy()
        c.cam_inverse_orient_rt = self.cam_inverse_orient_rt
        return c

    def PackIntoOptVarsVector(self, packed_vars: PackedFrameOptVars):
        out_ind = 0

        packed_vars.frame_ind = self.frame_ind
        packed_vars.variable_intrinsics = not self.K is None
        packed_vars.same_focal_length_xy = self.same_focal_length_xy

        if not self.K is None:
            if self.same_focal_length_xy:
                f = self.K[0,0]
                packed_vars[out_ind] = f
                out_ind += 1
            else:
                fx = self.K[0, 0]
                fy = self.K[1, 1]
                packed_vars[out_ind + 0] = fx
                packed_vars[out_ind + 1] = fy
                out_ind += 2
            u0 = self.K[0, 2]
            v0 = self.K[1, 2]
            packed_vars[out_ind + 0] = u0
            packed_vars[out_ind + 1] = v0
            out_ind += 2

        cam_direct_orient_rt = SE3Inv(self.cam_inverse_orient_rt)

        packed_vars[out_ind:out_ind + 3] = cam_direct_orient_rt[1][:]
        out_ind += 3

        suc, w_direct = AxisAngleFromRotMat(cam_direct_orient_rt[0])
        if not suc:
            w_direct = [0, 0, 0]

        packed_vars[out_ind:out_ind + 3] = w_direct[:]
        out_ind += 3
        assert out_ind == len(packed_vars), "can't put all variables into vector out_ind={} len(vars)={}".format(out_ind, len(packed_vars))

    def UnpackFromOptVarsVector(self, packed_opt_vars: PackedFrameOptVars):
        in_ind = 0

        self.frame_ind = packed_opt_vars.frame_ind
        self.same_focal_length_xy = packed_opt_vars.same_focal_length_xy

        if packed_opt_vars.variable_intrinsics:
            if self.K is None:
                self.K = np.zeros((3,3), dtype=self.elem_type)
            else:
                self.K.fill(0)
            self.K[2, 2] = 1

            if packed_opt_vars.same_focal_length_xy:
                f = packed_opt_vars[in_ind]
                self.K[0,0] = f
                self.K[1,1] = f
                in_ind += 1
            else:
                fx = packed_opt_vars[in_ind + 0]
                fy = packed_opt_vars[in_ind + 1]
                self.K[0,0] = fx
                self.K[1,1] = fy
                in_ind += 2
            u0 = packed_opt_vars[in_ind + 0]
            v0 = packed_opt_vars[in_ind + 1]
            self.K[0, 2] = u0
            self.K[1, 2] = v0
            in_ind += 2

        cam_orient_direct_T = packed_opt_vars[in_ind:in_ind + 3]
        in_ind += 3

        cam_orient_direct_W = packed_opt_vars[in_ind:in_ind + 3]
        in_ind += 3

        cam_orient_direct_R = RotMatFromAxisAngleOrIdentity(cam_orient_direct_W)
        self.cam_inverse_orient_rt = SE3Inv((cam_orient_direct_R, cam_orient_direct_T))

        assert in_ind == len(packed_opt_vars.packed_opt_vars), "can't get all variables from vector, got={} len(vars)={}".format(in_ind, len(packed_opt_vars))

def BundleAdjustmentKanataniReprojError(points_life, world_pnts, framei_from_world_RT_list,
                                        bundle_pnt_ids=None, overwrite_track_id=None, overwrite_x3D=None,
                                        overwrite_frame_ind=None, overwrite_K=None, overwrite_cam_inverse_orient_rt=None,
                                        cam_mat_pixel_from_meter=None,
                                        cam_mat_pixel_from_meter_list=None,
                                        pinf_pnt_count=None):
    if bundle_pnt_ids is None:
        bundle_pnt_ids = []
        for pnt_life in points_life:
            pnt3D_world = world_pnts[pnt_life.track_id]
            if pnt3D_world is None: continue
            pnt_id = pnt_life.track_id
            assert not pnt_id is None, "Required point track id to identify the track"
            bundle_pnt_ids.append(pnt_id)

    if not overwrite_track_id is None:
        assert not overwrite_x3D is None, "Provide 3D world point to overwrite"

    frames_count = len(framei_from_world_RT_list)

    err_sum = 0.0
    inf_pnt_count = 0

    for frame_ind in range(0, frames_count):
        R_inv,T_inv = framei_from_world_RT_list[frame_ind]

        cam = cam_mat_pixel_from_meter
        if not cam_mat_pixel_from_meter_list is None:
            cam = cam_mat_pixel_from_meter_list[frame_ind]

        if overwrite_frame_ind == frame_ind:
            if not overwrite_cam_inverse_orient_rt is None:
                R_inv, T_inv = overwrite_cam_inverse_orient_rt
            if not overwrite_K is None:
                cam = overwrite_K

        for pnt_id in bundle_pnt_ids:
            pnt_ind = pnt_id
            pnt_life = points_life[pnt_ind]
            pnt3D_world = world_pnts[pnt_life.track_id]
            if overwrite_track_id == pnt_life.track_id:
                pnt3D_world = overwrite_x3D

            if pnt3D_world is None: continue

            corner_pix = pnt_life.points_list_pixel[frame_ind]
            if corner_pix is None: continue

            x3D_cam = SE3Apply((R_inv,T_inv), pnt3D_world)
            x3D_pix = np.dot(cam, x3D_cam)
            zero_z = np.isclose(0, x3D_pix[2])
            if zero_z:
                assert not zero_z, "homog 2D point can't have Z=0"
                #inf_pnt_count += 1
                continue

            x = x3D_pix[0] / x3D_pix[2]
            y = x3D_pix[1] / x3D_pix[2]
            one_err = (x - corner_pix[0]) ** 2 + (y - corner_pix[1]) ** 2

            err_sum += one_err

    if not pinf_pnt_count is None:
        pinf_pnt_count[0] = inf_pnt_count

    return err_sum

def BundleAdjustmentKanataniReprojErrorNew(points_life, cam_mat_pixel_from_meter_list, world_pnts, framei_from_world_RT_list, bundle_pnt_ids=None, overwrite_track_id=None, overwrite_x3D=None, overwrite_frame_ind=None, overwrite_rt=None):
    if bundle_pnt_ids is None:
        bundle_pnt_ids = []
        for pnt_life in points_life:
            pnt3D_world = world_pnts[pnt_life.track_id]
            if pnt3D_world is None: continue
            pnt_id = pnt_life.track_id
            assert not pnt_id is None, "Required point track id to identif the track"
            bundle_pnt_ids.append(pnt_id)

    if not overwrite_track_id is None:
        assert not overwrite_x3D is None, "Provide 3D world point to overwrite"
    if not overwrite_frame_ind is None:
        assert not overwrite_rt is None, "Provide frame R,T to overwrite"

    frames_count = len(framei_from_world_RT_list)

    err_sum = 0.0

    for frame_ind in range(0, frames_count):
        if not overwrite_frame_ind is None and overwrite_frame_ind == frame_ind:
            R, T = overwrite_rt
        else:
            R, T = framei_from_world_RT_list[frame_ind]

        cam_mat_pixel_from_meter = cam_mat_pixel_from_meter_list[frame_ind]
        for pnt_id in bundle_pnt_ids:
            pnt_ind = pnt_id
            pnt_life = points_life[pnt_ind]
            if not overwrite_track_id is None and overwrite_track_id == pnt_life.track_id:
                pnt3D_world = overwrite_x3D
            else:
                pnt3D_world = world_pnts[pnt_life.track_id]

            if pnt3D_world is None: continue

            corner_pix = pnt_life.points_list_pixel[frame_ind]
            if corner_pix is None: continue

            x3D_cam = SE3Apply((R, T), pnt3D_world)
            x3D_pix = np.dot(cam_mat_pixel_from_meter, x3D_cam)
            assert not np.isclose(0, x3D_pix[2]), "homog 2D point can't have Z=0"
            x = x3D_pix[0] / x3D_pix[2]
            y = x3D_pix[1] / x3D_pix[2]
            one_err = (x - corner_pix[0]) ** 2 + (y - corner_pix[1]) ** 2

            err_sum += one_err

    return err_sum

class BundleAdjustmentKanatani:
    """
    Performs Bundle adjustment (BA) inplace. Iteratively shifts world points and cameras position and orientation so
    that the reprojection error is minimized.
    TODO: think about it: the synthetic scene, corrupted with noise, probably will not be repaired to one with zero reprojection error.
    source: "Bundle adjustment for 3-d reconstruction" Kanatani Sugaya 2010
    """
    def __init__(self, min_err_change_abs = 1e-8, min_err_change_rel = 1e-4, max_iter = None, debug = 0):
        """
        :param min_err_change_abs: stops if error change between two consecutive iterations is less than this value
        :param min_err_change_rel: stops if ratio of error change between two consecutive iterations and the first error is less than this value
        :param max_iter: stops when iterating over this number of iterations
        """
        self.points_life = None
        self.bundle_pnt_ids = None # ids of points for BA processing
        self.variable_intrinsics = None # camera intrinsics are the group of [fk,u0,v0], fk=focal length, (u0,v0)=princiapal point
        self.same_focal_length_xy = False # True for fx=fy
        self.cam_mat_pixel_from_meter = None # const camera matrix for all frames, variable_intrinsics=False
        self.cam_mat_pixel_from_meter_list = None # specific camera matrix for each frame, variable_intrinsics=True
        self.world_pnts = None
        self.framei_from_world_RT_list = None
        self.elem_type = None
        self.min_err_change_abs = min_err_change_abs
        self.min_err_change_rel = min_err_change_rel # set None to skip the check of relative change of the proj error
        self.max_iter = max_iter # set None to skip the check of the maximum number of iterations
        self.max_hessian_factor = 1e6 # set None to skip the check of the hessian's factor
        self.debug = debug
        # switches
        self.debug_processX = True # True in Release
        self.debug_processR = True # True in Release
        self.debug_processT = True # True in Release
        self.compute_derivatives_mode = "closeform"
        self.compute_derivatives_mode_options = ["closeform", "finitedifference"]
        self.estimate_corrections_mode = "twophases"
        self.estimate_corrections_mode_options = ["naive","twophases","twophases+check_with_naive"]
        # computation result
        self.err_value_initial = None
        self.err_value = None
        # const
        self.t1y = 1.0 # const, y-component of the first camera shift, usually T1y==1
        self.unity_comp_ind = 1 # 0 for X, 1 for Y; index of T1 to be set to unity
        self.normalize_pattern = None
        self.normalize_pattern_frame0 = None

    def ComputeInplace(self, pnt_track_list, salient_points, camera_poses,
                       cam_mat_pixel_from_meter=None, cam_mat_pixel_from_meter_list=None,
                       check_derivatives=False):
        """
        :return: True if optimization converges successfully.
        Stop conditions:
        1) If a change of error function slows down and becomes less than self.min_err_change
        NOTE: There is no sense to provide absolute threshold on error function because when noise exist, the error will
        not get close to zero.
        """
        assert not cam_mat_pixel_from_meter is None or not cam_mat_pixel_from_meter_list is None, "camera matrix must be specified for all/each frame"
        self.points_life = pnt_track_list

        self.POINT_VARS = 3  # [X Y Z]

        self.cam_mat_pixel_from_meter = cam_mat_pixel_from_meter
        self.cam_mat_pixel_from_meter_list = cam_mat_pixel_from_meter_list

        self.__MarkOptVarsOrderDependency()
        if not cam_mat_pixel_from_meter is None:
            self.variable_intrinsics = False
            self.FRAME_VARS = 6  # [T1 T2 T3 W1 W2 W3], w=rotation axis in 'direct' SO3 frame
            self.INTRINSICS_VARS = 0
        else:
            assert not self.cam_mat_pixel_from_meter_list is None
            self.variable_intrinsics = True
            # w=rotation axis in 'direct' SO3 frame
            if self.same_focal_length_xy:
                self.FRAME_VARS = 9   # [[f     u0 v0] T1 T2 T3 W1 W2 W3]
                self.INTRINSICS_VARS = 3
            else:
                self.FRAME_VARS = 10  # [[fx fy u0 v0] T1 T2 T3 W1 W2 W3]
                self.INTRINSICS_VARS = 4
        self.__UpdateNormalizePattern()

        self.world_pnts = salient_points
        self.framei_from_world_RT_list = camera_poses

        assert self.compute_derivatives_mode in self.compute_derivatives_mode_options, "Possible modes of derivatives computation {}".format(self.compute_derivatives_mode_options)
        assert self.estimate_corrections_mode in self.estimate_corrections_mode_options, "Possible modes of correction estimation {}".format(self.estimate_corrections_mode_options)

        # select tracks (salient 3D points) to use for bundle adjustment
        # NOTE: further the pnt_ind refers to pnt_id in the array of bundle points
        bundle_pnt_ids = []
        for pnt_life in self.points_life:
            pnt3D_world = self.world_pnts[pnt_life.track_id]
            if pnt3D_world is None: continue
            pnt_id = pnt_life.track_id
            assert not pnt_id is None, "Required point track id to identif the track"
            bundle_pnt_ids.append(pnt_id)
        self.bundle_pnt_ids = bundle_pnt_ids

        self.elem_type = type(salient_points[bundle_pnt_ids[0]][0])

        scene_normalizer = NormalizeWorldInplace(salient_points, camera_poses, bundle_pnt_ids, self.t1y, self.unity_comp_ind)

        result = self.__ComputeOnNormalizedWorld(check_derivatives)

        # check world is still normalized after optimization
        pmsg = ['']
        assert CheckWorldIsNormalized(camera_poses, bundle_pnt_ids, self.t1y, self.unity_comp_ind, pmsg), pmsg[0]

        scene_normalizer.RevertNormalization()
        return result

    def __ComputeOnNormalizedWorld(self, check_derivatives):
        el_type = self.elem_type
        frames_count = len(self.framei_from_world_RT_list)
        points_count = len(self.bundle_pnt_ids)

        # +3 for R0=Identity[3x3]
        # +3 for T0=[0 0 0]
        # +1 for T1y=1
        KNOWN_FRAME_VARS_COUNT = 7

        check_data_is_normalized = True
        if check_data_is_normalized:
            pmsg=['']
            assert CheckWorldIsNormalized(self.framei_from_world_RT_list, self.bundle_pnt_ids, self.t1y, self.unity_comp_ind,pmsg), pmsg[0]

        POINT_COMPS = self.POINT_VARS
        FRAME_COMPS = self.FRAME_VARS

        # derivatives data
        gradE = np.zeros(POINT_COMPS * points_count + FRAME_COMPS * frames_count, dtype=el_type)  # derivative of vars
        gradE2_onlyW = np.zeros(POINT_COMPS * points_count + FRAME_COMPS * frames_count, dtype=el_type)
        deriv_second_point = np.zeros((POINT_COMPS * points_count, (POINT_COMPS)), dtype=el_type)  # rows=[X1 Y1 Z1 X2 Y2 Z2...] cols=[X Y Z]
        deriv_second_frame = np.zeros((FRAME_COMPS * frames_count, (FRAME_COMPS)), dtype=el_type)  # rows=[T1 T2 T3 W1 W2 W3...for each frame] cols=[T1 T2 T3 W1 W2 W3]
        # rows=[X1 Y1 Z1 X2 Y2 Z2...for each point] cols=[T1 T2 T3 W1 W2 W3...for each frame]
        deriv_second_pointframe = np.zeros((POINT_COMPS * points_count, FRAME_COMPS * frames_count), dtype=el_type)
        corrections = np.zeros(POINT_COMPS * points_count + FRAME_COMPS * frames_count, dtype=el_type)  # corrections of vars

        # data to solve linear equations
        left_side1 = np.zeros((FRAME_COMPS * frames_count - KNOWN_FRAME_VARS_COUNT, FRAME_COMPS * frames_count - KNOWN_FRAME_VARS_COUNT), dtype=el_type)
        right_side = np.zeros(FRAME_COMPS * frames_count - KNOWN_FRAME_VARS_COUNT, dtype=el_type)
        matG = np.zeros((FRAME_COMPS * frames_count - KNOWN_FRAME_VARS_COUNT, FRAME_COMPS * frames_count - KNOWN_FRAME_VARS_COUNT), dtype=el_type)

        MAX_ABS_DIST = 0.1
        MAX_REL_DIST = 0.1 # 14327.78415796-14328.10677215=0.32261419 => rtol=2.2e-5
        def IsClose(a, b):
            return np.allclose(a, b, atol=MAX_ABS_DIST, rtol=MAX_REL_DIST)

        world_pnts_revert_copy = self.world_pnts.copy()
        framei_from_world_K_list_revert_copy = None
        if not self.cam_mat_pixel_from_meter_list is None:
            framei_from_world_K_list_revert_copy = self.cam_mat_pixel_from_meter_list.copy()
        framei_from_world_RT_list_revert_copy = self.framei_from_world_RT_list.copy()

        self.err_value_initial = self.__ReprojError(None, None, None, None)
        self.err_value = self.err_value_initial

        if self.debug >= 3:
            print("initial reproj_err={}".format(self.err_value_initial))

        ####
        def CopySrcDst(src_points, src_Ks, src_rts, dst_points, dst_Ks, dst_rts):
            for i in range(0, len(self.world_pnts)):
                copy = None
                if not src_points[i] is None:
                    copy = src_points[i].copy()
                dst_points[i] = copy
            if not src_Ks is None:
                for i in range(0, len(src_Ks)):
                    copyK = src_Ks[i].copy()
                    dst_Ks[i] = copyK

            for i in range(0, len(src_rts)):
                copyR = src_rts[i][0].copy()
                copyT = src_rts[i][1].copy()
                dst_rts[i] = (copyR,copyT)

        # NOTE: we don't check absolute error, because corrupted with noise data may have arbitrary large reproj err

        it = 1
        hessian_factor = 0.0001  # hessian's diagonal multiplier
        while True:
            if self.compute_derivatives_mode == "closeform":
                self.__ComputeDerivativesCloseForm(points_count, frames_count, check_derivatives, gradE, gradE2_onlyW, deriv_second_point, deriv_second_frame, deriv_second_pointframe)
            elif self.compute_derivatives_mode == "finitedifference":
                finitdiff_eps = 1e-5
                self.__ComputeDerivativesFiniteDifference(points_count, frames_count, check_derivatives, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, finitdiff_eps, None, None, None, None)

                gradE_tmp = gradE.copy()
                deriv_second_point_tmp = deriv_second_point.copy()
                deriv_second_frame_tmp = deriv_second_frame.copy()
                deriv_second_pointframe_tmp = deriv_second_pointframe.copy()

                self.__ComputeDerivativesFiniteDifference(points_count, frames_count, check_derivatives, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, finitdiff_eps, gradE_tmp, deriv_second_point_tmp, deriv_second_frame_tmp, deriv_second_pointframe_tmp)
            else: assert False

            # backup current state (world points and camera orientations)
            # world_pnts_revert_copy[:] = self.world_pnts[:]
            # framei_from_world_RT_list_revert_copy[:] = self.framei_from_world_RT_list[:]
            CopySrcDst(self.world_pnts, self.cam_mat_pixel_from_meter_list, self.framei_from_world_RT_list,
                       world_pnts_revert_copy, framei_from_world_K_list_revert_copy, framei_from_world_RT_list_revert_copy)

            # loop to find a hessian factor which decreases the target optimization function
            err_value_change = None
            while True:
                # 1. the normalization (known R0,T0,T1y) is applied to corrections introducing gaps (plane->gaps)
                # 2. the linear system of equations is solved for vector of gapped corrections
                # 3. the gapped corrections are converted back to the plane vector (gaps->plane)
                # 4. the plane vector of corrections is used to adjust the optimization variables

                if self.estimate_corrections_mode == 'naive':
                    self.__EstimateCorrectionsNaive(points_count, frames_count, hessian_factor, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, corrections)
                elif self.estimate_corrections_mode == 'twophases':
                    corr_ref = None
                    self.__EstimateCorrectionsDecomposedInTwoPhases(points_count, frames_count, hessian_factor, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, matG, left_side1, right_side, corrections, corr_ref)
                elif self.estimate_corrections_mode == 'twophases+check_with_naive':
                    self.__EstimateCorrectionsNaive(points_count, frames_count, hessian_factor, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, corrections)
                    corr_ref = corrections.copy()
                    corrections.fill(0)
                    self.__EstimateCorrectionsDecomposedInTwoPhases(points_count, frames_count, hessian_factor, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, matG, left_side1, right_side, corrections, corr_ref)

                self.__ApplyCorrections(points_count, frames_count, corrections)

                err_value_new = self.__ReprojError(None, None, None, None)

                err_value_change = err_value_new - self.err_value

                target_fun_decreased = err_value_change < 0
                if target_fun_decreased:
                    break

                # now, the value of target minimization function increases, try again with different params

                # restore saved state
                # self.world_pnts[:] = world_pnts_revert_copy[:]
                # self.framei_from_world_RT_list[:] = framei_from_world_RT_list_revert_copy[:]
                CopySrcDst(world_pnts_revert_copy, framei_from_world_K_list_revert_copy, framei_from_world_RT_list_revert_copy, self.world_pnts, self.cam_mat_pixel_from_meter_list, self.framei_from_world_RT_list)

                debug_successfull_revertion = True
                if debug_successfull_revertion:
                    opt_fun_value_reverted = self.__ReprojError(None, None, None, None)
                    assert np.isclose(self.err_value, opt_fun_value_reverted), "error reverting to the pre-adjustment state"

                if not self.max_hessian_factor is None and hessian_factor > self.max_hessian_factor:
                    # prevent overflow for too big factors
                    break

                hessian_factor *= 10  # prefer more the Steepest descent

            # check error really decreases; it fails on overflow of hessian's factor
            err_msg = None
            target_fun_decreased = err_value_change < 0
            if not target_fun_decreased:
                # the only way for target function to decrease is hessian overflow
                err_msg = "hessian_factor_overflow"
                return False, err_msg # failed

            self.err_value = err_value_new

            if self.debug >= 3:
                print("it={} reproj_err={} hessian_factor={}".format(it, self.err_value, hessian_factor))

            # stop condition: change of error as absolute value
            if math.fabs(err_value_change) < self.min_err_change_abs:
                if self.debug >= 3:
                    print("err_value_change={} min_err_change_abs={}".format(err_value_change, self.min_err_change_abs))
                return True, err_msg  # success

            # stop condition: change of error related to the first error
            if not self.min_err_change_rel is None:
                err_value_change_rel = math.fabs(err_value_change/self.err_value)
                if err_value_change_rel < self.min_err_change_rel:
                    if self.debug >= 3:
                        print("err_value_change_rel={} min_err_change_rel={}".format(err_value_change_rel, self.min_err_change_rel))
                    return True, err_msg # success

            hessian_factor /= 10  # prefer more the Gauss-Newton
            it += 1

            if not self.max_iter is None and it > self.max_iter:
                err_msg = "max_iter"
                return False, err_msg # failed

        assert False # we won't get here

    def __ComputeDerivativesCloseForm(self, points_count, frames_count, check_derivatives, gradE, gradE2_onlyW, deriv_second_point, deriv_second_frame, deriv_second_pointframe):
        POINT_COMPS = self.POINT_VARS
        FRAME_COMPS = self.FRAME_VARS

        eps = 1e-5 # finite difference step to approximate derivative

        # compute point derivatives
        for pnt_ind, pnt_id in enumerate(self.bundle_pnt_ids):
            pnt_life = self.points_life[pnt_id]
            pnt3D_world = self.world_pnts[pnt_life.track_id]
            assert not pnt3D_world is None

            for frame_ind in range(0, frames_count):
                corner_pix = pnt_life.points_list_pixel[frame_ind]
                if corner_pix is None: continue

                K = self.CamMatPixelsFromMeter(frame_ind)
                f0 = K[2, 2]

                R_inverse,T_inverse = self.framei_from_world_RT_list[frame_ind]
                P = np.zeros((3, 4), dtype=self.elem_type)
                P[0:3, 0:3] = np.dot(K, R_inverse)
                P[0:3, 3] = np.dot(K, T_inverse)

                x3D_cam = SE3Apply((R_inverse,T_inverse), pnt3D_world)
                x3D_pix = np.dot(K, x3D_cam)
                pqr = x3D_pix

                # dX,dY,dZ
                # each row contains (p,q,r) derivatives for X (row=0), Y,Z variables
                point_pqr_deriv = np.zeros((POINT_COMPS,3), dtype=self.elem_type) # count([p,q,r])=3
                self.__ComputePointPqrDerivatives(P, point_pqr_deriv)

                for xyz_ind in range(0,POINT_COMPS):
                    ax = self.__OneSummandGradEByVar(f0, pqr, corner_pix, point_pqr_deriv[xyz_ind, 0], point_pqr_deriv[xyz_ind, 1], point_pqr_deriv[xyz_ind, 2])
                    self.__MarkOptVarsOrderDependency()
                    gradE[pnt_ind*POINT_COMPS+xyz_ind] += ax

                # second derivative Point-Point
                # derivative(p) = [P[0,0] P[0,1] P[0,2]]
                # derivative(q) = [P[1,0] P[1,1] P[1,2]]
                # derivative(r) = [P[2,0] P[2,1] P[2,2]]
                for var1 in range(0,POINT_COMPS): # [X Y Z]
                    for var2 in range(0,POINT_COMPS): # [X Y Z]
                        ax = self.__SecondDerivFromPqrDerivative(pqr, point_pqr_deriv, point_pqr_deriv, var1, var2)
                        deriv_second_point[pnt_ind*POINT_COMPS+var1, var2] += ax

            # all frames where the current point visible are processed, and now derivatives are ready and can be checked
            if check_derivatives:
                if self.elem_type == np.float32:
                    print("WARNING: finite difference of err fun, dependent on many points and frames may not work properly due to rounding errors")

                gradPoint_close_form = gradE[pnt_ind * POINT_COMPS:(pnt_ind + 1) * POINT_COMPS]
                deriv_second_point_close_form = deriv_second_point[pnt_ind * POINT_COMPS:(pnt_ind + 1) * POINT_COMPS, 0:POINT_COMPS]
                self.__CheckPointDerivatives(gradPoint_close_form, deriv_second_point_close_form, pnt_life.track_id, pnt3D_world, eps)

        # dT
        grad_frames_section = points_count * POINT_COMPS # frames goes after points
        # each column correspnds to [p,q,r] components, each row corresponds to one of frame componets
        frame_pqr_deriv = np.zeros((FRAME_COMPS, 3)) # columns = d(p,q,r)/by_frame_vars ; count([p,q,r])=3
        for frame_ind in range(0, frames_count):
            R_inverse, T_inverse = self.framei_from_world_RT_list[frame_ind]
            R_direct, T_direct = SE3Inv((R_inverse,T_inverse))

            K = self.CamMatPixelsFromMeter(frame_ind)
            fx = K[0, 0]
            fy = K[1, 1]
            if self.same_focal_length_xy:
                assert np.isclose(fx, fy)
            u0, v0, f0 = K[0:3, 2]

            grad_frame_offset = grad_frames_section + frame_ind * self.FRAME_VARS

            for pnt_ind, pnt_id in enumerate(self.bundle_pnt_ids):
                pnt_life =  self.points_life[pnt_id]
                pnt3D_world = self.world_pnts[pnt_life.track_id]
                assert not pnt3D_world is None

                corner_pix = pnt_life.points_list_pixel[frame_ind]
                if corner_pix is None: continue

                x3D_cam = SE3Apply((R_inverse, T_inverse), pnt3D_world)
                x3D_pix = np.dot(K, x3D_cam)
                pqr = x3D_pix
                P = np.dot(K, np.hstack((R_inverse, T_inverse.reshape(3, 1))))

                self.__MarkOptVarsOrderDependency()

                out_frame_ind = grad_frame_offset
                if self.variable_intrinsics:
                    intrinsic_vars_count = self.__ComputeIntrinsicsDerivatives(frame_ind, pnt_ind, K, pqr, corner_pix, frame_pqr_deriv, 0, gradE, out_frame_ind)
                    out_frame_ind += intrinsic_vars_count

                # translation gradient
                gradp_by_tcomps = -(fx * R_direct[:, 0] + u0 * R_direct[:, 2]) # dp/d(Tx,Ty,Tz)
                gradq_by_tcomps = -(fy * R_direct[:, 1] + v0 * R_direct[:, 2]) # dq/d(Tx,Ty,Tz)
                gradr_by_tcomps = -(                      f0 * R_direct[:, 2]) # dr/d(Tx,Ty,Tz)
                out_frame_local_ind = out_frame_ind - grad_frame_offset
                frame_pqr_deriv[out_frame_local_ind:out_frame_local_ind+3,0] = gradp_by_tcomps
                frame_pqr_deriv[out_frame_local_ind:out_frame_local_ind+3,1] = gradq_by_tcomps
                frame_pqr_deriv[out_frame_local_ind:out_frame_local_ind+3,2] = gradr_by_tcomps

                gradE_by_tcomps = self.__OneSummandGradEByVar(f0, pqr, corner_pix, gradp_by_tcomps, gradq_by_tcomps, gradr_by_tcomps) # size=[3]
                gradE[out_frame_ind:out_frame_ind+3] += gradE_by_tcomps
                out_frame_ind += 3  # pass [Tx,Ty,Tz] gradients

                # rotation gradient
                rot1 = fx * R_direct[:, 0] + u0 * R_direct[:, 2]
                rot2 = fy * R_direct[:, 1] + v0 * R_direct[:, 2]
                rot3 = f0 * R_direct[:, 2]
                gradp_by_wcomps = np.cross(rot1, pnt3D_world-T_direct)
                gradq_by_wcomps = np.cross(rot2, pnt3D_world-T_direct)
                gradr_by_wcomps = np.cross(rot3, pnt3D_world-T_direct)
                out_frame_local_ind = out_frame_ind - grad_frame_offset
                frame_pqr_deriv[out_frame_local_ind:out_frame_local_ind+3,0] = gradp_by_wcomps
                frame_pqr_deriv[out_frame_local_ind:out_frame_local_ind+3,1] = gradq_by_wcomps
                frame_pqr_deriv[out_frame_local_ind:out_frame_local_ind+3,2] = gradr_by_wcomps

                gradW_vec = self.__OneSummandGradEByVar(f0, pqr, corner_pix, gradp_by_wcomps, gradq_by_wcomps, gradr_by_wcomps)
                gradE[out_frame_ind:out_frame_ind+3] += gradW_vec

                rotation_gradient_via_quaternions = False
                if rotation_gradient_via_quaternions:
                    suc, w_direct_norm, w_ang = LogSO3New(R_direct)
                    if not suc:
                        # TODO: when this case happens?
                        continue

                    q = QuatFromRotationMat(R_direct)
                    #w_norm, w_ang = AxisPlusAngleFromQuat(q)
                    w_direct = w_direct_norm * w_ang

                    # source: "A Recipe on the Parameterization of Rotation Matrices", Terzakis, 2012
                    # partial derivatives of [3x3] rotation matrix with respect to quaternion components, formulas (33-36)
                    F = np.zeros((4, 3, 3), dtype=(self.elem_type))  # 4 matrices [3x3]
                    F[0,0,0:3] = [q[0], -q[3], q[2]]
                    F[0,1,0:3] = [q[3], q[0], -q[1]]
                    F[0,2,0:3] = [-q[2], q[1], q[0]]

                    F[1,0,0:3] = [q[1], q[2], q[3]]
                    F[1,1,0:3] = [q[2], -q[1], -q[0]]
                    F[1,2,0:3] = [q[3], q[0], -q[1]]

                    F[2,0,0:3] = [-q[2], q[1], q[0]]
                    F[2,1,0:3] = [q[1], q[2], q[3]]
                    F[2,2,0:3] = [-q[0], q[3], -q[2]]

                    F[3,0,0:3] = [-q[3], -q[0], q[1]]
                    F[3,1,0:3] = [q[0], -q[3], q[2]]
                    F[3,2,0:3] = [q[1], q[2], q[3]]
                    F *= 2

                    # partial derivatives of [4x1] quaternion components with respect to [3x1] axis-angle components, formulas (38-40)
                    G = np.zeros((4, 3), dtype=(self.elem_type))  # 3 vectors [4x1]
                    ang = w_ang
                    sin_ang2 = math.sin(ang/2)
                    cos_ang2 = math.cos(ang/2)
                    cos_diff = (0.5 * cos_ang2 - sin_ang2 / ang) / ang ** 2
                    G[0, 0] = -0.5 * w_direct[0] * sin_ang2 / ang # dq0/du1
                    G[1, 0] = sin_ang2 / ang + w_direct[0] * w_direct[0] * cos_diff  # dq1/du1
                    G[2, 0] = w_direct[0] * w_direct[1] * cos_diff # dq2/du1
                    G[3, 0] = w_direct[0] * w_direct[2] * cos_diff # dq2/du1

                    G[0, 1] = -0.5 * w_direct[1] * sin_ang2 / ang
                    G[1, 1] = w_direct[0] * w_direct[1] * cos_diff
                    G[2, 1] = sin_ang2 / ang + w_direct[1] * w_direct[1] * cos_diff
                    G[3, 1] = w_direct[1] * w_direct[2] * cos_diff

                    G[0, 2] = -0.5 * w_direct[2] * sin_ang2 / ang
                    G[1, 2] = w_direct[0] * w_direct[2] * cos_diff
                    G[2, 2] = w_direct[1] * w_direct[2] * cos_diff
                    G[3, 2] = sin_ang2 / ang + w_direct[2] * w_direct[2] * cos_diff

                    # formula 41
                    dR = np.zeros((3, 3, 3), dtype=(self.elem_type))  # 3 matrices [3x3]
                    for var1 in range(0,3): # each component of axis-angle representation
                        for var2 in range(0, 4):
                            mat33 = F[var2,:,:]
                            scalar = G[var2,var1]
                            dR[var1,:,:] += mat33 * scalar

                    #print("gradp_byw={}".format(gradp_byw))
                    #print("gradq_byw={}".format(gradq_byw))
                    #print("gradr_byw={}".format(gradr_byw))
                    pqr_bywi = np.zeros((3, 3), dtype=(self.elem_type))
                    for wi in range(0,3):
                        pqr_bywi[0:3, wi] = self.cam_mat_pixel_from_meter.dot(dR[wi, :, :].T).dot(
                            np.hstack((np.eye(3, 3), -T_direct.reshape((3, 1))))).dot(np.hstack((pnt3D_world, 1)))
                    gradp_by_wcomps = pqr_bywi[0,0:3]
                    gradq_by_wcomps = pqr_bywi[1,0:3]
                    gradr_by_wcomps = pqr_bywi[2,0:3]
                    #print("gradp_byw={}".format(gradp_byw))
                    #print("gradq_byw={}".format(gradq_byw))
                    #print("gradr_byw={}".format(gradr_byw))
                    #print()
                    gradW_vec2 = (pqr[0] / pqr[2] - corner_pix[0] / f0) * (pqr[2] * gradp_by_wcomps - pqr[0] * gradr_by_wcomps) + \
                                (pqr[1] / pqr[2] - corner_pix[1] / f0) * (pqr[2] * gradq_by_wcomps - pqr[1] * gradr_by_wcomps)
                    gradW_vec2 *= 2 / pqr[2] ** 2
                    gradE2_onlyW[grad_frame_offset+3:grad_frame_offset+6] += gradW_vec2

                out_frame_ind += 3  # pass [Rwx,Rwy,Rwz] gradients

                # partial second derivative of the Frame components [f u0 v0 T1 T2 T3 W1 W2 W3]
                # it's computation depends solely on the first derivative
                D2_Frame_Frame = True
                if D2_Frame_Frame:
                    for var1 in range(0,FRAME_COMPS):
                        for var2 in range(0,FRAME_COMPS):
                            ax = self.__SecondDerivFromPqrDerivative(pqr, frame_pqr_deriv, frame_pqr_deriv, var1, var2)
                            deriv_second_frame[frame_ind*FRAME_COMPS+var1, var2] += ax # scalar

                # partial second derivative of the Point-Frame components [T1 T2 T3 W1 W2 W3]
                D2_Point_Frame = True
                if D2_Point_Frame:
                    point_pqr_deriv = np.zeros((POINT_COMPS, 3), dtype=self.elem_type)  # count([p,q,r])=3
                    self.__ComputePointPqrDerivatives(P, point_pqr_deriv)

                    for var1 in range(0,POINT_COMPS): # X Y Z
                        for var2 in range(0,FRAME_COMPS): # f u0 v0 T1 T2 T3 W1 W2 W3
                            ax = self.__SecondDerivFromPqrDerivative(pqr, point_pqr_deriv, frame_pqr_deriv, var1, var2)
                            deriv_second_pointframe[pnt_ind*POINT_COMPS+var1, frame_ind*FRAME_COMPS+var2] += ax


            pass # loop through points

            # all points in the current frame are processed, and now derivatives are ready and can be checked
            if check_derivatives:
                grad_frame_close_form = gradE[grad_frame_offset:out_frame_ind]
                grad_frame_close_form_onlyW = gradE2_onlyW[grad_frame_offset:out_frame_ind]
                deriv_second_frame_close_form = deriv_second_frame[frame_ind * FRAME_COMPS:(frame_ind + 1) * FRAME_COMPS, 0:FRAME_COMPS]
                deriv_second_allpoints_frame_close_form = deriv_second_pointframe[:,frame_ind * FRAME_COMPS:(frame_ind + 1) * FRAME_COMPS]

                self.__CheckFrameDerivatives(grad_frame_close_form, grad_frame_close_form_onlyW,
                                             deriv_second_frame_close_form,
                                             deriv_second_allpoints_frame_close_form,
                                             frame_ind, K, T_direct, R_direct, eps)

        pass # frames section

        c1 = np.any(~np.isfinite(gradE))
        c2 = np.any(~np.isfinite(deriv_second_point))
        c3 = np.any(~np.isfinite(deriv_second_frame))
        c4 = np.any(~np.isfinite(deriv_second_pointframe))
        if c1 or c2 or c3 or c4:
            assert False, "Derivatives must be real numbers"
        return None

    def __ComputeDerivativesFiniteDifference(self, points_count, frames_count, check_derivatives,
                                      gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, finitdiff_eps,
                                      gradE_hint, deriv_second_point_hint, deriv_second_frame_hint, deriv_second_pointframe_hint):
        """ Computes finite difference approximation of derivatives.
        :param finitdiff_eps: finite difference step to approximate derivative
        """
        POINT_VARS = self.POINT_VARS
        FRAME_VARS = self.FRAME_VARS

        affected_gradE = gradE[0:points_count*POINT_VARS+frames_count*FRAME_VARS]
        affected_gradE[:] = float('nan')
        affected_deriv_second_point = deriv_second_point[0:points_count*POINT_VARS, 0:POINT_VARS]
        affected_deriv_second_point[:] = float('nan')
        affected_deriv_second_frame = deriv_second_frame[0:frames_count*FRAME_VARS,0:FRAME_VARS]
        affected_deriv_second_frame[:] = float('nan')
        affected_deriv_second_pointframe = deriv_second_pointframe[0:points_count*POINT_VARS,0:frames_count*FRAME_VARS]
        affected_deriv_second_pointframe[:] = float('nan')

        # compute point derivatives
        for pnt_ind, pnt_id in enumerate(self.bundle_pnt_ids):
            pnt_life = self.points_life[pnt_id]
            pnt3D_world = self.world_pnts[pnt_life.track_id]
            assert not pnt3D_world is None

            # estimate dX,dY,dZ
            gradPoint = gradE[pnt_ind*self.POINT_VARS:(pnt_ind+1)*self.POINT_VARS]
            for xyz_ind in range(0, self.POINT_VARS):
                gradPoint[xyz_ind] = self.EstimateFirstPartialDerivPoint(pnt_id, pnt3D_world, xyz_ind, finitdiff_eps)

            if check_derivatives:
                gradPoint_hint = gradE_hint[pnt_ind*self.POINT_VARS:(pnt_ind+1)*self.POINT_VARS]
                print("gradXYZ_hint={} gradXYZ={}".format(gradPoint_hint, gradPoint))

                close = self.IsClose(gradPoint_hint, gradPoint)
                if not close:
                    assert False, "grad(Point) mismatch"

            # 2nd derivative Point-Point
            deriv_second_point_finitdif = deriv_second_point[pnt_ind*self.POINT_VARS:(pnt_ind+1)*self.POINT_VARS, 0:POINT_VARS]
            point_patch = PointPatch(pnt_id, pnt3D_world)
            for var1 in range(0,POINT_VARS):
                for var2 in range(0,POINT_VARS):
                    deriv_second_point_finitdif[var1,var2] = self.EstimateSecondPartialDerivPoint(point_patch, var1, var2, finitdiff_eps)

            if check_derivatives:
                d2_point_hint = deriv_second_point_hint[pnt_ind*self.POINT_VARS:(pnt_ind+1)*self.POINT_VARS, 0:POINT_VARS]
                print("deriv2nd_hint deriv2nd\n{}\n{}".format(d2_point_hint, deriv_second_point_finitdif))

                close = self.IsClose(d2_point_hint, deriv_second_point_finitdif)
                if not close:
                    assert False, "D2 Point mismatch"

            check_point_hessian_is_invertible = True
            if check_point_hessian_is_invertible:
                point_hessian = deriv_second_point_finitdif
                is_inverted = True
                try:
                    point_hessian_inv = LA.inv(point_hessian)
                except LA.LinAlgError:
                    print("ERROR: inverting 3x3 E, pnt_id={}".format(pnt_id))
                    is_inverted = False
                assert is_inverted, "Can't invert point hessian for pnt_id={}".format(pnt_id)

        # dT
        grad_frames_section = points_count * POINT_VARS  # frames goes after points
        for frame_ind in range(0, frames_count):
            cam_inverse_orient_rt = self.framei_from_world_RT_list[frame_ind]
            R_inverse, T_inverse = cam_inverse_orient_rt
            cam_direct_orient_rt = SE3Inv((R_inverse,T_inverse))
            R_direct, T_direct = cam_direct_orient_rt

            K = self.CamMatPixelsFromMeter(frame_ind)

            grad_frame_offset = grad_frames_section + frame_ind * self.FRAME_VARS

            for pnt_ind, pnt_id in enumerate(self.bundle_pnt_ids):
                pnt_life =  self.points_life[pnt_id]
                pnt3D_world = self.world_pnts[pnt_life.track_id]
                assert not pnt3D_world is None

                self.__MarkOptVarsOrderDependency()

                out_frame_ind = grad_frame_offset
                if self.variable_intrinsics:
                    grad_intrinsic_vars = gradE[grad_frame_offset:grad_frame_offset+self.INTRINSICS_VARS]

                    out_ind = 0
                    if self.same_focal_length_xy:
                        grad_fk_finitdif = self.EstimateFirstPartialDerivFocalLengthFk(frame_ind, K)
                        grad_intrinsic_vars[out_ind] = grad_fk_finitdif
                        out_ind += 1
                    else:
                        grad_fxfy_finitdif = self.EstimateFirstPartialDerivFocalLengthFxFy(frame_ind, K, finitdiff_eps)
                        grad_intrinsic_vars[out_ind+0] = grad_fxfy_finitdif[0]
                        grad_intrinsic_vars[out_ind+1] = grad_fxfy_finitdif[1]
                        out_ind += 2


                    grad_u0v0_finitdif = self.EstimateFirstPartialDerivPrincipalPoint(frame_ind, K, finitdiff_eps)
                    grad_intrinsic_vars[out_ind + 0] = grad_u0v0_finitdif[0]
                    grad_intrinsic_vars[out_ind + 1] = grad_u0v0_finitdif[1]
                    out_ind += 2

                    if check_derivatives:
                        grad_intrinsic_vars_hint = gradE_hint[grad_frame_offset:grad_frame_offset+self.INTRINSICS_VARS]

                        print("gradK_hint={} gradK={}".format(grad_intrinsic_vars_hint, grad_intrinsic_vars))
                        close = self.IsClose(grad_intrinsic_vars_hint, grad_intrinsic_vars)
                        if not close:
                            assert False, "error computing derivative of intrinsics"

                out_frame_ind += self.INTRINSICS_VARS

                # finite difference of translation gradient
                gradT_finitdif = gradE[out_frame_ind:out_frame_ind+3] # count([Tx Ty Tz])=3
                for t_ind in range(0, 3):
                    gradT_finitdif[t_ind] = self.EstimateFirstPartialDerivTranslationDirect(frame_ind, cam_direct_orient_rt, t_ind, finitdiff_eps, self.elem_type)

                if check_derivatives:
                    gradT_hint = gradE_hint[out_frame_ind:out_frame_ind+3] # count([Tx Ty Tz])=3
                    print("gradT_hint={} gradT_finitdiff={}".format(gradT_hint, gradT_finitdif))

                    close = self.IsClose(gradT_hint, gradT_finitdif)
                    if not close:
                        assert False, "ERROR gradT computation error"

                out_frame_ind += 3 # count([Tx Ty Tz])=3

                # finite difference of rotation W components
                suc, w_direct_norm, w_ang = LogSO3New(R_direct)
                if suc:
                    w_direct = w_direct_norm * w_ang
                    gradW_finitdif = gradE[out_frame_ind:out_frame_ind+3] # count([Wx Wy Wz])=3
                    for w_ind in range(0, 3):
                        gradW_finitdif[w_ind] = self.EstimateFirstPartialDerivRotationNew(frame_ind, w_direct, T_direct, w_ind, finitdiff_eps, self.elem_type)

                    if check_derivatives:
                        gradW_hint = gradE_hint[out_frame_ind:out_frame_ind+3]
                        print("gradW_finitdif={} gradW_hint={}".format(gradW_finitdif, gradW_hint))

                        close = True or self.IsClose(gradW_finitdif, gradW_hint)
                        if not close:
                            assert False, "ERROR"  # TODO: crashes on gradW_estim=[ 13.47551411  18.0067005    0.66238898] gradW_exact=[ 12.37611123  18.81862305   0.44619587] gradW2_exact=[ 13.49019804  18.01214564   0.66229125]

                    # 2nd derivative Frame-Frame
                    print("frame_ind={} Frame-Frame".format(frame_ind))
                    deriv_second_frame_finitdif = deriv_second_frame[
                                                  frame_ind * FRAME_VARS:(frame_ind + 1) * FRAME_VARS, 0:FRAME_VARS]
                    frame_patch = FramePatch(self.elem_type)
                    frame_patch.SetK(frame_ind, K, self.same_focal_length_xy)
                    frame_patch.SetCamInverseOrientRT(frame_ind, cam_inverse_orient_rt)

                    frame_packed_vars = PackedFrameOptVars(FRAME_VARS, self.elem_type)
                    frame_patch.PackIntoOptVarsVector(frame_packed_vars)

                    for var1 in range(0, FRAME_VARS):
                        for var2 in range(0, FRAME_VARS):
                            deriv_second_frame_finitdif[var1, var2] = self.EstimateSecondPartialDerivFrame(frame_packed_vars, var1, var2, finitdiff_eps, self.elem_type)

                    if check_derivatives:
                        d2_frameframe_hint = deriv_second_frame_hint[
                                             frame_ind * FRAME_VARS:(frame_ind + 1) * FRAME_VARS, 0:FRAME_VARS]
                        print("deriv2nd_W_finitdif deriv2nd_W_hint\n{}\n{}".format(deriv_second_frame_finitdif,d2_frameframe_hint))

                        close = True or self.IsClose(deriv_second_frame_finitdif,d2_frameframe_hint)
                        if not close: # TODO: fails!!!!!!!!!!!!!!!!!
                            assert False, "ERROR"

                    # 2nd derivative Point-Frame
                    print("frame_ind={} Point-Frame".format(frame_ind))
                    deriv_second_pointframe_finitdif = deriv_second_pointframe[
                                                       pnt_ind * POINT_VARS:(pnt_ind + 1) * POINT_VARS,
                                                       frame_ind * FRAME_VARS:(frame_ind + 1) * FRAME_VARS]
                    for var1 in range(0, POINT_VARS):
                        for var2 in range(0, FRAME_VARS):
                            deriv_second_pointframe_finitdif[var1, var2] = self.EstimateSecondPartialDerivPointFrame(pnt_id, pnt3D_world, frame_packed_vars, var1, var2, finitdiff_eps)

                    if check_derivatives:
                        d2_pointframe_hint = deriv_second_pointframe_hint[
                                             pnt_ind * POINT_VARS:(pnt_ind + 1) * POINT_VARS,
                                             frame_ind * FRAME_VARS:(frame_ind + 1) * FRAME_VARS]
                        print("deriv2nd_PointFrame_finitdif deriv2nd_PointFrame_hint\n{}\n{}".format(deriv_second_pointframe_finitdif, d2_pointframe_hint))

                        close = self.IsClose(deriv_second_pointframe_finitdif, d2_pointframe_hint)
                        if not close:
                            assert False, "ERROR"

            pass # loop through points

        assert np.all(np.isfinite(affected_gradE)), "Failed to correctly set gradE"
        assert np.all(np.isfinite(affected_deriv_second_point)), "Failed to correctly set deriv_second_point"
        assert np.all(np.isfinite(affected_deriv_second_frame)), "Failed to correctly set deriv_second_frame"
        assert np.all(np.isfinite(affected_deriv_second_pointframe)), "Failed to correctly set deriv_second_pointframe"

    # formula 8, returns scalar or vector depending on gradp_byvar type
    def __OneSummandGradEByVar(self, f0, pqr, corner_pix, gradp_byvar, gradq_byvar, gradr_byvar):
        result = (pqr[0] / pqr[2] - corner_pix[0] / f0) * (pqr[2] * gradp_byvar - pqr[0] * gradr_byvar) + \
                 (pqr[1] / pqr[2] - corner_pix[1] / f0) * (pqr[2] * gradq_byvar - pqr[1] * gradr_byvar)
        result *= 2 / pqr[2] ** 2
        return result

    # formula 9
    def __SecondDerivFromPqrDerivative(self, pqr, deriv_pqr_by_var1, deriv_pqr_by_var2, var1, var2):
        s = (pqr[2] * deriv_pqr_by_var1[var1, 0] - pqr[0] * deriv_pqr_by_var1[var1, 2]) * (pqr[2] * deriv_pqr_by_var2[var2, 0] - pqr[0] * deriv_pqr_by_var2[var2, 2]) + \
            (pqr[2] * deriv_pqr_by_var1[var1, 1] - pqr[1] * deriv_pqr_by_var1[var1, 2]) * (pqr[2] * deriv_pqr_by_var2[var2, 1] - pqr[1] * deriv_pqr_by_var2[var2, 2])
        s *= 2 / pqr[2] ** 4
        return s

    def __ComputePointPqrDerivatives(self, P, point_pqr_deriv):
        point_pqr_deriv[0, 0:3] = P[0:3, 0]  # d(p,q,r)/dX
        point_pqr_deriv[1, 0:3] = P[0:3, 1]  # d(p,q,r)/dY
        point_pqr_deriv[2, 0:3] = P[0:3, 2]  # d(p,q,r)/dZ

    def __ComputeIntrinsicsDerivatives(self, frame_ind, pnt_ind, K, pqr, corner_pix, grad_pqr_by_dvar, grad_pqr_by_dvar_out_ind, gradE, gradE_out_ind):
        fx = K[0, 0]
        fy = K[1, 1]
        u0, v0, f0 = K[0:3, 2]

        out_ind = gradE_out_ind

        # focal length (fk) gradient if fx=fy=fk or
        # two focal length gradients for fxk and fyk if fx!=fy
        if self.same_focal_length_xy:
            fk = fx
            gradp_byf = (1 / fk) * (pqr[0] - u0 / f0 * pqr[2]) # scalar
            gradq_byf = (1 / fk) * (pqr[1] - v0 / f0 * pqr[2])
            gradr_byf = 0

            grad_pqr_by_dvar[grad_pqr_by_dvar_out_ind,0:3] = [gradp_byf, gradq_byf, gradr_byf]
            grad_pqr_by_dvar_out_ind += 1

            grad_fk_close_form = self.__OneSummandGradEByVar(f0, pqr, corner_pix, gradp_byf, gradq_byf, gradr_byf)  # scalar
            gradE[out_ind] += grad_fk_close_form
            out_ind += 1
        else:
            gradp_byfx = (1 / fx) * pqr[0] - u0 / (f0 * fx) * pqr[2]
            gradq_byfx = 0
            gradr_byfx = 0

            gradp_byfy = 0
            gradq_byfy = (1 / fy) * pqr[1] - v0 / (f0 * fy) * pqr[2]
            gradr_byfy = 0

            grad_focal_length_x = self.__OneSummandGradEByVar(f0, pqr, corner_pix, gradp_byfx, gradq_byfx, gradr_byfx)  # scalar
            grad_focal_length_y = self.__OneSummandGradEByVar(f0, pqr, corner_pix, gradp_byfy, gradq_byfy, gradr_byfy)  # scalar

            gradE[out_ind+0] += grad_focal_length_x
            gradE[out_ind+1] += grad_focal_length_y

            #if frame_ind == 0: print("xch-fxfy {} {} {} {} {}".format(pnt_ind, grad_focal_length_x, grad_focal_length_y, gradE[out_ind+0], gradE[out_ind+1]))

            out_ind += 2

            grad_pqr_by_dvar[grad_pqr_by_dvar_out_ind+0, 0:3] = [gradp_byfx, gradq_byfx, gradr_byfx]
            grad_pqr_by_dvar[grad_pqr_by_dvar_out_ind+1, 0:3] = [gradp_byfy, gradq_byfy, gradr_byfy]
            grad_pqr_by_dvar_out_ind += 2

        # principal point (u0,v0) gradient
        gradp_byu0 = (1 / f0) * pqr[2]
        gradq_byu0 = 0
        gradr_byu0 = 0

        gradp_byv0 = 0
        gradq_byv0 = (1 / f0) * pqr[2]
        gradr_byv0 = 0

        grad_u0 = self.__OneSummandGradEByVar(f0, pqr, corner_pix, gradp_byu0, gradq_byu0, gradr_byu0)  # scalar
        grad_v0 = self.__OneSummandGradEByVar(f0, pqr, corner_pix, gradp_byv0, gradq_byv0, gradr_byv0)  # scalar

        gradE[out_ind+0] += grad_u0
        gradE[out_ind+1] += grad_v0
        out_ind += 2

        grad_pqr_by_dvar[grad_pqr_by_dvar_out_ind + 0, 0:3] = [gradp_byu0, gradq_byu0, gradr_byu0]
        grad_pqr_by_dvar[grad_pqr_by_dvar_out_ind + 1, 0:3] = [gradp_byv0, gradq_byv0, gradr_byv0]
        grad_pqr_by_dvar_out_ind += 2

        return out_ind - gradE_out_ind

    # Estimate the first derivative of a world point [X Y Z] around the given component index (0=X,1=Y,2=Z)
    def EstimateFirstPartialDerivPoint(self, pnt_id, pnt0, xyz_ind, eps):
        pnt3D_left = pnt0.copy()
        pnt3D_left[xyz_ind] -= eps
        x1_err_sum = self.__ReprojError(pnt_id, pnt3D_left, None, None)

        pnt3D_right = pnt0.copy()
        pnt3D_right[xyz_ind] += eps
        x2_err_sum = self.__ReprojError(pnt_id, pnt3D_right, None, None)
        return (x2_err_sum - x1_err_sum) / (2 * eps)

    def EstimateFirstPartialDerivFocalLengthFk(self, frame_ind, K, eps):
        K_left = K.copy()
        K_left[0, 0] -= eps
        K_left[1, 1] -= eps
        err1 = self.__ReprojError(overwrite_frame_ind=frame_ind, overwrite_K=K_left)

        K_right = K.copy()
        K_right[0, 0] += eps
        K_right[1, 1] += eps
        err2 = self.__ReprojError(overwrite_frame_ind=frame_ind, overwrite_K=K_right)
        return (err2 - err1) / (2 * eps)

    def EstimateFirstPartialDerivFocalLengthFxFy(self, frame_ind, K, eps):
        grad_fxfy = np.zeros(2, dtype=self.elem_type)
        for i in range(0, 2):
            K_left = K.copy()
            K_left[i, i] -= eps
            err1 = self.__ReprojError(overwrite_frame_ind=frame_ind, overwrite_K=K_left)

            K_right = K.copy()
            K_right[i, i] += eps
            err2 = self.__ReprojError(overwrite_frame_ind=frame_ind, overwrite_K=K_right)
            grad_fxfy[i] = (err2 - err1) / (2 * eps)
        return grad_fxfy

    def EstimateFirstPartialDerivPrincipalPoint(self, frame_ind, K, eps):
        grad_u0v0 = np.zeros(2, dtype=self.elem_type)
        for i in range(0, 2):
            K_left = K.copy()
            K_left[i, 2] -= eps
            err1 = self.__ReprojError(overwrite_frame_ind=frame_ind, overwrite_K=K_left)

            K_right = K.copy()
            K_right[i, 2] += eps
            err2 = self.__ReprojError(overwrite_frame_ind=frame_ind, overwrite_K=K_right)
            grad_u0v0[i] = (err2 - err1) / (2 * eps)
        return grad_u0v0

    # TODO: remove
    def EstimateFirstPartialDerivTranslation(self, frame_ind, T_direct, R_direct, w_direct, t_ind, eps):
        R_direct_tmp = R_direct
        if R_direct_tmp is None:
            assert not w_direct is None
            suc, R_direct_tmp = RotMatFromAxisAngle(w_direct)
            assert suc

        t1dir = T_direct.copy()
        t1dir[t_ind] -= eps
        r1inv, t1inv = SE3Inv((R_direct_tmp, t1dir))
        t1_err_sum = self.__ReprojError(None, None, frame_ind, None, (r1inv, t1inv))

        t2dir = T_direct.copy()
        t2dir[t_ind] += eps
        r2inv, t2inv = SE3Inv((R_direct_tmp, t2dir))
        t2_err_sum = self.__ReprojError(None, None, frame_ind, None, (r2inv, t2inv))
        return (t2_err_sum - t1_err_sum) / (2 * eps)

    def EstimateFirstPartialDerivTranslationDirect(self, frame_ind, cam_direct_orient_rt, t_ind, eps, elem_type):
        """ The RT is in direct mode, because all close form derivatives calculated for direct RT-mode, not inverse RT mode. """
        t1 = cam_direct_orient_rt[1].copy()
        t1[t_ind] -= eps

        cam_inverse_orient_rt1 = SE3Inv((cam_direct_orient_rt[0], t1))
        frame_patch = FramePatch(elem_type, frame_ind)
        frame_patch.cam_inverse_orient_rt = cam_inverse_orient_rt1
        t1_err_sum = self.__ReprojErrorNew(frame_patch=frame_patch)

        t2 = cam_direct_orient_rt[1].copy()
        t2[t_ind] += eps

        cam_inverse_orient_rt2 = SE3Inv((cam_direct_orient_rt[0], t2))
        frame_patch.cam_inverse_orient_rt = cam_inverse_orient_rt2
        t2_err_sum = self.__ReprojErrorNew(frame_patch=frame_patch)
        return (t2_err_sum - t1_err_sum) / (2 * eps)

    # TODO: remove
    def EstimateFirstPartialDerivRotation(self, frame_ind, t_direct, w_direct, w_ind, eps):
        w1 = w_direct.copy()
        w1[w_ind] -= eps
        suc, R1_direct = RotMatFromAxisAngle(w1)
        assert suc
        R1inv, T1inv = SE3Inv((R1_direct, t_direct))
        R1_err_sum = self.__ReprojError(None, None, frame_ind, None, (R1inv, T1inv))

        w2 = w_direct.copy()
        w2[w_ind] += eps
        suc, R2_direct = RotMatFromAxisAngle(w2)
        assert suc
        R2inv, T2inv = SE3Inv((R2_direct, t_direct))
        R2_err_sum = self.__ReprojError(None, None, frame_ind, None, (R2inv, T2inv))
        return (R2_err_sum - R1_err_sum) / (2 * eps)

    def EstimateFirstPartialDerivRotationNew(self, frame_ind, w_direct, T_direct, w_ind, eps, elem_type):
        """ The RT is in direct mode, because all close form derivatives calculated for direct RT-mode, not inverse RT mode. """
        w1_direct = w_direct.copy()
        w1_direct[w_ind] -= eps

        suc, cam_direct_orient_R1 = RotMatFromAxisAngle(w1_direct)
        assert suc
        cam_inverse_orient_rt1 = SE3Inv((cam_direct_orient_R1, T_direct))
        frame_patch = FramePatch(elem_type, frame_ind, cam_inverse_orient_rt=cam_inverse_orient_rt1)
        R1_err_sum = self.__ReprojErrorNew(frame_patch=frame_patch)

        w2_direct = w_direct.copy()
        w2_direct[w_ind] += eps

        suc, cam_direct_orient_R2 = RotMatFromAxisAngle(w2_direct)
        assert suc
        cam_inverse_orient_rt2 = SE3Inv((cam_direct_orient_R2, T_direct))
        frame_patch.cam_inverse_orient_rt = cam_inverse_orient_rt2
        R2_err_sum = self.__ReprojErrorNew(frame_patch=frame_patch)
        return (R2_err_sum - R1_err_sum) / (2 * eps)

    def EstimateSecondPartialDerivPoint(self, point_patch: PointPatch, var1, var2, eps):
        patch_tmp = point_patch.Copy()
        patch_tmp.AddDelta(var1, +eps)
        patch_tmp.AddDelta(var2, +eps)
        f1 = self.__ReprojErrorNew(point_patch=patch_tmp)

        patch_tmp = point_patch.Copy()
        patch_tmp.AddDelta(var1, +eps)
        patch_tmp.AddDelta(var2, -eps)
        f2 = self.__ReprojErrorNew(point_patch=patch_tmp)

        patch_tmp = point_patch.Copy()
        patch_tmp.AddDelta(var1, -eps)
        patch_tmp.AddDelta(var2, +eps)
        f3 = self.__ReprojErrorNew(point_patch=patch_tmp)

        patch_tmp = point_patch.Copy()
        patch_tmp.AddDelta(var1, -eps)
        patch_tmp.AddDelta(var2, -eps)
        f4 = self.__ReprojErrorNew(point_patch=patch_tmp)

        # second order central finite difference formula
        # https://en.wikipedia.org/wiki/Finite_difference
        deriv2_value = (f1 - f2 - f3 + f4) / (4 * eps * eps)
        return deriv2_value

    def EstimateSecondPartialDerivFrame(self, frame_packed_vars:PackedFrameOptVars, var1, var2, eps, elem_type):
        frame_patch = FramePatch(elem_type)

        # var1+eps, var2+eps
        frame_packed_vars_tmp = frame_packed_vars.Copy()
        frame_packed_vars_tmp[var1] += eps
        frame_packed_vars_tmp[var2] += eps
        frame_patch.UnpackFromOptVarsVector(frame_packed_vars_tmp)
        f1 = self.__ReprojErrorNew(frame_patch=frame_patch)

        # var1+eps, var2-eps
        frame_packed_vars_tmp = frame_packed_vars.Copy()
        frame_packed_vars_tmp[var1] += eps
        frame_packed_vars_tmp[var2] -= eps
        frame_patch.UnpackFromOptVarsVector(frame_packed_vars_tmp)
        f2 = self.__ReprojErrorNew(frame_patch=frame_patch)

        # var1-eps, var2+eps
        frame_packed_vars_tmp = frame_packed_vars.Copy()
        frame_packed_vars_tmp[var1] -= eps
        frame_packed_vars_tmp[var2] += eps
        frame_patch.UnpackFromOptVarsVector(frame_packed_vars_tmp)
        f3 = self.__ReprojErrorNew(frame_patch=frame_patch)

        # var1-eps, var2-eps
        frame_packed_vars_tmp = frame_packed_vars.Copy()
        frame_packed_vars_tmp[var1] -= eps
        frame_packed_vars_tmp[var2] -= eps
        frame_patch.UnpackFromOptVarsVector(frame_packed_vars_tmp)
        f4 = self.__ReprojErrorNew(frame_patch=frame_patch)

        # second order central finite difference formula
        # https://en.wikipedia.org/wiki/Finite_difference
        deriv2_value = (f1 - f2 - f3 + f4) / (4 * eps * eps)
        return deriv2_value

    def EstimateSecondPartialDerivPointFrame(self, pnt_id, point_vars, packed_frame_vars:PackedFrameOptVars, var1, var2, eps):
        elem_type = type(point_vars[0])
        point_patch = PointPatch()
        frame_patch = FramePatch(elem_type)

        # construct neighbour points
        point_vars1 = point_vars.copy()
        point_vars1[var1] -= eps
        point_vars2 = point_vars.copy()
        point_vars2[var1] += eps

        frame_vars1 = packed_frame_vars.Copy()
        frame_vars1[var2] -= eps
        frame_vars2 = packed_frame_vars.Copy()
        frame_vars2[var2] += eps

        # var1+eps, var2+eps
        point_patch.SetPoint(pnt_id, point_vars2)
        frame_patch.UnpackFromOptVarsVector(frame_vars2)
        f1 = self.__ReprojErrorNew(point_patch=point_patch, frame_patch=frame_patch)

        # var1+eps, var2-eps
        point_patch.SetPoint(pnt_id, point_vars2)
        frame_patch.UnpackFromOptVarsVector(frame_vars1)
        f2 = self.__ReprojErrorNew(point_patch=point_patch, frame_patch=frame_patch)

        # var1-eps, var2+eps
        point_patch.SetPoint(pnt_id, point_vars1)
        frame_patch.UnpackFromOptVarsVector(frame_vars2)
        f3 = self.__ReprojErrorNew(point_patch=point_patch, frame_patch=frame_patch)

        # var1-eps, var2-eps
        point_patch.SetPoint(pnt_id, point_vars1)
        frame_patch.UnpackFromOptVarsVector(frame_vars1)
        f4 = self.__ReprojErrorNew(point_patch=point_patch, frame_patch=frame_patch)

        # second order central finite difference formula
        # https://en.wikipedia.org/wiki/Finite_difference
        deriv2_value = (f1 - f2 - f3 + f4) / (4 * eps * eps)
        return deriv2_value

    def IsClose(self, a, b):
        MAX_ABS_DIST = 0.1
        MAX_REL_DIST = 0.1  # 14327.78415796-14328.10677215=0.32261419 => rtol=2.2e-5
        return np.allclose(a, b, atol=MAX_ABS_DIST, rtol=MAX_REL_DIST)

    def __CheckPointDerivatives(self, gradPoint_close_form, deriv_second_point_close_form, pnt_id, pnt3D_world, eps):
        POINT_COMPS = self.POINT_VARS

        # estimate dX,dY,dZ
        gradPoint_finitdif = np.zeros(self.POINT_VARS, dtype=self.elem_type)
        for xyz_ind in range(0, POINT_COMPS):
            gradPoint_finitdif[xyz_ind] = self.EstimateFirstPartialDerivPoint(pnt_id, pnt3D_world, xyz_ind, eps)

        if self.debug >=3 : print("gradXYZ_finitdif={} gradXYZ_close={}".format(gradPoint_finitdif, gradPoint_close_form))

        close = self.IsClose(gradPoint_finitdif, gradPoint_close_form)
        if not close:
            assert False, "grad(Point) mismatch"

        # 2nd derivative Point-Point
        deriv_second_point_finitdif = np.zeros((POINT_COMPS,POINT_COMPS), dtype=self.elem_type)
        point_patch = PointPatch(pnt_id, pnt3D_world)
        for var1 in range(0,POINT_COMPS):
            for var2 in range(0,POINT_COMPS):
                deriv_second_point_finitdif[var1,var2] = self.EstimateSecondPartialDerivPoint(point_patch, var1, var2, eps)
        if self.debug >= 3: print("deriv2nd_estim deriv2nd_exact\n{}\n{}".format(deriv_second_point_finitdif, deriv_second_point_close_form))

        close = self.IsClose(deriv_second_point_finitdif, deriv_second_point_close_form)
        if not close:
            assert False, "D2 Point mismatch"

        check_point_hessian_is_invertible = True
        if check_point_hessian_is_invertible:
            point_hessian = deriv_second_point_close_form
            is_inverted = True
            try:
                point_hessian_inv = LA.inv(point_hessian)
            except LA.LinAlgError:
                if self.debug >= 3: print("ERROR: inverting 3x3 E, pnt_id={}".format(pnt_id))
                is_inverted = False
            assert is_inverted, "Can't invert point hessian for pnt_id={}".format(pnt_id)

    def __CheckFrameDerivatives(self, grad_frame_close_form, grad_frame_close_form_onlyW,
                                deriv_second_frame_close_form,
                                deriv_second_allpoints_frame_close_form,
                                frame_ind, K, T_direct, R_direct, eps):
        POINT_VARS = self.POINT_VARS
        FRAME_VARS = self.FRAME_VARS

        cam_direct_orient_rt = (R_direct, T_direct)
        cam_inverse_orient_rt = SE3Inv(cam_direct_orient_rt)

        self.__MarkOptVarsOrderDependency()

        in_ind = 0
        if self.variable_intrinsics:
            if self.same_focal_length_xy:
                grad_fk_close_form = grad_frame_close_form[in_ind]  # fk
                in_ind += 1

                grad_fk_finitdif = self.EstimateFirstPartialDerivFocalLengthFk(frame_ind, K, eps)
                if self.debug >= 3: print("gradfk_finitdif={} gradfk_close={}".format(grad_fk_finitdif, grad_fk_close_form))
                close = self.IsClose(grad_fk_finitdif, grad_fk_close_form)
                if not close:
                    assert False, "gradFk computation error"
            else:
                grad_fxfy_close_form = grad_frame_close_form[in_ind:in_ind + 2]  # fx fy
                in_ind += 2

                grad_fxfy_finitdif = self.EstimateFirstPartialDerivFocalLengthFxFy(frame_ind, K, eps)

                if self.debug >= 3: print("gradfxfy_finitdif={} gradfxfy_close={}".format(grad_fxfy_finitdif, grad_fxfy_close_form))
                close = self.IsClose(grad_fxfy_finitdif, grad_fxfy_close_form)
                if False and not close:
                    assert False, "gradFxFy computation error"

            grad_u0v0_close_form = grad_frame_close_form[in_ind:in_ind + 2]  # u0 v0
            in_ind += 2

            grad_u0v0_finitdif = self.EstimateFirstPartialDerivPrincipalPoint(frame_ind, K, eps)
            if self.debug >= 3: print("gradu0v0_finitdif={} gradK_close={}".format(grad_u0v0_finitdif, grad_u0v0_close_form))
            close = self.IsClose(grad_u0v0_finitdif, grad_u0v0_close_form)
            if not close:
                assert False, "gradU0V0 computation error"

        # finite difference of translation gradient
        gradT_close_form = grad_frame_close_form[in_ind:in_ind + 3]
        in_ind += 3

        gradT_finitdif = np.zeros(3, dtype=self.elem_type)
        for t_ind in range(0, 3):
            gradT_finitdif[t_ind] = self.EstimateFirstPartialDerivTranslationDirect(frame_ind, cam_direct_orient_rt, t_ind, eps, self.elem_type)

        if self.debug >= 3: print("gradT_finitdif={} gradT_close={}".format(gradT_finitdif, gradT_close_form))

        close = self.IsClose(gradT_finitdif, gradT_close_form)
        if False and not close:
            assert False, "ERROR gradT computation error"

        # finite difference of rotation W components
        suc, w_direct_norm, w_ang = LogSO3New(R_direct)
        if not suc:
            in_ind += 3
        else:
            w_direct = w_direct_norm * w_ang
            gradW_finitdif = np.zeros(3, dtype=self.elem_type)
            for w_ind in range(0, 3):
                gradW_finitdif[w_ind] = self.EstimateFirstPartialDerivRotationNew(frame_ind, w_direct, T_direct, w_ind, eps, self.elem_type)

            gradW_close_form = grad_frame_close_form[in_ind:in_ind + 3]
            gradW_close_form_onlyW = grad_frame_close_form_onlyW[in_ind:in_ind + 3]
            in_ind += 3

            if self.debug >= 3: print("gradW_finitdif={} gradW_close={} gradW2_close={}".format(gradW_finitdif, gradW_close_form, gradW_close_form_onlyW))

            close = self.IsClose(gradW_finitdif, gradW_close_form)
            if False and not close:
                assert False, "ERROR"  # TODO: crashes on gradW_estim=[ 13.47551411  18.0067005    0.66238898] gradW_exact=[ 12.37611123  18.81862305   0.44619587] gradW2_exact=[ 13.49019804  18.01214564   0.66229125]
            close = self.IsClose(gradW_finitdif, gradW_close_form_onlyW)
            if False and not close:
                assert False, "ERROR"

            # 2nd derivative Frame-Frame
            deriv_second_frame_finitdif = np.zeros((FRAME_VARS, FRAME_VARS), dtype=self.elem_type)
            frame_patch = FramePatch(self.elem_type)
            if self.variable_intrinsics:
                frame_patch.SetK(frame_ind, K, self.same_focal_length_xy)
            frame_patch.SetCamInverseOrientRT(frame_ind, cam_inverse_orient_rt)

            frame_packed_vars = PackedFrameOptVars(FRAME_VARS, self.elem_type)
            frame_patch.PackIntoOptVarsVector(frame_packed_vars)

            for var1 in range(0, FRAME_VARS):
                for var2 in range(0, FRAME_VARS):
                    deriv_second_frame_finitdif[var1, var2] = self.EstimateSecondPartialDerivFrame(frame_packed_vars, var1, var2, eps, self.elem_type)

            if self.debug >= 3: print("deriv2nd_RT_finitdif deriv2nd_RT_close\n{}\n{}".format(deriv_second_frame_finitdif,deriv_second_frame_close_form))

            close = self.IsClose(deriv_second_frame_finitdif, deriv_second_frame_close_form)
            if False and not close:
                assert False, "ERROR"

            # 2nd derivative Point-Frame
            deriv_second_pointframe_finitdif = np.zeros((POINT_VARS, FRAME_VARS), dtype=self.elem_type)
            for pnt_life in self.points_life:
                pnt_id = pnt_life.track_id
                pnt_ind = pnt_id
                pnt3D_world = self.world_pnts[pnt_ind]

                deriv_second_pointframe_finitdif.fill(0)
                for var1 in range(0, POINT_VARS):
                    for var2 in range(0, FRAME_VARS):
                        deriv_second_pointframe_finitdif[var1, var2] = self.EstimateSecondPartialDerivPointFrame(pnt_id, pnt3D_world, frame_packed_vars, var1, var2, eps)

                deriv_second_pointframe_close_form = deriv_second_allpoints_frame_close_form[pnt_ind*POINT_VARS:(pnt_ind+1)*POINT_VARS,0:FRAME_VARS]
                if self.debug >= 3:
                    print("deriv2nd_PointFrame_finitdif deriv2nd_PointFrame_close\n{}\n{}".format(deriv_second_pointframe_finitdif, deriv_second_pointframe_close_form))

                close = self.IsClose(deriv_second_pointframe_finitdif, deriv_second_pointframe_close_form)
                if False and not close:
                    assert False, "ERROR"

        assert len(grad_frame_close_form) == in_ind

    def __FillHessian(self, points_count, frames_count, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, hessian_factor, A):
        POINT_VARS = self.POINT_VARS
        FRAME_VARS = self.FRAME_VARS
        assert A.shape[0] == points_count*POINT_VARS + frames_count*FRAME_VARS
        assert A.shape[0] == A.shape[1]
        for pnt_ind in range(0, points_count):
            i = pnt_ind * POINT_VARS
            A[i:i+POINT_VARS, i:i+POINT_VARS] = deriv_second_point[i:i+POINT_VARS, 0:POINT_VARS]

            fp = deriv_second_pointframe[i:i+POINT_VARS, 0:frames_count * FRAME_VARS] # [3x6*frames_count]
            A[i:i+POINT_VARS, points_count * POINT_VARS:] = fp
            A[points_count * POINT_VARS:, i:i+POINT_VARS] = fp.T

        for frame_ind in range(0, frames_count):
            i = points_count * POINT_VARS + frame_ind * FRAME_VARS
            A[i:i + FRAME_VARS,i:i + FRAME_VARS] = deriv_second_frame[frame_ind * FRAME_VARS:(frame_ind + 1) * FRAME_VARS]

        # scale diagonal elements
        for i in range(0, A.shape[0]):
            A[i, i] *= (1 + hessian_factor)

    def __UpdateNormalizePattern(self):
        # R0=Identity, T0=[0,0,0], T1y=1
        self.__MarkOptVarsOrderDependency()

        norm_pattern = []
        off = 0
        # Frame0
        off += self.INTRINSICS_VARS # [(f | fx fy) u0 v0]
        norm_pattern.append(off + 0) # T0x
        norm_pattern.append(off + 1) # T0y
        norm_pattern.append(off + 2) # T0z
        norm_pattern.append(off + 3) # W0x
        norm_pattern.append(off + 4) # W0y
        norm_pattern.append(off + 5) # W0z
        off += 6

        # Frame1
        off += self.INTRINSICS_VARS  # [(f | fx fy) u0 v0]
        norm_pattern.append(off + self.unity_comp_ind) # T1y
        self.normalize_pattern = np.array(norm_pattern)

        self.normalize_pattern_frame0 = np.array([i for i in norm_pattern if i < self.FRAME_VARS])
        self.normalize_pattern_frame1 = np.array([i-self.FRAME_VARS for i in norm_pattern if self.FRAME_VARS <= i and i < 2*self.FRAME_VARS])

    def __EstimateCorrectionsNaive(self, points_count, frames_count, hessian_factor, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, corrections):
        POINT_VARS = self.POINT_VARS
        n = len(corrections)
        A = np.zeros((n, n), dtype=self.elem_type)
        self.__FillHessian(points_count, frames_count, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, hessian_factor, A)

        # remove rows/columns corresponding to normalized variables
        self.__MarkOptVarsOrderDependency()
        A_gaps = A.copy()
        A_gaps = np.delete(A_gaps, points_count*POINT_VARS+self.normalize_pattern, axis=1)  # delete normalized columns
        A_gaps = np.delete(A_gaps, points_count*POINT_VARS+self.normalize_pattern, axis=0)  # delete normalized rows
        b      = -gradE.copy()
        b_gaps = np.delete(b, points_count*POINT_VARS+self.normalize_pattern, axis=None)  # delete normalized columns

        corrects_naive_gaps = LA.solve(A_gaps, b_gaps)

        check = True
        if check:
            grads_gaps = A_gaps.dot(corrects_naive_gaps)
            diff = grads_gaps - b_gaps
            small_value = LA.norm(diff) # this value should be small and shows that corrections are right
            if not small_value < 1 and self.debug >= 3:
                print("warning: naive: should be small, small_value={} hessian_factor={}".format(small_value, hessian_factor))

        self.__FillCorrectionsPlainFromGaps(points_count, corrects_naive_gaps, corrections)

    def __EstimateCorrectionsDecomposedInTwoPhases(self, points_count, frames_count, hessian_factor, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, matG,
                                                   left_side1, right_side, corrections, corrections_reference=None):
        """ Computes updates for optimization variables"""
        POINT_VARS = self.POINT_VARS
        FRAME_VARS = self.FRAME_VARS

        self.__MarkOptVarsOrderDependency()

        if not corrections_reference is None:
            corrections_reference_gaps = np.delete(corrections_reference, points_count * POINT_VARS + self.normalize_pattern, axis=None)

        # convert 2nd derivatives Frame-Frame matrix into the square shape
        matG.fill(0)
        frame_offset = 0
        for frame_ind in range(0, frames_count):
            ax = deriv_second_frame[frame_ind * FRAME_VARS:(frame_ind + 1) * FRAME_VARS, 0:FRAME_VARS].copy() # [6x6]

            if frame_ind == 0: # skip T0=[0 0 0] R0=Identity
                ax = np.delete(ax, self.normalize_pattern_frame0, axis=1) # delete column
                ax = np.delete(ax, self.normalize_pattern_frame0, axis=0) # delete row

            elif frame_ind == 1: # skip T1x or T1y
                ax = np.delete(ax, self.normalize_pattern_frame1, axis=1) # delete column
                ax = np.delete(ax, self.normalize_pattern_frame1, axis=0) # delete row

            size = ax.shape[0]

            # scale diagonal elements
            for i in range(0, size):
                ax[i,i] *= (1 + hessian_factor)

            matG[frame_offset:frame_offset+size, frame_offset:frame_offset+size] = ax
            frame_offset += size

        left_side1.fill(0)
        right_side.fill(0)

        # calculate deltas for frame unknowns
        for pnt_ind in range(0, points_count):
            point_frame = deriv_second_pointframe[pnt_ind*POINT_VARS:(pnt_ind+1)*POINT_VARS,:] # 3 x 6*frames_count
            point_frame = np.delete(point_frame, self.normalize_pattern, axis=1) # delete normalized columns

            point_hessian = deriv_second_point[pnt_ind*POINT_VARS:(pnt_ind+1)*POINT_VARS,0:POINT_VARS] # 3x3

            # scale diagonal elements
            point_hessian_scaled = point_hessian.copy() # 3x3
            for i in range(0, point_hessian.shape[0]):
                point_hessian_scaled[i,i] *= (1 + hessian_factor)

            assert np.all(np.isfinite(point_hessian_scaled)), "Possibly to big hessian factor c={}".format(hessian_factor)

            try:
                point_hessian_inv = LA.inv(point_hessian_scaled)
            except LA.LinAlgError:
                assert False, "ERROR: inverting 3x3 E={}".format(point_hessian_scaled)

            gradeE_point = gradE[pnt_ind*POINT_VARS:(pnt_ind+1)*POINT_VARS] # 3x1

            # left side
            ax = point_frame.T.dot(point_hessian_inv).dot(point_frame) # 6*frames_count x 6*frames_count
            left_side1 += ax

            # right side
            ax = point_frame.T.dot(point_hessian_inv).dot(gradeE_point)
            right_side += ax


        left_side1 = matG - left_side1  # G-sum(F.E.F)
        frame_derivs_packed =  gradE[POINT_VARS*points_count:]
        frame_derivs_packed = np.delete(frame_derivs_packed, self.normalize_pattern, axis=None)
        right_side -= frame_derivs_packed # sum(F.E.gradE)-Df
        corects_frame = LA.solve(left_side1, right_side)

        if not corrections_reference is None:
            diff_corrections_frames = LA.norm(corects_frame - corrections_reference_gaps[points_count*POINT_VARS:])

        # calculate deltas for point unknowns
        corrects_points = np.zeros(points_count*POINT_VARS, dtype=self.elem_type)
        for pnt_ind in range(0, points_count):
            point_frame = deriv_second_pointframe[pnt_ind*POINT_VARS:(pnt_ind+1)*POINT_VARS,:] # 3 x 6*frames_count
            point_frame = np.delete(point_frame, self.normalize_pattern, axis=1)  # delete normalized columns

            gradeE_point = gradE[pnt_ind*POINT_VARS:(pnt_ind+1)*POINT_VARS] # 3x1

            partB = point_frame.dot(corects_frame) + gradeE_point

            point_hessian = deriv_second_point[pnt_ind * POINT_VARS:(pnt_ind + 1) * POINT_VARS, 0:POINT_VARS]  # 3x3

            # scale diagonal elements
            point_hessian_scaled = point_hessian.copy() # 3x3
            for i in range(0, point_hessian.shape[0]):
                point_hessian_scaled[i,i] *= (1 + hessian_factor)

            point_hessian_inv = LA.inv(point_hessian_scaled)

            deltas_one_point = -point_hessian_inv.dot(partB)
            corrects_points[pnt_ind * POINT_VARS:(pnt_ind + 1) * POINT_VARS] = deltas_one_point[0:POINT_VARS]

        if not corrections_reference is None:
            diff_corrections_points = LA.norm(corrects_points - corrections_reference_gaps[0:points_count * POINT_VARS])

        corrections_gaps = np.hstack((corrects_points, corects_frame))
        self.__FillCorrectionsPlainFromGaps(points_count, corrections_gaps, corrections)

        diff_corrections_all = None
        if not corrections_reference is None:
            diff_corrections_all = LA.norm(corrections - corrections_reference)
            assert diff_corrections_all < 1

        assert np.all(np.isfinite(corrections)), "Change of a variable must be a real number"

        check = True
        if check:
            n = len(corrections)
            A = np.zeros((n, n), dtype=self.elem_type)
            self.__FillHessian(points_count, frames_count, gradE, deriv_second_point, deriv_second_frame, deriv_second_pointframe, hessian_factor, A)

            # remove rows/columns corresponding to normalized variables
            A_gaps = A.copy()
            A_gaps = np.delete(A_gaps, points_count*POINT_VARS + self.normalize_pattern, axis=1)  # delete normalized columns
            A_gaps = np.delete(A_gaps, points_count*POINT_VARS + self.normalize_pattern, axis=0)  # delete normalized rows
            b_gaps = -gradE.copy()
            b_gaps = np.delete(b_gaps, points_count*POINT_VARS + self.normalize_pattern, axis=None)

            grads = A_gaps.dot(corrections_gaps)
            diff = grads - b_gaps
            small_value = LA.norm(diff)
            if not small_value < 1 and self.debug >= 3:
                print("warning: two-phases: should be small, small_value={} hessian_factor={}".format(small_value, hessian_factor))
                assert small_value < 1

    def __FillCorrectionsPlainFromGaps(self, points_count, corrections_gaps, corrections_plain):
        """ copy corrections with gaps to plain corrections """
        self.__MarkOptVarsOrderDependency()

        corrections_plain.fill(float('nan'))

        # some corrections are fixed during normalization, thus the gaps appear:
        # T0=[0 0 0] R0=Identity, T1y(or T1x)=1 determined by @unity_comp_ind
        # corrections plain: [point_corrections 0fx 0fy 0u0 0v0 0Tx 0Ty 0Tz 0Wx 0Wy 0Wz 1fx 1fy 1u0 1v0 1Tx 1Ty 1Tz 1Wx 1Wy 1Wz]
        # corrections wgaps: [point_corrections 0fx 0fy 0u0 0v0                         1fx 1fy 1u0 1v0 1Tx     1Tz 1Wx 1Wy 1Wz]

        in_ind  = 0
        out_ind = 0

        # Frame0
        # copy corrections for points
        corrections_plain[out_ind:out_ind+self.POINT_VARS * points_count] = corrections_gaps[in_ind:in_ind+self.POINT_VARS * points_count]
        in_ind  += self.POINT_VARS * points_count
        out_ind += self.POINT_VARS * points_count

        # [(f | fx fy) u0 v0]
        corrections_plain[out_ind:out_ind+self.INTRINSICS_VARS] = corrections_gaps[out_ind:out_ind+self.INTRINSICS_VARS]
        in_ind  += self.INTRINSICS_VARS
        out_ind += self.INTRINSICS_VARS

        # set zero corrections for frame0 (T0=[0,0,0] R0=Identity), which means 'no adjustments'
        no_adj = 0

        # zero W (axis in angle-axis representation) means identity rotation
        corrections_plain[out_ind:out_ind + 3 + 3] = no_adj  # [T0x T0y T0z Wx Wy Wz]
        out_ind += 3 # count([Tx Ty Tz])=3
        out_ind += 3 # count([Wx Wy Wz])=3

        # Frame1
        # [(f | fx fy) u0 v0]
        corrections_plain[out_ind:out_ind+self.INTRINSICS_VARS] = corrections_gaps[in_ind:in_ind+self.INTRINSICS_VARS]
        in_ind  += self.INTRINSICS_VARS
        out_ind += self.INTRINSICS_VARS

        # set zero correction for frame1 T1y=fixed_const
        corrections_plain[out_ind + self.unity_comp_ind] = no_adj  # T1x or T1y

        # copy other corrections of T1 intact
        corrections_plain[out_ind + (1 - self.unity_comp_ind)] = corrections_gaps[in_ind + 0]  # T1x or T1y
        corrections_plain[out_ind + 2]                         = corrections_gaps[in_ind + 1]  # T1z
        in_ind  += 2 # count(Tx Ty Tz without Tx or Ty)=2
        out_ind += 3 # count(Tx Ty Tz)=3

        # copy corrections for other frames intact
        corrections_plain[out_ind:] = corrections_gaps[in_ind:]
        assert np.all(np.isfinite(corrections_plain)), "Failed to copy normalized corrections"

        check_back = True
        if check_back:
            cs_back = corrections_plain.copy()
            cs_gaps_back = np.delete(cs_back, points_count*self.POINT_VARS + self.normalize_pattern, axis=None)
            diff = corrections_gaps - cs_gaps_back
            small_value = LA.norm(diff)
            assert np.isclose(0, small_value)

    def __ApplyCorrections(self, points_count, frames_count, corrections):
        for pnt_id in self.bundle_pnt_ids:
            pnt_ind = pnt_id
            deltaX = corrections[pnt_ind * self.POINT_VARS:(pnt_ind + 1) * self.POINT_VARS]
            if self.debug_processX:
                self.world_pnts[pnt_id] += deltaX

        self.__MarkOptVarsOrderDependency()

        deltaK = np.zeros((3,3), dtype=self.elem_type) # camera intrinsics
        for frame_ind in range(0, frames_count):
            rt_inversed = self.framei_from_world_RT_list[frame_ind]
            rt_direct = SE3Inv(rt_inversed)

            deltaF = corrections[points_count*self.POINT_VARS + frame_ind    *self.FRAME_VARS:
                                 points_count*self.POINT_VARS + (frame_ind+1)*self.FRAME_VARS]

            in_ind = 0

            if self.variable_intrinsics:
                deltaK_params = deltaF[in_ind:self.INTRINSICS_VARS]
                FillIntrinsics3x3(deltaK_params, self.same_focal_length_xy, deltaK)
                deltaK[2,2] = 0 # deltas for K[2,2] is 0, not 1

                K = self.CamMatPixelsFromMeter(frame_ind)
                K += deltaK
                in_ind += self.INTRINSICS_VARS

            deltaT = deltaF[in_ind:in_ind+3] # count([Tx Ty Tz])=3
            if self.debug_processT:
                T_direct_new = rt_direct[1] + deltaT
            else:
                T_direct_new = rt_direct[1]
            in_ind += 3

            deltaW = deltaF[in_ind:in_ind+3] # count([Wx Wy Wz])=3
            all_zero = np.all(np.abs(deltaW) < 1e-5)
            if all_zero or not self.debug_processR:
                R_direct_new = rt_direct[0]
            else:
                suc, rot_delta = RotMatFromAxisAngle(deltaW)
                assert suc
                R_direct_new = rot_delta.dot(rt_direct[0])
            in_ind += 3

            rt_inversed_new = SE3Inv((R_direct_new, T_direct_new))
            self.framei_from_world_RT_list[frame_ind] = rt_inversed_new

    def __ReprojError(self, overwrite_track_id = None, overwrite_x3D=None, overwrite_frame_ind=None, overwrite_K=None, overwrite_rt=None):
        if not overwrite_K is None or not overwrite_rt is None:
            assert not overwrite_frame_ind is None
        if not overwrite_x3D is None:
            assert not overwrite_track_id is None

        err_sum = BundleAdjustmentKanataniReprojError(self.points_life, self.world_pnts, self.framei_from_world_RT_list, self.bundle_pnt_ids,
                                                      overwrite_track_id, overwrite_x3D,
                                                      overwrite_frame_ind, overwrite_K, overwrite_rt,
                                                      cam_mat_pixel_from_meter=self.cam_mat_pixel_from_meter,
                                                      cam_mat_pixel_from_meter_list=self.cam_mat_pixel_from_meter_list)
        return err_sum

    def __ReprojErrorNew(self, point_patch: PointPatch = None, frame_patch: FramePatch = None):
        overwrite_pnt_id = None
        if not point_patch is None:
            overwrite_pnt_id = point_patch.pnt_id

        overwrite_pnt3D_world = None
        if not point_patch is None:
            overwrite_pnt3D_world = point_patch.pnt3D_world

        overwrite_frame_ind = None
        if not frame_patch is None:
            overwrite_frame_ind = frame_patch.frame_ind

        overwrite_K = None
        if not frame_patch is None:
            overwrite_K = frame_patch.K

        overwrite_cam_inverse_orient_rt = None
        if not frame_patch is None:
            overwrite_cam_inverse_orient_rt = frame_patch.cam_inverse_orient_rt

        if not overwrite_K is None or not overwrite_cam_inverse_orient_rt is None:
            assert not overwrite_frame_ind is None
        if not overwrite_pnt3D_world is None:
            assert not overwrite_pnt_id is None

        err_sum = BundleAdjustmentKanataniReprojError(self.points_life, self.world_pnts, self.framei_from_world_RT_list, self.bundle_pnt_ids,
                                                      overwrite_pnt_id, overwrite_pnt3D_world,
                                                      overwrite_frame_ind, overwrite_K, overwrite_cam_inverse_orient_rt,
                                                      cam_mat_pixel_from_meter=self.cam_mat_pixel_from_meter,
                                                      cam_mat_pixel_from_meter_list=self.cam_mat_pixel_from_meter_list)
        return err_sum

    def ReprojErrorTmp(self, overwrite_track_id = None, overwrite_x3D=None, overwrite_frame_ind=None, overwrite_rt=None):
        return self.__ReprojError(overwrite_track_id, overwrite_x3D, overwrite_frame_ind, overwrite_rt)

    def ErrDecreaseRatio(self):
        return self.err_value / self.err_value_initial

    def CamMatPixelsFromMeter(self, frame_ind):
        if self.variable_intrinsics:
            assert not self.cam_mat_pixel_from_meter_list is None
            return self.cam_mat_pixel_from_meter_list[frame_ind]
        else:
            assert not self.cam_mat_pixel_from_meter is None
            return self.cam_mat_pixel_from_meter

    # Designate the code, which depends on ordering of optimization variables [[(f | fx fy) u0 v0] Tx Ty Tz Wx Wy Wz]
    def __MarkOptVarsOrderDependency(self): pass


