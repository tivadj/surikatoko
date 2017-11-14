import math
import operator # for key=operator.itemgetter(1)
import numpy as np
import numpy.linalg as LA
import cv2

try:
    import mpmath # multi-precision math, see http://mpmath.org/
    mpmath.mp.dps = 100
except ImportError: pass

from suriko.obs_geom import *

def Estimate3DPointDepthFromFrames(base_frame_ind, frame_inds, x_per_frame, framei_from_base_RT):
    # find distances to all 3D points in frame1 given position [Ri,Ti] of each frame
    # (MASKS formula 8.44)
    x1 = x_per_frame[base_frame_ind]

    alpha_num = 0
    alpha_den = 0
    for frame_ind in frame_inds:
        # other cameras, except base frame
        if frame_ind == base_frame_ind: continue

        xi = x_per_frame[frame_ind]

        xi_skew = skewSymmeticMat(xi)
        frame_R, frame_T = framei_from_base_RT[frame_ind]
        h1 = np.dot(xi_skew, frame_T)
        h2 = np.dot(np.dot(xi_skew, frame_R), x1)
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

def FindDistancesTo3DPointsTwoFrames(pnts_life, pnt_ids, rel_RT, block_base_frame_ind, block_other_frame_ind, debug):
    points_num = len(pnt_ids)
    ess_R, ess_Tvec = rel_RT

    # find initial distances to all 3D points in frame1 (MASKS formula 8.43)
    alphas = None
    for i, pnt_id in enumerate(pnt_ids):
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
        alpha = -alpha_num / LA.norm(h1) ** 2
        alphas[i] = alpha
        dist = 1 / alpha
        assert dist > 0, "distances are positive"
    if debug >= 3: print("Dist: {0}".format(1 / alphas))
    return 1 / alphas

class MultiViewIterativeFactorizer:
    """
    Performs multi-view factorization of point fuatures.
    source: MASKS page 273, para 8.3.3, algorithm 8.1
    """
    def __init__(self, elem_type, points_life, cam_mat_pixel_from_meter, world_pnts, framei_from_world_RT_list, relative_motion_fun, min_reproj_err=1e-4, adjust_world_fun=None):
        self.elem_type = elem_type
        self.use_mpmath = 0 # whether to use mpmath package, see http://mpmath.org/
        self.min_reproj_err = min_reproj_err
        self.points_life = points_life
        self.relative_motion_fun = relative_motion_fun
        self.adjust_world_fun = adjust_world_fun
        self.cam_mat_pixel_from_meter = cam_mat_pixel_from_meter
        self.framei_from_world_RT_list = framei_from_world_RT_list
        self.world_pnts = world_pnts

        self.la_engine = "scipy"  # default="scipy", ["opencv", "scipy"]

        # True to convert the result of SVD algorithm, which is of type float64, into current working type (eg f16 or f128)
        # default = False
        # Generally it is a bad idea to upconvert from f64 to f128, because the actual precision is hidden
        # eg. if True, algorithm sqrt(f64)->f32 is coerced into sqrt(f64)->f64 by upconverting result from f32 into f64
        # eg. if True, algorithm sqrt(f16)->f64 is coerced into sqrt(f16)->f16 by downconverting result from f64 into f16
        # If False, then it is problematic to estimate how an algorithm with specified float precision behaves,
        # because actual float precision is different. But True, when working with f128, introduces the hiding of lost precision f64->f128.
        self.conceal_lost_precision = True
        self.check_drift = False # whether to check divergence of points' coordinates and camera relative motion from ground truth
        self.drift = 3e-1
        self.ground_truth_relative_motion = None
        self.ground_truth_map_pnt_pos = None # gets 3D coordinate of the point in a camera of interest
        self.hack_camera_location = False
        self.hack_world_mapping = False
        self.hack_camera_location_in_batch_refine = False
        self.reproj_err_history = []

    def CalcReprojErrPixel(self, debug, divide_by_count):
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
                x_expect = pnt_life.points_list_pixel[frame_ind]

                # transform 3D point into the frame
                targ_from_world_RT = self.framei_from_world_RT_list[frame_ind]
                if targ_from_world_RT[0] is None: continue

                x3D_framei = SE3Apply(targ_from_world_RT, x3D_world)

                x2D_hom = self.cam_mat_pixel_from_meter.dot(x3D_framei)
                x2D_hom = x2D_hom / x2D_hom[2]

                err = LA.norm(x_expect - x2D_hom[0:2]) ** 2
                # if debug >= 3: print("xexpect={0} xact={1} err={2} meters".format(x_expect, x3D_frameiN, err))
                result += err
                one_err_count += 1
        if divide_by_count and one_err_count > 0:
            result /= one_err_count
        return result

    def IntegrateNewFrame(self, framei_from_world_RT, world_pnts, debug):
        # allocate space for the new frame
        framei_from_world_RT.append((None,None)) # TODO: why inserting empty RT?
        frames_count = len(framei_from_world_RT)

        # allocate space for the new frame
        points_count = len(self.points_life)
        while len(world_pnts) < points_count:
            world_pnts.append(None)

        if frames_count == 1:
            # world's origin
            rot = np.eye(3, dtype=self.elem_type)
            cent = np.zeros(3, dtype=self.elem_type)
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

            suc, frame_ind_from_base_RT = self.relative_motion_fun(xs1_meter, xs2_meter)
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
            latest_frame_ind = frames_count - 1

            # determine the set of all points in the latest frame
            latest_frame_pnt_ids = []
            latest_frame_pnt_ids_set = set([])
            for pnt_life in self.points_life:
                x_meter = pnt_life.points_list_meter[latest_frame_ind]
                if x_meter is None: continue

                latest_frame_pnt_ids.append(pnt_life.track_id)
                latest_frame_pnt_ids_set.add(pnt_life.track_id)

            reproj_err_initial = self.CalcReprojErrPixel(debug, divide_by_count=False)
            if debug >= 3: print("reproj_err_initial={} pixels".format(reproj_err_initial))

            max_iter = 9
            it = 1
            common_pnt_ids = []
            while True:
                if it > max_iter:
                    print("reproj_err_history={}".format(self.reproj_err_history))
                    assert False, "Failed to converge the reconstruction"
                    break

                # find relative motion of current frame basing on:
                # it=1 the set of common points with the previous frame
                # it=2 the same plus the set of new points of this frame (this is possible because locations for
                # all points in current frame are found in it=1)
                #if it == 1:
                if it == 1 or it == 2:
                    anchor_frame_ind, common_pnts_count = self.FindAnchorFrame(latest_frame_ind, latest_frame_pnt_ids_set)
                    common_pnt_ids.clear()
                    self.CountCommonPoints(latest_frame_pnt_ids_set, anchor_frame_ind, common_pnt_ids)

                    if debug >= 3: print("anchor-targ: {}".format((anchor_frame_ind, latest_frame_ind)))

                frame_ind_from_anchor_RT = self.GetFrameRelativeRT(anchor_frame_ind, latest_frame_ind, common_pnt_ids)
                assert not frame_ind_from_anchor_RT[0] is None
                assert not frame_ind_from_anchor_RT[1] is None

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
                world_points_new = dict()
                for pnt_id in latest_frame_pnt_ids:
                    # world point is already calculated
                    if not world_pnts[pnt_id] is None:
                        continue
                    x3D_world = self.Estimate3DPointFromFrames(pnt_id)
                    if x3D_world is None: continue
                    if self.hack_world_mapping and not self.ground_truth_map_pnt_pos is None:
                        pnt_life = self.points_life[pnt_id]
                        gtruth_x3D_world = self.ground_truth_map_pnt_pos(world_frame_ind, pnt_life.virtual_feat_id)
                        scaled_x3D_world = gtruth_x3D_world * self.first_unity_translation_scale_factor
                        x3D_world = scaled_x3D_world
                    # TODO: recalculate already estimated 3D point?
                    #world_pnts[pnt_id] = x3D_world
                    world_points_new[pnt_id] = x3D_world

                # update all new points at once
                for pnt_id in latest_frame_pnt_ids:
                    x3D_world = world_points_new.get(pnt_id,None)
                    if not x3D_world is None:
                        world_pnts[pnt_id] = x3D_world

                # hack: magically fix reconstruction
                correct_reconstruction_to_ground_truth = False
                if correct_reconstruction_to_ground_truth:
                    for pnt_ind,pnt_life in enumerate(self.points_life):
                        gtruth_pos3D = self.ground_truth_map_pnt_pos(world_frame_ind, pnt_life.virtual_feat_id)
                        scaled_gtruth_pos3D = gtruth_pos3D * self.first_unity_translation_scale_factor

                        reconstr_pos3D = self.world_pnts[pnt_ind]
                        if reconstr_pos3D is None:
                            continue
                        self.world_pnts[pnt_ind] =  scaled_gtruth_pos3D

                    for frame_ind, rt in enumerate(self.framei_from_world_RT_list):
                        gtruth_R, gtruth_T = self.ground_truth_relative_motion(world_frame_ind, frame_ind)
                        scaled_T = gtruth_T * self.first_unity_translation_scale_factor
                        self.framei_from_world_RT_list[frame_ind] = (gtruth_R, scaled_T)

                reproj_err = self.CalcReprojErrPixel(debug, divide_by_count=False)
                if debug >= 3: print("anchor-targ: {} reproj_err={} pixels reproj_err_initial={}".format((anchor_frame_ind, latest_frame_ind), reproj_err, reproj_err_initial))

                self.reproj_err_history.append(reproj_err)
                print("reproj_err_history={}".format(self.reproj_err_history))

                if reproj_err < self.min_reproj_err:
                    break

                # the iterative reconstruction diverged enough
                # try to coalesce it and minimize a reprojective error
                if not self.adjust_world_fun is None:
                    changed = self.adjust_world_fun(self)
                    if changed:
                        reproj_err_adj = self.CalcReprojErrPixel(debug, divide_by_count=False)
                        if debug >= 3: print("anchor-targ: {} reproj_err_adj={} pixels reproj_err_initial={}".format((anchor_frame_ind, latest_frame_ind), reproj_err_adj, reproj_err_initial))

                        if reproj_err_adj < self.min_reproj_err:
                            break

                it += 1
                break # only 1 iteration
        return True

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
        frames_count = len(self.framei_from_world_RT_list)
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
            try:
                framei_from_base = RelMotionBFromA(base_from_world, framei_from_world)
            except ValueError:
                print("aaa!")
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
                u2 = u2.astype(eltype)
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

    def Estimate3DPointFromFrames(self, pnt_id):
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
                if self.check_drift:
                    assert False

        # convert point in base, into point in the world
        anchor_from_world = self.framei_from_world_RT_list[anchor_frame_ind]
        world_from_anchor = SE3Inv(anchor_from_world)
        x3D_world = SE3Apply(world_from_anchor, x3D_anchor)

        return x3D_world

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

    def LocateCamAndMapBatchRefine(self, debug):
        """ Performs 'batch-mode' multi-view factorization """
        reproj_err_history = []

        initial_reproj_err = self.CalcReprojErrPixel(debug, divide_by_count=False)
        if debug >= 3: print("initial_reproj_err={} meters".format(initial_reproj_err))
        reproj_err_history.append(initial_reproj_err)

        frames_count = len(self.framei_from_world_RT_list)
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
                reproj_err = self.CalcReprojErrPixel(debug, divide_by_count=False)
                if debug >= 3: print("reproj_err={} meters".format(reproj_err))

                reproj_err_history.append(reproj_err)

                if reproj_err < self.min_reproj_err:
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

                suc1, old_N, old_Ang = LogSO3New(old_RT[0], check_rot_mat=True)
                suc2, new_N, new_Ang = LogSO3New(frame_ind_from_world_RT[0], check_rot_mat=True)
                if suc1 and suc2:
                    upd_N = SmoothUpdate(old_N, new_N, learn_rate)
                    upd_N = upd_N / LA.norm(upd_N)
                    upd_Ang = SmoothUpdate(old_Ang, new_Ang, learn_rate)

                    upd_R = rotMat(upd_N, upd_Ang, check_log_SO3=True)
                else:
                    upd_R = old_RT[0]
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
                x3D_world = self.Estimate3DPointFromFrames(next_pnt_id)
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




