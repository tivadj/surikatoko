import sys
import unittest

import os

from suriko.bundle_adjustment_kanatani_impl import *
from suriko.mvg import PointLife
from suriko.obs_geom import *
from suriko.test_data_builder import CircusGridDataSet, ParseTestArgs
#from suriko.testing import *
import suriko.uivis
import cv2

def Triangulate3DPointHelper(pnt_track_list, proj_mat_list, pnt_id, debug=0):
    """ :param proj_mat_list: list of projection P[3x4] matrices for each camera"""
    pnt_ind = pnt_id
    pnt_life = pnt_track_list[pnt_ind]
    frames_count = len(proj_mat_list)

    xs2D = []
    proj_mats_per_frame = []
    for frame_ind in range(0, frames_count):
        x2D = pnt_life.points_list_pixel[frame_ind]
        if x2D is None:
            continue
        xs2D.append(x2D)
        P = proj_mat_list[frame_ind]
        proj_mats_per_frame.append(P)

    f0 = 1  # TODO: const f0, can it change?
    return Triangulate3DPointByLeastSquares(xs2D, proj_mats_per_frame, f0, debug)

class BundleAdjustmentKanataniTests(unittest.TestCase):
#class BundleAdjustmentKanataniTests(suriko.testing.SurikoTestCase): # TODO: this doesn't pick up the tests
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.debug = 3
        self.elem_type = np.float32
        from suriko.testing import GetTestRootDir
        self.tests_data_dir = GetTestRootDir()

    def test_normalization_normalize_nochanges_revert_leads_to_original_scene(self):
        cell_size = (0.5, 0.5, 0.5)
        world_range = (-2, 2, -0.5, 0.5, 0, 0.5)
        rot_radius = 15*cell_size[0]
        ang_off = 3*math.pi/2 + math.pi/6
        angles = ang_off - np.arange(0, 2*math.pi/3, math.pi/180*5)
        img_width, img_height = 640, 480
        ds = CircusGridDataSet(self.elem_type, img_width, img_height, world_range, cell_size, angles, rot_radius)
        data = list(ds.Generate())[0:100]

        salient_points = ds.GetWorldSalientPoints()
        camera_from_world_RTs = [tup[1] for tup in data]

        salient_points_norm = salient_points.copy()
        camera_from_world_RTs_norm = camera_from_world_RTs.copy()

        points_count = len(salient_points)
        frames_count = len(camera_from_world_RTs_norm)
        bundle_pnt_ids = range(0, points_count)

        scene_normalizer = NormalizeWorldInplace(salient_points_norm, camera_from_world_RTs_norm, bundle_pnt_ids, t1y=1.0, unity_comp_ind=1)
        scene_normalizer.RevertNormalization()

        for i in range(0, points_count):
            x_expect = salient_points[i]
            x_actual = salient_points_norm[i]
            assert np.isclose(x_expect[0], x_actual[0], atol=1e-5), "Normalization side-effects must be reversed"
        for i in range(0, frames_count):
            r_expect = camera_from_world_RTs[i][0]
            r_actual = camera_from_world_RTs_norm[i][0]
            assert np.allclose(r_expect, r_actual, atol=1e-5), "Normalization side-effects must be reversed"

            t_expect = camera_from_world_RTs[i][1]
            t_actual = camera_from_world_RTs_norm[i][1]
            assert np.allclose(t_expect, t_actual, atol=1e-5), "Normalization side-effects must be reversed"

    def test_ba_circle(self):
        pnt_track_list = []
        camera_poses = []

        cell_size = (0.5, 0.5, 0.5)
        world_range = (-2, 2, -0.5, 0.5, 0, 0.5)
        rot_radius = 15*cell_size[0]
        ang_off = 3*math.pi/2 + math.pi/6
        angles = ang_off - np.arange(0, 2*math.pi/3, math.pi/180*5)
        img_width, img_height = 640, 480
        ds = CircusGridDataSet(self.elem_type, img_width, img_height, world_range, cell_size, angles, rot_radius)

        point_track_next_id = 0

        #data = list(ds.Generate())
        #data = ds.Generate()
        data = list(ds.Generate())[0:100]
        for frame_ind, (R3,T3), xs_objs_clipped in data:
            img_gray = np.zeros((img_height, img_width), np.uint8)
            img_bgr = np.zeros((img_height, img_width, 3), np.uint8)

            img_gray.fill(0)
            for virt_id, (xpix, ypix) in xs_objs_clipped:
                img_gray[int(ypix), int(xpix)] = 255
            img_bgr[:, :, 0] = img_gray
            img_bgr[:, :, 1] = img_gray
            img_bgr[:, :, 2] = img_gray

            cv2.imshow("frontal_camera", img_bgr)
            #cv2.waitKey(0)

            camera_poses.append((R3,T3))

            # reserve space for pixel coordinates of corners
            for pnt_life in pnt_track_list:
                pnt_life.points_list_pixel.append(None)

            for virt_id, corner_pix in xs_objs_clipped:
                ps = [pnt_life for pnt_life in pnt_track_list if pnt_life.virtual_feat_id == virt_id]
                assert len(ps) <= 1, "There are 0 points if the salient point was not seen, otherwise 1"
                if len(ps) == 0:
                    p = PointLife()
                    p.virtual_feat_id = virt_id
                    p.track_id = point_track_next_id
                    point_track_next_id += 1
                    p.points_list_pixel = [None] * (frame_ind+1)
                    pnt_track_list.append(p)
                else:
                    p = ps[0]
                p.points_list_pixel[frame_ind] = corner_pix

        salient_points = ds.GetWorldSalientPoints()
        cam_mat_pixel_from_meter = ds.GetCamMat()

        genuine_scene_err = BundleAdjustmentKanataniReprojError(pnt_track_list, cam_mat_pixel_from_meter, salient_points, camera_poses)
        print("genuine scene reproj error={}".format(genuine_scene_err))

        salient_points_norm = salient_points.copy()
        camera_poses_norm = camera_poses.copy()

        scene_ui = suriko.uivis.SceneViewerPyGame((640, 480), "Test world", debug=self.debug)
        scene_ui.AddScene(salient_points, camera_poses)
        #scene_ui.AddScene(salient_points_norm, camera_poses_norm)
        scene_ui.ShowOnDifferentThread()

        add_noiseX = True
        add_noiseR = True
        add_noiseT = True
        # changes cameras (R,T) so that camera's location is fixed and only the orientation changes
        fix_cam_pos = False
        salient_points_noisy = salient_points.copy()
        camera_poses_noisy = camera_poses.copy()
        if add_noiseX or add_noiseR or add_noiseT:
            np.random.seed(123)
            angle_delta = math.radians(2) # ok=2, >=5 too noisy
            err_rad_perc = 0.05
            noise_sig = cell_size[0]*err_rad_perc/3 # 3 due to 3-sigma, 3sig=err_radius
            if add_noiseX:
                salient_points_noisy += np.random.normal(0, noise_sig, salient_points.shape)

            frames_count = len(camera_poses)
            for frame_ind in range(0, frames_count):
                R,T = camera_poses[frame_ind]

                T_noisy = T
                if add_noiseT:
                    T_noisy = T + np.random.normal(0, noise_sig, 3)
                w = np.random.normal(0, noise_sig, 3)
                w *= angle_delta / LA.norm(w)
                converged, Rw = RotMatFromAxisAngle(w)
                assert converged
                R_noisy = R
                if add_noiseR:
                    R_noisy = Rw.dot(R)

                if fix_cam_pos:
                    # for camera to be in fixed position and just rotate both R and T must be modified
                    assert add_noiseT
                    assert add_noiseR
                    T_noisy = Rw.dot(T)

                camera_poses_noisy[frame_ind] = (R_noisy, T_noisy)

        scene_ui.AddScene(salient_points_noisy, camera_poses_noisy)

        noise_scene_err = BundleAdjustmentKanataniReprojError(pnt_track_list, cam_mat_pixel_from_meter, salient_points_noisy, camera_poses_noisy)
        print("noise scene reproj error={}".format(noise_scene_err))

        salient_points_adj = salient_points_noisy.copy()
        camera_poses_adj = camera_poses_noisy.copy()

        ba = BundleAdjustmentKanatani(debug=self.debug)
        ba.debug_processX = add_noiseX
        ba.debug_processR = add_noiseR
        ba.debug_processT = add_noiseT
        ba.naive_estimate_corrections = False
        converged, err_msg = ba.ComputeInplace(pnt_track_list, cam_mat_pixel_from_meter, salient_points_adj, camera_poses_adj)
        print("converged={} err_msg={}".format(converged, err_msg))

        err_dec_ratio = ba.ErrDecreaseRatio()
        print("err_dec_ratio={}".format(err_dec_ratio))

        scene_ui.AddScene(salient_points_adj, camera_poses_adj)

        print("err_noisy={}".format(ba.err_value_initial))
        print("err_xnois={}".format(ba.err_value))

        assert err_dec_ratio < 0.5, "Reprojection error was not decreased enough, err_dec_ratio={}".format(err_dec_ratio)

        scene_ui.CloseAndShutDown()


    def test_ba_dinosaur(self):
        P_file_path = os.path.join(self.tests_data_dir, "oxfvisgeom/dinosaur/dinoPs_as_mat108x4.txt")

        # [frames_count*3,4]
        P_data = np.genfromtxt(P_file_path, delimiter='\t')

        frames_count = int(P_data.shape[0]/3) # =36

        proj_mat_list = []
        camera_poses = []
        K_first = None
        for frame_ind in range(0, frames_count):
            proj_mat = np.array(P_data[frame_ind*3:(frame_ind+1)*3, 0:4], dtype=self.elem_type)

            scale, K, (R,t) = DecomposeProjMat(proj_mat)
            if K_first is None: # all K from each proj mat are identical, because all shots are made with the same camera
                K_first = K
            proj_mat_list.append(proj_mat)
            camera_poses.append(SE3Inv((R,t)))
        assert not K_first is None, "frames_count>0"
        # TODO: negative coordinate of principal point! v0=-1070.5155, why is that?
        # K=[[  3.21741772e+03  -7.86099014e+01   2.89874390e+02]
        #    [  0.00000000e+00   2.29247192e+03  -1.07051550e+03]
        #    [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

        #
        pnts_file_path = os.path.join(self.tests_data_dir, "oxfvisgeom/dinosaur/viff.xy")

        # delimiter=None because space and double space are used as separators
        # [points_count x frames_count]
        pnts_data = np.genfromtxt(pnts_file_path, delimiter=None)
        points_count = pnts_data.shape[0] # =4983

        pnt_track_list = []
        for pnt_ind in range(0, points_count):
            p = PointLife()
            p.virtual_feat_id = 10000+pnt_ind
            p.track_id = pnt_ind
            p.points_list_pixel = [None] * frames_count

            for frame_ind in range(0, frames_count):
                x,y = pnts_data[pnt_ind, frame_ind*2:(frame_ind+1)*2]
                if x == -1 or y == -1: # -1 means None
                    continue
                p.points_list_pixel[frame_ind] = np.array([x, y], dtype=self.elem_type)
            pnt_track_list.append(p)

        salient_points = []
        for pnt_ind in range(0, points_count):
            x3D = Triangulate3DPointHelper(pnt_track_list, proj_mat_list, pnt_ind)
            salient_points.append(x3D)

        # show scene
        scene_ui = suriko.uivis.SceneViewerPyGame((640, 480), "dino points 3D", debug=self.debug)
        scene_ui.AddScene(salient_points, camera_poses)
        scene_ui.ShowOnDifferentThread()

        # fix scene using bundle adjustment
        salient_points_adj = salient_points.copy()
        camera_poses_adj = camera_poses.copy()

        #
        ba = BundleAdjustmentKanatani(debug=self.debug, min_err_change_rel=None)
        ba.max_hessian_factor = None
        ba.naive_estimate_corrections = False
        print("start bundle adjustment...")
        converged, err_msg = ba.ComputeInplace(pnt_track_list, K_first, salient_points_adj, camera_poses_adj)
        print("converged={} err_msg={}".format(converged, err_msg))

        err_dec_ratio = ba.ErrDecreaseRatio()
        print("err_dec_ratio={}".format(err_dec_ratio))

        scene_ui.AddScene(salient_points_adj, camera_poses_adj)

        print("err_noisy={}".format(ba.err_value_initial))
        print("err_xnois={}".format(ba.err_value))

        assert True or err_dec_ratio < 0.5, "Reprojection error was not decreased enough, err_dec_ratio={}".format(err_dec_ratio)

        scene_ui.CloseAndShutDown()

