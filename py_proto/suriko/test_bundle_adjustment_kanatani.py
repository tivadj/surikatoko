import numpy as np
import unittest
from suriko.bundle_adjustment_kanatani_impl import BundleAdjustmentKanatani
from suriko.mvg import PointLife
from suriko.obs_geom import *
from suriko.test_data_builder import CircusGridDataSet, ParseTestArgs
import cv2

class BundleAdjustmentKanataniTests(unittest.TestCase):
    def test1(self):
        #args = ParseTestArgs()
        debug = 3
        elem_type = np.float32
        img_width, img_height = 640, 480

        pnt_track_list = []
        camera_frames = []

        cell_size = (0.5, 0.5, 0.5)
        world_range = (-2, 2, -0.5, 0.5, 0, 0.5)
        rot_radius = 15*cell_size[0]
        ds = CircusGridDataSet(elem_type, img_width, img_height, world_range, cell_size, rot_radius)

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

            camera_frames.append((R3,T3))

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

        add_noiseX = False
        add_noiseR = True
        add_noiseT = False
        salient_points_noisy = salient_points.copy()
        camera_frames_noisy = camera_frames.copy()
        if add_noiseX or add_noiseR or add_noiseT:
            np.random.seed(221)
            err_rad_perc = 0.01
            noise_sig = cell_size[0]*err_rad_perc/3 # 3 due to 3-sigma, 3sig=err_radius
            if add_noiseX:
                salient_points_noisy += np.random.normal(0, noise_sig, salient_points.shape)

            frames_count = len(camera_frames)
            for frame_ind in range(0, frames_count):
                R,T = camera_frames[frame_ind]

                T_noisy = T
                if add_noiseT:
                    T_noisy = T + np.random.normal(0, noise_sig, 3)
                w = np.random.normal(0, noise_sig, 3)
                suc, Rw = RotMatFromAxisAngle(w)
                assert suc
                R_noisy = R
                if add_noiseR:
                    R_noisy = Rw.dot(R)

                camera_frames_noisy[frame_ind] = (R_noisy, T_noisy)

        salient_points_adj = salient_points_noisy.copy()
        camera_frames_adj = camera_frames_noisy.copy()

        ba = BundleAdjustmentKanatani(debug=debug)
        ba.max_hessian_factor = None
        ba.processX = add_noiseX
        ba.processR = add_noiseR
        ba.processT = add_noiseT
        ba.naive_estimate_corrections = False
        suc, err_msg = ba.ComputeInplace(pnt_track_list, cam_mat_pixel_from_meter, salient_points_adj, camera_frames_adj)
        print("suc={} err_msg={} err_change_ratio={}".format(suc, err_msg, ba.ErrChangeRatio()))

        dist_points_bef = LA.norm(salient_points_noisy - salient_points)
        dist_points_aft = LA.norm(salient_points_adj - salient_points)

        rs,ts = list(unzip(camera_frames))
        rs, ts = np.array(rs),np.array(ts)
        rs_noisy,ts_noisy = list(unzip(camera_frames_noisy))
        rs_noisy, ts_noisy = np.array(rs_noisy),np.array(ts_noisy)
        rs_adj,ts_adj = list(unzip(camera_frames_adj))
        rs_adj, ts_adj = np.array(rs_adj), np.array(ts_adj)

        dist_r_bef = LA.norm(rs_noisy - rs)
        dist_r_aft = LA.norm(rs_adj - rs)
        dist_t_bef = LA.norm(ts_noisy - ts)
        dist_t_aft = LA.norm(ts_adj - ts)
        dist_bef = dist_points_bef + dist_r_bef + dist_t_bef
        dist_aft = dist_points_aft + dist_r_aft + dist_t_aft
        print("dist_noisy={}".format(dist_bef))
        print("dist_xnois={}".format(dist_aft))
        print("err_noisy={}".format(ba.err_value_initial))
        print("err_xnois={}".format(ba.err_value))
        assert suc
