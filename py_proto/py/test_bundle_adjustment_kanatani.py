import numpy as np
import unittest
import py.la_utils
from py.bundle_adjustment_kanatani_impl import BundleAdjustmentKanatani
from py.mvg import PointLife
from py.obs_geom import *
from py.test_data_builder import CircusGridDataSet
import cv2

class BundleAdjustmentKanataniTests(unittest.TestCase):
    def test1(self):
        img_width, img_height = 640, 480

        pnt_track_list = []
        camera_frames = []

        ds = CircusGridDataSet(np.float32, img_width, img_height)

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

        add_noise = True
        salient_points_noisy = salient_points.copy()
        camera_frames_noisy = camera_frames.copy()
        if add_noise:
            np.random.seed(221)
            noise_sig = 0.01
            salient_points_noisy += np.random.normal(0, noise_sig, salient_points.shape)

            frames_count = len(camera_frames)
            for frame_ind in range(0, frames_count):
                R,T = camera_frames[frame_ind]

                T_noisy = T + np.random.normal(0, noise_sig, 3)
                w = np.random.normal(0, noise_sig, 3)
                suc, Rw = RotMatFromAxisAngle(w)
                assert suc
                R_noisy = Rw.dot(R)
                camera_frames_noisy[frame_ind] = (R_noisy, T_noisy)

        salient_points_adj = salient_points_noisy.copy()
        camera_frames_adj = camera_frames_noisy.copy()

        ba = BundleAdjustmentKanatani()
        improved, err_change_ratio = ba.BundleAdjustmentComputeInplace(pnt_track_list, cam_mat_pixel_from_meter, salient_points_adj, camera_frames_adj)
        print("err_change_ratio={}".format(err_change_ratio))

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
        print("dist_bef={}".format(dist_bef))
        print("dist_aft={}".format(dist_aft))
        print()
