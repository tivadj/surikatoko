import math
import numpy as np
import numpy.linalg as LA

from py.obs_geom import *

class CrystallGridDataSet:
    def __init__(self, el_type, img_width, img_height, provide_ground_truth = True):
        self.el_type = el_type
        self.img_width = img_width
        self.img_height = img_height
        self.provide_ground_truth = provide_ground_truth
        self.xs3D = []
        self.xs3D_virtual_ids = []
        self.ground_truth_R_per_frame = []
        self.ground_truth_T_per_frame = []
        self.debug = None
        self.cam_mat_pixel_from_meter = None
        self.cam_mat_changed = None # event that fires when the camera matrix is estimated

    def Generate(self):
        cx, cy, cz = 0.3, 0.3, 0.05  # cell size between atoms of the crystal
        Wx0, Wx, Wy0, Wy = 0.0, 5.0, -5.0, 0.0  # world size
        move_steps_count = 20
        move_step = min(Wx - Wx0, Wy - Wy0) / (
        move_steps_count)  # world is crossed in at least N steps, 0.1 to jit coords if step=cell size
        dist_to_central_point = 3 * max(cx, cy, cz)
        num_visible_2dpoints = 6 * 6 * 2
        do_wobble = True  # alters linear movement with rotation to prevent degenerate cases in 3D reconstruction
        wobble_freq = 1 / (move_steps_count / 7)
        wobble_amplitude = math.pi / 18  # max deviation of the wobble angle
        trample_on_the_spot = False
        trample_steps = 4  # max number of steps in one direction (after this number the direction is reversed)

        # create world's salient points
        next_virt_id = 10001
        inclusive_gap = 1.0e-8  # small value to make iteration inclusive
        for z in np.arange(0, cz + inclusive_gap, cz):
            for x in np.arange(Wx0, Wx + inclusive_gap, cx):
                # xjit = cz/10 # offset x to prevent overlapping of trajectories
                xjit = 0  # offset x to prevent overlapping of trajectories
                for y in np.arange(Wy0, Wy + inclusive_gap, cy):
                    xtmp = x + xjit  # offset changes for each change of Y
                    xjit = -xjit
                    pnt = np.array([xtmp, y, z], self.el_type)
                    self.xs3D.append(pnt)
                    self.xs3D_virtual_ids.append(next_virt_id)
                    next_virt_id += 1
                    # with cameras_lock: self.xs3d.append(pnt)
        self.xs3D = np.array(self.xs3D)

        pnt_ids = None
        cam_mat_pixel_from_meter = None
        R2 = None
        T2 = None
        cell_width = None
        period_right = True
        period_val = -1
        pad = 0  # eg: 0..2
        road = [(Wx0 + pad, Wy - pad),  # bot-right=start
                (Wx0 + pad, Wy0 + pad),  # bot-left
                (Wx - pad, Wy0 + pad),  # top-left
                (Wx - pad, Wy - pad)]  # top-right
        # add move_step to upper bound to make inclusive
        centralY_list_left = list(np.arange(road[0][1], road[1][1], -move_step))
        centralX_list_left = [road[0][0]] * len(centralY_list_left)
        centralX_list_up = list(np.arange(road[1][0], road[2][0], move_step))
        centralY_list_up = [road[1][1]] * len(centralX_list_up)
        centralY_list_right = list(np.arange(road[2][1], road[3][1], move_step))
        centralX_list_right = [road[2][0]] * len(centralY_list_right)
        centralX_list_down = list(np.arange(road[3][0], road[0][0], -move_step))
        centralY_list_down = [road[3][1]] * len(centralX_list_down)
        centralX_list = centralX_list_left + centralX_list_up + centralX_list_right + centralX_list_down
        centralY_list = centralY_list_left + centralY_list_up + centralY_list_right + centralY_list_down
        cam_poses_xy = list(enumerate(zip(centralX_list, centralY_list)))
        for i, (centralX, centralY) in cam_poses_xy:
            # for centralX, centralY in [(centralX_list[0],centralY_list[0]),(centralX_list[0],centralY_list[0])]:

            if trample_on_the_spot:
                if period_right:
                    period_val += 1
                    if period_val == trample_steps:
                        period_val = trample_steps - 2
                        period_right = False
                else:
                    period_val -= 1
                    if period_val == -1:
                        period_val = -1 + 2
                        period_right = True
                actual_index = period_val
                _, (centralX, centralY) = cam_poses_xy[actual_index]

            print("central-XY=({},{})".format(centralX, centralY))

            # cam3
            cam3_from_world = np.eye(4, 4, dtype=self.el_type)
            # centralX,centralY = cx,-cy
            cam3_from_world = SE3Mat(None, np.array([-centralX, -centralY, 0]), dtype=self.el_type).dot(
                cam3_from_world)  # stay on atom which will be in the center of the view
            # (handX,handY) = the distance to central atom in (X,Y) plane
            handX = dist_to_central_point / math.sqrt(3)
            handY = handX
            cam3_from_world = SE3Mat(None, np.array([handX, -handY, 0]), dtype=self.el_type).dot(
                cam3_from_world)  # offset in (X,Y) plane
            handZ = handX  # the altitude above the (X,Y) plane
            cam3_from_world = SE3Mat(None, np.array([0, 0, handZ]), dtype=self.el_type).dot(
                cam3_from_world)  # offset in (X,Y) plane
            # point OZ in the direction of the central atom
            cam3_from_world = SE3Mat(rotMat([0, 1, 0], -math.pi / 2), None, dtype=self.el_type).dot(cam3_from_world)
            wobble_ang = 0
            if do_wobble:
                wobble_ang = math.sin(i * 2 * math.pi * wobble_freq) * wobble_amplitude
            cam3_from_world = SE3Mat(rotMat([1, 0, 0], -math.pi / 4 - wobble_ang), None, dtype=self.el_type).dot(cam3_from_world)
            cam3_from_world = SE3Mat(rotMat([0, 1, 0], math.radians(75)), None, dtype=self.el_type).dot(
                cam3_from_world)  # rotate down OZ towards the central point
            cam3_from_world = SE3Mat(rotMat([0, 0, 1], -math.pi / 2), None, dtype=self.el_type).dot(
                cam3_from_world)  # align axis in image (column,row) format, OX=right, OY=down
            R3 = cam3_from_world[0:3, 0:3].astype(self.el_type)
            T3 = cam3_from_world[0:3, 3].astype(self.el_type)

            if self.provide_ground_truth:
                self.ground_truth_R_per_frame.append(R3)
                self.ground_truth_T_per_frame.append(T3)

            xs3D_cam3 = np.dot(R3, self.xs3D.T).T + T3

            corrupt_with_noise = False
            if corrupt_with_noise:
                cell_width = max(cx, cy, cz)
                noise_perc = 0.01
                proj_err_pix = noise_perc * cell_width  # 'radius' of an error
                print("proj_err_pix={0}".format(proj_err_pix))
                n3 = np.random.rand(len(self.xs3D), 3) * 2 * proj_err_pix - proj_err_pix
                xs3D_cam3 += n3

            # perform general projection 3D->2D
            xs_img3 = xs3D_cam3.copy()
            for i in range(0, len(xs_img3)):
                xs_img3[i, :] /= xs_img3[i, -1]

            # set pixels formation matrix, so that specified number of projected 3D points is visible
            if cam_mat_pixel_from_meter is None:
                # example of pixel_from_meter camera matrix
                cam_mat_pixel_from_meter = np.array([
                    [880, 0, self.img_width / 2],
                    [0, 660, self.img_height / 2],
                    [0., 0., 1.]], self.el_type)

                # project all 3D points in the image and look at closest N points
                # the maximum of (X,Y,Z) will determine the alphaX=focus_dist*sx
                p1_cam3 = cam3_from_world.dot([centralX, centralY, 0, 1])
                p1_cam3 = p1_cam3[0:3]
                dists = [LA.norm(p - p1_cam3) for p in xs_img3]
                closest_pnts = sorted(zip(xs3D_cam3, xs_img3, dists), key=lambda item: item[2])
                assert len(closest_pnts) > 0, "Camera must observe at least one point"
                far_point_ind = num_visible_2dpoints - 1
                if far_point_ind >= len(closest_pnts):
                    far_point_ind = len(closest_pnts) - 1
                far_point_meter = closest_pnts[far_point_ind][0]
                max_rad_meter = max(abs(far_point_meter[0]), abs(far_point_meter[1]))
                max_z = abs(far_point_meter[2])

                # x_image_meter = focus_dist*X/Z, MASKS formula 3.4
                # x_image_pixel = x_image_meter * sx
                # => x_image_pixel = focus_dist*sx*X/Z
                # let alphaX = focus_dist*sx = x_image_pixel/X*Z
                alphaX = (self.img_width / 2) / max_rad_meter * max_z
                alphaY = (self.img_height / 2) / max_rad_meter * max_z

                # imageX (columns) is directed in the direction of OY of camera
                # imageY (rows) is directed in the direction of -OX of camera
                # xcol = x*alphaX+xcenter
                # yrow = y*alphaY+ycenter
                # where (xcenter,ycenter) is the principal point (the center) of the image in pixels
                cam_mat_pixel_from_meter = np.array([
                    [alphaX, 0, self.img_width / 2],
                    [0, alphaY, self.img_height / 2],
                    [0.0, 0, 1]], self.el_type)
                print("cam_mat_pixel_from_meter=\n{}".format(cam_mat_pixel_from_meter))
                if not self.cam_mat_changed is None:
                    self.cam_mat_pixel_from_meter = cam_mat_pixel_from_meter
                    self.cam_mat_changed(cam_mat_pixel_from_meter)

            xs_pixel_all = cam_mat_pixel_from_meter.dot(xs_img3.T).T
            xs_objs_clipped = [(virt_id, (xpix, ypix)) for (virt_id, (xpix, ypix, w)) in zip(self.xs3D_virtual_ids, xs_pixel_all) if xpix < self.img_width and xpix >= 0 and ypix < self.img_height and ypix >= 0]
            frame_ind = i
            yield frame_ind, (R3,T3), xs_objs_clipped
        pass

    # returns [R,T], such that X2=[R,T]*X1
    def GroundTruthRelativeMotion(self, img_ind1, img_ind2):
        # ri from world
        r1_fromW = self.ground_truth_R_per_frame[img_ind1]
        t1_fromW = self.ground_truth_T_per_frame[img_ind1]
        r2_fromW = self.ground_truth_R_per_frame[img_ind2]
        t2_fromW = self.ground_truth_T_per_frame[img_ind2]

        # X1=M_1w*Xw, X2=M_2w*Xw => X2=M_2w*inv(M_1w)*X1
        r2_from1 = r2_fromW.dot(r1_fromW.T)
        t2_from1 = -r2_from1.dot(t1_fromW) + t2_fromW
        return (r2_from1, t2_from1)

    def GroundTruthMapPointPos(self, img_ind, map_point_id):
        pos_world = None
        for virt_id, pos in zip(self.xs3D_virtual_ids, self.xs3D):
            if virt_id == map_point_id:
                pos_world = pos
                break
        if not pos_world is None:
            # Xcam = M_camw*Xw
            cam_from_world_R = self.ground_truth_R_per_frame[img_ind]
            cam_from_world_T = self.ground_truth_T_per_frame[img_ind]
            pos_cam = SE3Apply((cam_from_world_R, cam_from_world_T), pos_world)
            return pos_cam

        return None

    def CamMatChanged(self, on_computed_cam_mat_fun):
        self.cam_mat_changed = on_computed_cam_mat_fun

class CircusGridDataSet:
    def __init__(self, el_type, img_width, img_height, provide_ground_truth=True):
        self.el_type = el_type
        self.img_width = img_width
        self.img_height = img_height
        self.provide_ground_truth = provide_ground_truth
        self.xs3D = []
        self.xs3D_virtual_ids = []
        self.ground_truth_R_per_frame = []
        self.ground_truth_T_per_frame = []
        self.debug = None
        self.cam_mat_pixel_from_meter = None
        self.cam_mat_changed = None  # event that fires when the camera matrix is estimated
        self.salient_points_created = None # event that fires when the world's salient 3D points are created

    def Generate(self):
        cx, cy, cz = 1, 0.5, 0.05  # cell size between atoms of the crystal
        Wx0, Wx, Wy0, Wy = -2, 2, -0.5, 0.5  # world size

        # create world's salient points
        next_virt_id = 10001
        inclusive_gap = 1.0e-8  # small value to make iteration inclusive
        for z in np.arange(0, cz + inclusive_gap, cz):
            for x in np.arange(Wx0, Wx + inclusive_gap, cx):
                for y in np.arange(Wy0, Wy + inclusive_gap, cy):
                    # x plus small offset to avoid centering on stable point
                    pnt = np.array([x+0.2, y, z], self.el_type)
                    self.xs3D.append(pnt)
                    self.xs3D_virtual_ids.append(next_virt_id)
                    next_virt_id += 1
                    # with cameras_lock: self.xs3d.append(pnt)
        self.xs3D = np.array(self.xs3D)
        if not self.salient_points_created is None:
            self.salient_points_created(self.xs3D)

        frame_ind = 0
        # add move_step to upper bound to make inclusive
        rot_angles = list(np.arange(0, 2*math.pi, math.pi/180))
        for ang in rot_angles:
            # cam3
            cam3_from_world = np.eye(4, 4, dtype=self.el_type)
            # angle=0 corresponds to OX (to the right) axis
            # -ang to move clockwise
            #abs_ang = 3*math.pi/2 + math.pi/6 - ang
            abs_ang = math.pi/2 + math.pi/6 - ang
            shiftX = 5*cx*math.cos(abs_ang)
            shiftY = 5*cx*math.sin(abs_ang)
            shiftZ = 5*cx
            cam3_from_world = SE3Mat(None, np.array([-shiftX, -shiftY, -shiftZ]), dtype=self.el_type).dot(cam3_from_world)

            # move OY towards direction 'towards center'
            toCenterXOY = [-shiftX, -shiftY, 0] # the direction towards center O
            oy = [0, 1, 0]
            ang_yawOY = np.sign(np.cross(oy, toCenterXOY).dot([0,0,1])) * math.acos(np.dot(oy, toCenterXOY) / (LA.norm(oy)*LA.norm(toCenterXOY)))
            cam3_from_world = SE3Mat(rotMat([0, 0, 1], -ang_yawOY), None, dtype=self.el_type).dot(cam3_from_world)

            # look down towards the center
            look_down_ang = math.atan2(shiftZ, LA.norm([shiftX, shiftY]))
            cam3_from_world = SE3Mat(rotMat([1, 0, 0], look_down_ang - math.pi/2), None, dtype=self.el_type).dot(cam3_from_world)
            R3 = cam3_from_world[0:3, 0:3].astype(self.el_type)
            T3 = cam3_from_world[0:3, 3].astype(self.el_type)

            if self.provide_ground_truth:
                self.ground_truth_R_per_frame.append(R3)
                self.ground_truth_T_per_frame.append(T3)

            xs3D_cam3 = np.dot(R3, self.xs3D.T).T + T3

            corrupt_with_noise = False
            if corrupt_with_noise:
                cell_width = max(cx, cy, cz)
                noise_perc = 0.01
                proj_err_pix = noise_perc * cell_width  # 'radius' of an error
                print("proj_err_pix={0}".format(proj_err_pix))
                n3 = np.random.rand(len(self.xs3D), 3) * 2 * proj_err_pix - proj_err_pix
                xs3D_cam3 += n3

            # perform general projection 3D->2D
            xs_img3 = xs3D_cam3.copy()
            for i in range(0, len(xs_img3)):
                xs_img3[i, :] /= xs_img3[i, -1]

            # set pixels formation matrix, so that specified number of projected 3D points is visible
            if self.cam_mat_pixel_from_meter is None:
                # example of pixel_from_meter camera matrix
                self.cam_mat_pixel_from_meter = np.array([
                    [880, 0, self.img_width / 2],
                    [0, 660, self.img_height / 2],
                    [0., 0., 1.]], self.el_type)
                if not self.cam_mat_changed is None:
                    self.cam_mat_changed(self.cam_mat_pixel_from_meter)

            xs_pixel_all = self.cam_mat_pixel_from_meter.dot(xs_img3.T).T
            xs_objs_clipped = [(virt_id, (xpix, ypix)) for (virt_id, (xpix, ypix, w)) in
                               zip(self.xs3D_virtual_ids, xs_pixel_all) if
                               xpix < self.img_width and xpix >= 0 and ypix < self.img_height and ypix >= 0]
            yield frame_ind, (R3, T3), xs_objs_clipped
            frame_ind += 1
        pass

    # returns [R,T], such that X2=[R,T]*X1
    def GroundTruthRelativeMotion(self, img_ind1, img_ind2):
        # ri from world
        r1_fromW = self.ground_truth_R_per_frame[img_ind1]
        t1_fromW = self.ground_truth_T_per_frame[img_ind1]
        r2_fromW = self.ground_truth_R_per_frame[img_ind2]
        t2_fromW = self.ground_truth_T_per_frame[img_ind2]

        # X1=M_1w*Xw, X2=M_2w*Xw => X2=M_2w*inv(M_1w)*X1
        r2_from1 = r2_fromW.dot(r1_fromW.T)
        t2_from1 = -r2_from1.dot(t1_fromW) + t2_fromW
        return (r2_from1, t2_from1)

    def GroundTruthMapPointPos(self, img_ind, map_point_id):
        pos_world = None
        for virt_id, pos in zip(self.xs3D_virtual_ids, self.xs3D):
            if virt_id == map_point_id:
                pos_world = pos
                break
        if not pos_world is None:
            # Xcam = M_camw*Xw
            cam_from_world_R = self.ground_truth_R_per_frame[img_ind]
            cam_from_world_T = self.ground_truth_T_per_frame[img_ind]
            pos_cam = SE3Apply((cam_from_world_R, cam_from_world_T), pos_world)
            return pos_cam

        return None

    def CamMatChanged(self, on_computed_cam_mat_fun):
        self.cam_mat_changed = on_computed_cam_mat_fun

    def GetWorldSalientPoints(self):
        return self.xs3D

    def GetCamMat(self):
        return self.cam_mat_pixel_from_meter