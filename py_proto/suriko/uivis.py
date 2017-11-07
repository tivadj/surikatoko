import time
import threading
import numpy as np
import numpy.linalg as LA
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame

class SceneViewerPyGame:
    def __init__(self, win_size, caption, debug = 0):
        self.win_size = win_size
        self.caption = caption
        self.debug = debug

        # viewport
        self.orthoRadius = None
        self.scene_scale = None
        self.rgbcolor_per_scene = np.float32([[1, 0, 0], [0, 0, 1], [0,0,0]])
        self.scenes_count = 0
        self.scenes_visibile = [] # flag per scene, True to indicate a scene is visible
        self.draw_cameras = True
        self.draw_salient_points = True

        # world
        self.eye = None # position of the observer TODO: rename obs_pos, obs_center, obs_up
        self.center = None # obs_view_target
        self.up = None
        self.xs3d = [] # 3D structure (mapping in SLAM)
        self.list_of_xs3d = [] # 3D structure (mapping in SLAM)
        self.list_of_camera_from_world_RTs = []  # position and orientation of camera for each image (localization in SLAM)

        self.data_lock = threading.Lock()
        self.processed_images_counter = 0 # used to calculate how quickly ui redraws
        self.world_map_changed_flag = False # used to invalidate the viewport
        self.ui_thread = None
        self.do_computation_flag = True

    def SetScene(self, salient_points, camera_poses):
        self.list_of_xs3d = [salient_points]
        self.list_of_camera_from_world_RTs = [camera_poses]
        self.scenes_count = 1
        self.scenes_visibile = [True]
        self.Invalidate()

    def AddScene(self, salient_points, camera_poses):
        self.list_of_xs3d.append(salient_points)
        self.list_of_camera_from_world_RTs.append(camera_poses)
        self.scenes_visibile.append(True)
        self.scenes_count += 1
        self.Invalidate()

    def __SetupOnThisThread(self):
        glutInit()

        width, height = self.win_size

        """ Setup window and pygame environment. """
        pygame.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(self.caption)
        # pygame.key.set_repeat(10, 10)  # allow multiple keyDown events per single key press
        pygame.key.set_repeat(5, 1)  # allow multiple keyDown events per single key press

        gap = 0
        glViewport(gap, gap, width - 2 * gap, height - 2 * gap)

        glPointSize(3)

        # glEnable(GL_LIGHTING)
        # glDisable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        # glCullFace(GL_BACK)
        glCullFace(GL_FRONT_AND_BACK)
        # glEnable(GL_COLOR_MATERIAL)
        # glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        # glEnable(GL_NORMALIZE)
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

        if len(self.list_of_xs3d) > 0 and len(self.list_of_xs3d[0]) > 0:
            one_scene_points = self.list_of_xs3d[0]
            cent = np.average(one_scene_points, axis=0)
        else:
            cent = np.array([0, 0, 0])
        if self.debug >= 3: print("model average={0}".format(cent))
        self.eye = np.array([0, 0, -2], dtype=np.float32)
        self.center = cent
        self.up = np.array([0, -1, 0], dtype=np.float32)
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], self.center[0], self.center[1], self.center[2], self.up[0],
                  self.up[1], self.up[2])

    def Invalidate(self):
        with self.data_lock:  # notify model is changed, redraw is required
            self.world_map_changed_flag = True

    def Show(self):
        self.__SetupOnThisThread()
        self.__PyGameLoop()
        pygame.quit()

    def ShowOnDifferentThread(self):
        assert self.ui_thread is None
        self.ui_thread = threading.Thread(name="SceneViewer", target=self.Show)
        self.ui_thread.start()

    def CloseAndShutDown(self):
        with self.data_lock:  # notify model is changed, redraw is required
            self.do_computation_flag = False

        if not self.ui_thread is None:
            self.ui_thread.join()
            self.ui_thread = None

        if self.debug >= 3: print("uivis.CloseAndShutDown")

    def __PyGameLoop(self):
        require_redraw = False
        t1 = time.time()
        processed_images_counter_prev = -1
        while True:
            with self.data_lock:
                # check cancel request
                if not self.do_computation_flag:
                    break

            if pygame.event.peek():
                event = pygame.event.poll()
                # pygame.event.clear()  # clear all other events; try if got transparent window which is not redrawen
                if event.type == pygame.QUIT: # close button is clicked
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    break
                moved = self.__handlePyGameEvent(event)
                if moved:
                    require_redraw = True

            # query if model has changed
            processed_images_counter = -1
            with self.data_lock:
                # notify model is changed, redraw is required
                if self.world_map_changed_flag:
                    require_redraw = True
                    self.world_map_changed_flag = False  # reset the flag to avoid further redraw
                processed_images_counter = self.processed_images_counter

            #require_redraw = True # forcibly, exhaustively redraw a scene
            if require_redraw:
                t2 = time.time()
                #print("redraw")
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self.__DrawData()

                # images per sec
                if t2 - t1 > 0.5:  # display images per sec every X seconds
                    if processed_images_counter_prev != -1:
                        ips = (processed_images_counter - processed_images_counter_prev) / (t2 - t1)
                        title_str = "images_per_sec={0:05.2f}".format(ips)
                        pygame.display.set_caption(title_str)
                    processed_images_counter_prev = processed_images_counter
                    t1 = t2

                pygame.display.flip()  # flip after ALL updates have been made
                require_redraw = False
            pass  # require redraw

            # give other threads an opportunity to progress;
            # without yielding, the pygame thread consumes the majority of cpu
            time.sleep(0.03)
        pass  # game loop

        with self.data_lock:  # notify model is changed, redraw is required
            self.do_computation_flag = False

    def __DrawData(self):
        # draw the center of the world
        self.__DrawAxes(0.5)

        if self.draw_cameras:
            self.__DrawCameras()

        if self.draw_salient_points:
            with self.data_lock:
                glBegin(GL_POINTS)
                for scene_ind, xs3d in enumerate(self.list_of_xs3d):
                    if not self.scenes_visibile[scene_ind]:
                        continue
                    c = self.rgbcolor_per_scene[scene_ind]
                    glColor3f(c[0], c[1], c[2])

                    for pt in xs3d:
                        glVertex3f(pt[0],pt[1],pt[2])
                glEnd()

    def __DrawAxes(self, axis_seg_len):
        # draw axes in the local coordinates
        ax = axis_seg_len
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(ax, 0, 0) # OX
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, ax, 0) # OY
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, ax) # OZ
        glEnd()

    def __DrawPhysicalCamera(self, camR, camT, color, cam_to_world):
        # transform to the camera frame
        # cam_to_world=inv(world_to_cam)=[Rt,-Rt.T]
        cam_to_world.fill(0)
        cam_to_world[0:3, 0:3] = camR.T
        cam_to_world[0:3, 3] = -camR.T.dot(camT)
        cam_to_world[-1, -1] = 1

        glPushMatrix()
        glMultMatrixf(cam_to_world.ravel('F'))

        ax = 0.4
        self.__DrawAxes(ax)

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
        glColor3f(color[0], color[1], color[2])
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

    def __DrawCameras(self, draw_camera_each_frame=True):
        cam_to_world = np.eye(4, 4, dtype=np.float32)
        cam_pos_world = np.zeros(4, np.float32)

        with self.data_lock:
            # process multiple scenes
            for scene_ind, camera_from_world_RTs in enumerate(self.list_of_camera_from_world_RTs):
                if not self.scenes_visibile[scene_ind]:
                    continue

                cam_pos_world_prev = np.zeros(4, np.float32)
                cam_pos_world_prev_inited = False

                # draw the center of the axes
                track_col = self.rgbcolor_per_scene[scene_ind]
                glColor3f(track_col[0], track_col[0], track_col[0])

                cameras_count = len(camera_from_world_RTs)
                for i in range(0, cameras_count):
                    camRT = camera_from_world_RTs[i]
                    if camRT is None:
                        continue
                    camR, camT = camRT

                    # get position of the camera in the world: cam_to_world*(0,0,0,1)=cam_pos
                    cam_pos_world_tmp = -camR.T.dot(camT)
                    #assert np.isclose(1, cam_pos_world_tmp[-1]), "Expect cam_to_world(3,3)==1"
                    cam_pos_world[0:3] = cam_pos_world_tmp[0:3]

                    # draw trajectory of the camera
                    if cam_pos_world_prev_inited:
                        glBegin(GL_LINES)
                        glColor3f(1, 1, 1)
                        glVertex3fv(cam_pos_world_prev[0:3])
                        glVertex3fv(cam_pos_world[0:3])
                        glEnd()

                    glBegin(GL_POINTS)
                    glVertex3fv(cam_pos_world[0:3])
                    glEnd()

                    if draw_camera_each_frame:
                        self.__DrawPhysicalCamera(camR, camT, track_col, cam_to_world)

                    cam_pos_world_prev, cam_pos_world = cam_pos_world, cam_pos_world_prev
                    cam_pos_world_prev_inited = True
                pass # frame poses

                # draw head camera at the latest frame position
                # find the latest frame
                cam_ind = cameras_count - 1
                while cam_ind >= 0 and camera_from_world_RTs[cam_ind] is None:
                    cam_ind -= 1

                # current <- the latest frame position
                if cam_ind >= 0:
                    cam_rt = camera_from_world_RTs[cam_ind]
                    assert not cam_rt is None
                    camR, camT = cam_rt
                    self.__DrawPhysicalCamera(camR, camT, track_col, cam_to_world)
                pass # process head camera
            pass # two H decompositions
        pass # lock

    def __handlePyGameEvent(self, event):
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
        if event.type == pygame.KEYUP:
            if event.key >= pygame.K_0 and event.key <= pygame.K_9:
                # pygame.event.pump()
                # if not pygame.key.get_pressed()[event.key]:
                #     pass
                # print("{} pygame.key.get_pressed()[event.key]={}".format(time.time(), pygame.key.get_pressed()[event.key]))

                ind = event.key - pygame.K_0
                self.scenes_visibile[ind] = not self.scenes_visibile[ind]
                print("scenes_visibile[{}]={} scenes_visibile={}".format(ind, self.scenes_visibile[ind], self.scenes_visibile))
                moved = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pass

        if moved:
            glLoadIdentity()

            s = self.scene_scale
            glScale(s, s, s)

            gluLookAt(self.eye[0], self.eye[1], self.eye[2], self.center[0], self.center[1], self.center[2],
                      self.up[0], self.up[1], self.up[2])
            #print("eye={0} up={1} center={2} orthoRadius={3}".format(self.eye, self.up, self.center, self.orthoRadius))

        return moved
