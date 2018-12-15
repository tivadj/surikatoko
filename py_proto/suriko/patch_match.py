import math
import numpy as np
from suriko.geom_elements import Point2i, Size2i, Rect2i, IntersectRects


def PatchMean(image, top_left_pos:Point2i, patch_size:Size2i):
    sum:float = 0
    for row in range(0, patch_size.height):
        for col in range(0, patch_size.width):
            i = float(image[top_left_pos.y + row, top_left_pos.x + col])
            sum += i
    return sum / (patch_size.width*patch_size.height)

def PatchVariance(image, top_left_pos:Point2i, patch_size:Size2i, patch_mean):
    sum: float = 0
    for row in range(0, patch_size.height):
        for col in range(0, patch_size.width):
            i = float(image[top_left_pos.y + row, top_left_pos.x + col])
            sum += (i - patch_mean) ** 2
    return sum / (patch_size.width*patch_size.height)

# Note, this doesn't track the count of numbers, pushed into it.
class PatchMeanAndVarianceAlgo:
    def __init__(self):
        self.sum_x = float(0)
        self.sum_xx = float(0)

    def PutNext(self, i):
        self.sum_x += i
        self.sum_xx += i * i

    def GetMeanAndVariance(self, num_el):
        mean = self.sum_x / num_el
        varia = self.sum_xx / num_el - mean * mean
        return mean, varia

# Gets the mean and the variance in one pass.
def PatchMeanAndVariance(image, top_left_pos:Point2i, patch_size:Size2i):
    sum_x:float = 0
    sum_xx:float = 0
    for row in range(0, patch_size.height):
        for col in range(0, patch_size.width):
            i = float(image[top_left_pos.y + row, top_left_pos.x + col])
            sum_x += i
            sum_xx += i*i
    num_el = patch_size.width*patch_size.height
    mean = sum_x / num_el                  # E[X]
    varia = sum_xx / num_el - mean * mean  # var(X)=E[X^2]-E[X]^2
    return mean, varia


# Implements OpenCV.templateMatch(method=CV_TM_CCOEFF_NORMED)
# https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
def MatchPatchCorrCoefNormed(image, top_left_pos:Point2i, patch, patch_mean, patch_var) -> (bool,float):
    sum_xdev_ydev:float = 0  # sum of deviation(x)*deviation(y)
    patch_size = Size2i(patch.shape[1],patch.shape[0])

    image_mean, image_var = PatchMeanAndVariance(image, top_left_pos, patch_size)
    if image_var == 0:
        return False, -1

    for row in range(0, patch_size.height):
        for col in range(0, patch_size.width):
            t = float(patch[row, col])
            i = float(image[top_left_pos.y + row, top_left_pos.x + col])
            sum_xdev_ydev += (t-patch_mean) * (i-image_mean)

    num_el = patch_size.width*patch_size.height
    corr = sum_xdev_ydev / (math.sqrt(patch_var) * math.sqrt(image_var))

    # the original formula uses sum of squares of residuals, but we used variance
    # hence, additionally divide numerator by adjusting factor
    corr /= num_el
    return True, corr

def MatchPatchInSearchRect(image, top_left_search_rect:Rect2i, patch, patch_mean, patch_var) -> (Point2i, float):
    win = Rect2i(0, 0, image.shape[1] - patch.shape[1], image.shape[0] - patch.shape[0])
    search_rect_safe = IntersectRects(top_left_search_rect, win)

    max_corr_coeff = -1
    max_corr_coeff_top_left = None
    for row in range(search_rect_safe.y, search_rect_safe.Bottom()):
        for col in range(search_rect_safe.x, search_rect_safe.Right()):
            test_top_left = Point2i(col,row)

            suc,corr = MatchPatchCorrCoefNormed(image, test_top_left, patch, patch_mean, patch_var)
            if not suc: continue
            #print("test cell={} corr={}".format(test_cell, corr))

            if corr > max_corr_coeff:
                max_corr_coeff = corr
                max_corr_coeff_top_left = test_top_left
    return max_corr_coeff_top_left,max_corr_coeff

class PatchMatchRecord:
    def __init__(self, frame_ind:int, center:Point2i, corr_coef:float):
        self.frame_ind = frame_ind
        self.center = center
        self.corr_coef = corr_coef

    def __str__(self):
        return "[{} {} {}]".format(self.frame_ind, self.center, self.corr_coef)

def TopLeft(center:Point2i, patch_size:Size2i) -> Point2i:
    return Point2i(center.x-patch_size.width/2, center.y-patch_size.height/2)

def LoadPatchMatchDict(file_path:str):
    data_array = np.loadtxt(file_path, skiprows=1)
    patch_data_dict = dict([(x[0], PatchMatchRecord(x[0], Point2i(x[1], x[2]), x[3])) for x in data_array])
    return patch_data_dict