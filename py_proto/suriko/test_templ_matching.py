import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from suriko.geom_elements import *
from suriko.OpenCV_interop import *
from suriko.patch_match import *


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

def TestMeanAndVar(data_dir):
    img0 = cv2.imread(os.path.join(data_dir, "rawoutput0000.pgm"), flags=cv2.IMREAD_GRAYSCALE)
    patch_top_left=Point2i(170,104)
    patch_size=Size2i(17, 17)
    mean = PatchMean(img0, patch_top_left, patch_size)
    print("Mean={}".format(mean))

    varia = PatchVariance(img0, patch_top_left, patch_size, mean)
    print("Variance={}".format(varia))

    mean2, varia2 = PatchMeanAndVariance(img0, patch_top_left, patch_size)
    print("Mean={} variance={}".format(mean2,varia2))

    patch = img0[patch_top_left.y:patch_top_left.y + patch_size.height, patch_top_left.x:patch_top_left.x + patch_size.width]
    zero_off = Point2i(0,0) # zero offset in the patch itself
    p1,p2 = PatchMeanAndVariance(patch, zero_off, patch_size)
    pass

def TestMatchInConcreteLocation(data_dir):
    img0 = cv2.imread(os.path.join(data_dir, "rawoutput0000.pgm"), flags=cv2.IMREAD_GRAYSCALE)
    patch_top_left=Point2i(170,104)
    patch_size=Size2i(17, 17)

    patch = img0[patch_top_left.y:patch_top_left.y + patch_size.height, patch_top_left.x:patch_top_left.x + patch_size.width]
    patch_mean, patch_var = PatchMeanAndVariance(img0, patch_top_left, patch_size)
    print("Mean={} variance={}".format(patch_mean,patch_var))

    test_top_left = Point2i(170,103)
    img1 = cv2.imread(os.path.join(data_dir, "rawoutput0018.pgm"), flags=cv2.IMREAD_GRAYSCALE)
    suc,corr = MatchPatchCorrCoefNormed(img1, test_top_left, patch, patch_mean, patch_var)
    if not suc: return
    print("Corr={}".format(corr))

def TestMatchInSearchRect(data_dir, initial_feat_center:Point2i, patch_size:Size2i, start_image_ind, search_size:Size2i,patch_data_dict:Dict[int,PatchMatchRecord]):
    # first file
    file_list = os.listdir(data_dir)
    origin_patch_image_ind = 0 # take original patch from this image
    first_file_name = file_list[origin_patch_image_ind]
    img0 = cv2.imread(os.path.join(data_dir, first_file_name), flags=cv2.IMREAD_GRAYSCALE)
    patch_half_width = int(patch_size.width / 2)
    patch_half_height = int(patch_size.height / 2)
    patch_top_left=Point2i(initial_feat_center.x - patch_half_width, initial_feat_center.y - patch_half_height)
    patch_bot_right = Point2i(patch_top_left.x + patch_size.width, patch_top_left.y + patch_size.height)

    patch = img0[patch_top_left.y:patch_top_left.y + patch_size.height, patch_top_left.x:patch_top_left.x + patch_size.width]
    patch_mean, patch_var = PatchMeanAndVariance(img0, patch_top_left, patch_size)
    print("Patch mean={} var={}".format(patch_mean,patch_var))

    # copy, to prevent corrupting the patch object
    img0_draw = img0.copy()
    cv2.rectangle(img0_draw, ocv(patch_top_left), ocv(patch_bot_right), 255)
    cv2.imshow("img0", img0_draw)

    # process all files, including the first one
    file_ind:int = start_image_ind
    prev_file_ind = file_ind - 1
    if file_ind in patch_data_dict:
        cur_patch_data = patch_data_dict[prev_file_ind]
        cur_center = cur_patch_data.center
    else:
        cur_center = initial_feat_center
    go_next:bool = True

    while file_ind < len(file_list):
        file_name = file_list[file_ind]
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            file_ind += 1
            continue
        img1 = cv2.imread(os.path.join(data_dir, file_path), flags=cv2.IMREAD_GRAYSCALE)

        # search in vicinity of a current position
        cur_top_left = TopLeft(cur_center, patch_size)
        cur_top_left_search_rect = Rect2i(cur_top_left.x-search_size.width/2, cur_top_left.y-search_size.height/2, search_size.width, search_size.height)

        best_top_left,corr_coeff = MatchPatchInSearchRect(img1, cur_top_left_search_rect, patch, patch_mean, patch_var)
        best_center = Point2i(best_top_left.x + patch_half_width,best_top_left.y + patch_half_height)
        print("f={} center={} top-left={} corr={} file={}".format(file_ind, best_center, best_top_left, corr_coeff, file_name))

        new_rect = Rect2i(best_top_left.x, best_top_left.y, patch_size.height, patch_size.width)
        win_name = "match patches"
        cv2.rectangle(img1, ocv(new_rect.TL()), ocv(new_rect.BR()), 255)
        cv2.imshow(win_name, img1)

        key = cv2.waitKey(0)
        if key == ord('g'): # good, go further
            pass
        elif key == ord('s'): # skip, go further
            best_center=Point2i(-1,-1)
            corr_coeff=-1
        elif key == ord('b'):  # break
            break
        elif key == ord('r'): # retry
            file_ind += 0 # try again this image
            continue
        elif key == 45: # NumPad-Minus=go previous
            file_ind -= 1
            cur_patch_data = patch_data_dict[file_ind]
            cur_center = cur_patch_data.center
            print("f={} center={} file={}".format(file_ind, cur_center, file_name))
            continue
        elif key == 43: # NumPad-Plus=go next
            file_ind += 1
            cur_patch_data = patch_data_dict[file_ind]
            cur_center = cur_patch_data.center
            print("f={} center={} file={}".format(file_ind, cur_center, file_name))
            continue
        elif key == ord('='): # set current frame here
            cur_patch_data = patch_data_dict[file_ind]
            cur_center = cur_patch_data.center
            print("f={} center={} file={}".format(file_ind, cur_center, file_name))
            continue
        elif key == ord('`'): # switch go 'next/forward' and retry again
            go_next = not go_next
            print("go_next={}".format(go_next))
            continue
        else: # retry
            file_ind += 0 # try again this image
            continue

        rec = PatchMatchRecord(file_ind, best_center, corr_coeff)
        patch_data_dict[file_ind] = rec

        if best_center.x != -1:
            cur_center = best_center
        file_ind += 1 if go_next else -1

def RunMatchTemplateInImageSeq(data_dir):
    patch_top_left=Point2i(170,104)
    patch_size = Size2i(17, 17)
    patch_center = Point2i(patch_top_left.x+patch_size.width/2,patch_top_left.y+patch_size.height/2)

    out_file_path="suriko/interm/feat_x{}_y{}.txt".format(patch_center.x, patch_center.y)
    patch_data_dict:Dict[int,PatchMatchRecord] = dict()
    start_image_ind = 0
    if os.path.exists(out_file_path):
        patch_data_dict = LoadPatchMatchDict(out_file_path)
        start_image_ind = max(patch_data_dict.keys())
        start_image_ind += 1 # process next

    vicinity_size = Size2i(35,35)
    TestMatchInSearchRect(data_dir, initial_feat_center=patch_center, patch_size=patch_size,
                          start_image_ind=start_image_ind, search_size=vicinity_size,
                          patch_data_dict=patch_data_dict)
    # store result
    npdata = np.array([[
        r.frame_ind,
        r.center.x,
        r.center.y,
        r.corr_coef]
        for r in patch_data_dict.values()])
    outfmt=[
        "%i", # frame_ind
        "%i", # center.x
        "%i", # ceter.y
        "%f", # corr coef
    ]
    np.savetxt(out_file_path, npdata, header="Frame CentX CentY CorrC", fmt=outfmt)


# Analyze correlation coefficients for all pixels in a single image
def RunMatchTemplateInSingleImage(data_dir):
    first_file_name = "rawoutput0000.pgm"
    image = cv2.imread(os.path.join(data_dir, first_file_name), flags=cv2.IMREAD_GRAYSCALE)

    patch_size = Size2i(17, 17)
    patch_center = Point2i(178, 112)
    patch_top_left = TopLeft(patch_center, patch_size)


    patch = image[patch_top_left.y:patch_top_left.y + patch_size.height, patch_top_left.x:patch_top_left.x + patch_size.width]
    patch_mean, patch_var = PatchMeanAndVariance(image, patch_top_left, patch_size)
    print("Patch mean={} var={}".format(patch_mean,patch_var))

    class TopLeftCorr:
        def __init__(self, top_left, corr_coef):
            self.top_left = top_left
            self.corr_coef = corr_coef

    list_corr_coefs:List[TopLeftCorr] = []

    bot = image.shape[0] - patch_size.height
    right = image.shape[1] - patch_size.width
    for top_left_y in range(0, bot):
        for top_left_x in range(0, right):
            test_top_left=Point2i(top_left_x,top_left_y)
            #test_top_left = Point2i(139, 214)
            suc,corr = MatchPatchCorrCoefNormed(image, test_top_left, patch, patch_mean, patch_var)
            if not suc:
                print("top_left={} corr=null".format(test_top_left))
                continue
            print("top_left={} corr={}".format(test_top_left, corr))
            list_corr_coefs.append(TopLeftCorr(test_top_left, corr))
            #if len(list_corr_coefs) > 1000: break

    # top-10 matches
    def ByCorrCoeff(x:TopLeftCorr):
        return x.corr_coef
    top10= sorted(list_corr_coefs, reverse=True, key=ByCorrCoeff)[0:10]
    print("top10:")
    for x in top10:
        print("top_left={} corr={}".format(x.top_left, x.corr_coef))

    # histogram of correlation coefficients
    corr_coefs = [x.corr_coef for x in list_corr_coefs]
    plt.hist(corr_coefs)
    plt.show()

    print("end")


if __name__ == '__main__':
    data_dir = "E:/distrib/cv/SceneLib/testseqmonoslam/TestSeqMonoSLAM_Decimated"
    #TestMeanAndVar(data_dir)
    #TestMatchInConcreteLocation(data_dir)
    #RunMatchTemplateInImageSeq(data_dir)
    RunMatchTemplateInSingleImage(data_dir)
