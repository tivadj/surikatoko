import numpy as np
from suriko.geom_elements import Point2i, Size2i


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