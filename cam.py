import cv2
from utils import mkdir, normalize
import numpy as np


class FeatureHook:

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output.cpu().numpy()

    def remove(self):
        self.hook.remove()


def get_cam(feature_map, fc_weight):
    b, c, h, w = feature_map.shape
    cam = fc_weight.dot(feature_map.reshape(c, h*w))
    cam = cam.reshape(h, w)
    cam = normalize(cam)
    return cam


def blend_cam(img, cam, size=(1000, 100)):
    h, w = img.shape
    if h > w:
        img, cam = img.T, cam.T
    img = cv2.resize(normalize(img), size)
    cam = cv2.resize(normalize(cam), size)
    res = ((img*0.5+cam*0.5)*255).astype(int).clip(0, 255).astype(np.uint8)
    return res

