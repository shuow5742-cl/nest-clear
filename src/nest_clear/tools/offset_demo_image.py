"""用与无 TCP 测试相同的相机参数生成工具偏移演示图。"""
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np

from nest_clear.tooloffset.session import nominal_targets
from nest_clear.workers.tooloffset import ToolOffsetWorker


def main(root):
    pose = np.array([145.,85.]); origin = np.array([130.,70.]); angle = .02
    matrix = 1.005 * np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    def pixel(p):
        world = origin + matrix @ np.asarray(p)
        return np.array([2000 + 20*(world[0]-pose[0]), 1500 - 20*(world[1]-pose[1])])
    image = np.full((3000,4000), 245, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for index, point in nominal_targets().items():
        center = pixel(point)
        if not (-150 < center[0] < 4150 and -150 < center[1] < 3150): continue
        xy = tuple(np.rint(center).astype(int))
        cv2.circle(image, xy, 37, 10, 6, cv2.LINE_AA)
        for axis in ([5,0],[0,5]):
            v = np.asarray(axis)
            cv2.line(image, tuple(np.rint(pixel(point-v)).astype(int)), tuple(np.rint(pixel(point+v)).astype(int)), 10, 2, cv2.LINE_AA)
        marker = cv2.aruco.generateImageMarker(dictionary, index, 100)
        destination = np.float32([pixel(point+[5,9]),pixel(point+[9,9]),pixel(point+[9,5]),pixel(point+[5,5])])
        transform = cv2.getPerspectiveTransform(np.float32([[0,0],[99,0],[99,99],[0,99]]), destination)
        image = np.minimum(image, cv2.warpPerspective(marker, transform, (4000,3000), borderValue=255))
    class Events:
        directory = Path(root)
        run_directory = None
        def emit(self, *args, **kwargs): pass
    ctx = SimpleNamespace(events=Events(), calibration=SimpleNamespace(K=np.array([[2000.,0,2000],[0,2000.,1500],[0,0,1.]])))
    session = {'shots':0, 'engine':SimpleNamespace(current=12)}
    ToolOffsetWorker().execute({'captured':(image,{}), 'context':ctx, 'session':session, 'task_id':1})


if __name__ == '__main__': main(sys.argv[1])
