"""按模组位姿渲染畸变 ChArUco 板的合成相机。"""
import cv2
import numpy as np

from nest_clear.comm import PLCClient
from nest_clear.camcal import CameraCalibConfig
from nest_clear.camcal.session import _board

K = np.array([[4650., 0., 2020.], [0., 4655., 1510.], [0., 0., 1.]])
DISTORTION = np.array([-.06, .03, .0002, -.0001, 0.])
ROTATION = cv2.Rodrigues(np.array([.008, .011, .026]))[0] @ np.diag([1., -1., -1.])
_PLANE_ROTATION = cv2.Rodrigues(np.array([.002, .0028, .01]))[0]
BOARD_ROTATION = _PLANE_ROTATION @ np.diag([1., -1., -1.])
BOARD_ORIGIN = np.array([-52.5, -40., 0.]) + _PLANE_ROTATION @ np.array([0., 80., 0.])


def sample_board(texture, map_x, map_y):
    """用逐像素板面坐标重采样，并用灰色表示板外背景。"""
    return cv2.remap(texture, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=180)


class CharucoScene:
    """像素射线与实际倾斜板面求交，纹理按双线性插值采样。"""
    def __init__(self, size=(4024, 3036), k=K, distortion=DISTORTION,
                 rotation=ROTATION, board_rotation=BOARD_ROTATION,
                 board_origin=BOARD_ORIGIN, config=None):
        self.size = tuple(size)
        self.k, self.distortion = np.asarray(k), np.asarray(distortion)
        self.rotation, self.board_rotation = np.asarray(rotation), np.asarray(board_rotation)
        self.board_origin = np.asarray(board_origin)
        self.config = config or CameraCalibConfig()
        self.texture = (_board(self.config).generateImage((2100, 1600)).astype(np.float32) * .8).astype(np.uint8)
        # 分块构造，避免生成全图双精度临时数组。
        w, h = self.size
        self.rays = np.empty((h, w, 3), np.float32)
        for y in range(0, h, 128):
            yy, xx = np.mgrid[y:min(y + 128, h), :w].astype(np.float32)
            pixels = np.stack((xx + .5, yy + .5), axis=-1)
            uv = cv2.undistortPointsIter(pixels.reshape(-1, 1, 2), self.k, self.distortion, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-12)).reshape(*xx.shape, 2)
            self.rays[y:y + len(xx), :, :2] = uv
            self.rays[y:y + len(xx), :, 2] = 1
        self.rays = self.rays @ (self.rotation.T @ self.board_rotation)

    def render(self, xyz, *, glare=None, overexposed=False, focus_z=110.):
        w, h = self.size
        if overexposed:
            return np.full((h, w), 255, np.uint8)
        origin = (np.asarray(xyz) - self.board_origin) @ self.board_rotation
        # 校正纹理栅格与 OpenCV 角点中心的半像素约定。
        t = -origin[2] / self.rays[:, :, 2]
        mx = ((origin[0] + t * self.rays[:, :, 0]) * self.texture.shape[1] /
              (self.config.cols * self.config.square_mm) - .5).astype(np.float32)
        my = ((origin[1] + t * self.rays[:, :, 1]) * self.texture.shape[0] /
              (self.config.rows * self.config.square_mm) - .5).astype(np.float32)
        image = sample_board(self.texture, mx, my)
        if abs(float(xyz[2]) - focus_z) > 2.6:
            image = cv2.GaussianBlur(image, (3, 3), .5)
        if glare is not None:
            center, axes = glare
            cv2.ellipse(image, center, axes, 0, 0, 360, 255, -1)
        return image


class SimulatedCharucoCamera:
    """从 PLC 实际运动状态生成拍摄图像。"""
    def __init__(self, plc, *, glare_frames=(), overexposed_frames=()):
        self.plc = plc
        self.io = PLCClient(plc.host, plc.port, 1, plc.contract)
        self.scene = CharucoScene()
        self.glare_frames = set(glare_frames)
        self.overexposed_frames = set(overexposed_frames)
        self.frames = 0

    def open(self):
        return True

    def close(self):
        self.io.close()

    def get_frame_size(self):
        return self.scene.size

    def snapshot(self):
        self.frames += 1
        feedback = self.io.read_real32_block(self.plc.contract['actual_x'].address, 3)
        xyz = getattr(self.plc, 'physical_xyz', feedback)
        glare = ((2012, 1518), (100, 80)) if self.frames in self.glare_frames else None
        return self.scene.render(xyz, glare=glare, overexposed=self.frames in self.overexposed_frames)
