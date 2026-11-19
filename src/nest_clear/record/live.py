import logging
import os
import cv2
from nest_clear.paths import live_image_file

def write_live(root, name, image):
    try:
        path = live_image_file(root, name); path.parent.mkdir(parents=True, exist_ok=True)
        ok, data = cv2.imencode('.jpg', image)
        if not ok: raise ValueError('JPEG 编码失败')
        tmp = path.with_suffix('.tmp'); tmp.write_bytes(data.tobytes()); os.replace(tmp, path)
    except Exception:
        logging.getLogger(__name__).exception('写实时图片失败')
