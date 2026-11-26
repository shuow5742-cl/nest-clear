"""Real Tk geometry with synthetic frames and a virtual X display.

Run directly with system Python under Xvfb to bypass the pytest fixtures.
Pass --shots DIRECTORY to also capture the two resolutions and scan dialog.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageGrab


def synthetic_input(directory):
    from nest_clear.paths import live_image_file, tooltip_preview_file

    cameras, previews = {}, {}
    for name, background, x in (('front', '#ecf0ee', 970), ('side', '#dce8ed', 910)):
        path = live_image_file(directory, 'baseline_' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (1920, 1080), background)
        draw = ImageDraw.Draw(image)
        for y in range(0, 1080, 80):
            draw.line((0, y, 1920, y), fill='#c5d1d3')
        draw.line((x, 0, x, 1080), fill='#6b9e83', width=2)
        draw.line((0, 590, 1920, 590), fill='#6b9e83', width=2)
        draw.polygon([(x-42, 0), (x+42, 0), (x+29, 425), (x+16, 563),
                      (x+7, 586), (x+4, 505), (x-4, 505), (x-8, 588),
                      (x-17, 560), (x-29, 425)], fill='#343d46')
        draw.ellipse((x-35, 555, x+35, 625), outline='#168448', width=3)
        draw.text((28, 26), 'SYNTHETIC / ' + name.upper(), fill='#283943')
        image.save(path)
        rows = [dict(exposure=i, recognized=True, prongs=2, prong_ref=2, contrast=60+i,
                     sharpness_percent=100-i, background=160+i*4, saturated_fraction=.02,
                     tool_whiteness=.01, qualified=i == 3, selected=i == 3,
                     reasons=[] if i == 3 else ['清晰度不足']) for i in range(1, 13)]
        previews[name] = {'on': True, 'device': f'usb-0:{4 if name == "front" else 8}',
                          'exposure_limits': {'min': 1, 'max': 125, 'step': 1},
                          'exposure_saved': {'exposure': 3, 'time': '2026-11-25'},
                          'exposure_source': '已测曝光', 'exposure_scan_rows': rows}
        cameras[name] = {'source': 'band_gradient_p995', 'percent': 98, 'band_gradient': 1.28,
                         'peak': 1.3, 'detection_status': '精定位', 'tip_uv': [x, 587],
                         'edge_width_px': 1.8, 'edge_width_um': 24.3, 'peak_edge_width_um': 23.8,
                         'contrast': 88, 'exposure_ms': .3, 'history': [69, 78, 86, 85, 93, 98]}
    path = tooltip_preview_file(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'cameras': cameras}), encoding='utf-8')
    return {'mode': '空闲', 'baseline': {'phase': 'live', 'tool': 1, 'preview': previews,
                                      'message': '两路相机已精定位，等待建立对刀基准',
                                      'live_delta': [.006, -.003], 'live_inside': True}}


@contextmanager
def baseline_window(width, height):
    blocked = dict.fromkeys(('cv2', 'numpy', 'yaml'))
    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, blocked):
        from nest_clear.ui.app import App

        directory = Path(directory)
        data = synthetic_input(directory)
        try:
            app = App(directory, page='baseline', ignore_before_ns=0)
        except tk.TclError as exc:
            if 'display' in str(exc) and ('connect' in str(exc) or 'no display name' in str(exc)):
                raise unittest.SkipTest(f'需要可访问的虚拟 X 显示：{exc}') from exc
            raise
        root = app.root
        try:
            root.state('normal')
            root.attributes('-zoomed', False)
            root.geometry(f'{width}x{height}+0+0')
            app.update_baseline(data)
            root.update_idletasks()
            pictures = [view['picture'] for view in app.baseline_views.values()]
            deadline = time.monotonic() + 5
            while not all(pic.photo is not None for pic in pictures):
                if time.monotonic() >= deadline:
                    raise AssertionError('等待两路合成帧超时')
                for pic in pictures:
                    pic.poll()
                root.update_idletasks()
                time.sleep(.01)
            yield app, data, pictures
        finally:
            root.destroy()


def geometry(picture):
    return picture.winfo_width(), picture.winfo_height(), picture.photo.width(), picture.photo.height()


class BaselineLayoutTest(unittest.TestCase):
    def test_synthetic_frame_geometry_and_scan_popup(self):
        with baseline_window(1920, 1080) as (app, data, pictures):
            root = app.root
            self.assertEqual((root.winfo_width(), root.winfo_height()), (1920, 1080))
            sizes = [geometry(pic) for pic in pictures]
            for pic, (width, height, photo_width, photo_height) in zip(pictures, sizes):
                self.assertGreaterEqual(photo_width, 900)
                self.assertLessEqual(photo_width, width)
                self.assertLessEqual(photo_height, height)
                self.assertLessEqual(abs(photo_height - photo_width * 9 / 16), 2)
                self.assertLessEqual(pic.winfo_rooty() - root.winfo_rooty() + height, 1080)
            for _ in range(5):
                for pic in pictures:
                    area = pic.master
                    area.event_generate('<Configure>', width=area.winfo_width(), height=area.winfo_height())
                root.update_idletasks()
                self.assertEqual([geometry(pic) for pic in pictures], sizes)
            app.baseline_views['front']['scan_button'].invoke()
            root.update_idletasks()
            popup = app.baseline_scan_dialogs['front']
            self.assertTrue(popup['window'].winfo_ismapped())
            text = popup['table'].cget('text')
            self.assertEqual(len(text.splitlines()), 13)
            self.assertIn('齿数（参考 2）', text)
            self.assertIn('0.3', text)
            self.assertIn('★', text)
            item = data['baseline']['preview']['front']
            item.update(exposure_busy=True, exposure_progress='正在测曝光 第1/12档',
                        exposure_scan_rows=item['exposure_scan_rows'][:1])
            app.update_baseline(data)
            root.update_idletasks()
            self.assertIn('第1/12档', popup['status'].cget('text'))
            self.assertEqual(len(popup['table'].cget('text').splitlines()), 2)
            popup['window'].destroy()
            self.assertNotIn('front', app.baseline_scan_dialogs)

    def test_smaller_window_keeps_cameras_and_controls_visible(self):
        with baseline_window(1366, 768) as (app, _, pictures):
            widgets = [app.baseline_start, *app.baseline_tools, app.baseline_measure,
                       app.baseline_manual, app.baseline_aux, app.baseline_save, app.baseline_end,
                       app.runout_start, app.runout_press, app.runout_end, *pictures]
            for view in app.baseline_views.values():
                widgets.extend(view[key] for key in ('toggle', 'exposure', 'reset', 'exposure_entry',
                                                     'exposure_set', 'exposure_restore', 'exposure_save', 'scan_button'))
            for widget in widgets:
                self.assertTrue(widget.winfo_ismapped())
                self.assertLessEqual(widget.winfo_rootx() - app.root.winfo_rootx() + widget.winfo_width(), 1366)
                self.assertLessEqual(widget.winfo_rooty() - app.root.winfo_rooty() + widget.winfo_height(), 768)
            for pic in pictures:
                _, _, width, height = geometry(pic)
                self.assertGreater(width, 500)
                self.assertLessEqual(abs(height - width * 9 / 16), 2)


def capture(window, path):
    window.update_idletasks()
    x, y = window.winfo_rootx(), window.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + window.winfo_width(), y + window.winfo_height())).save(path)


def screenshots(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for width, height in ((1920, 1080), (1366, 768)):
        with baseline_window(width, height) as (app, _, pictures):
            print(f'{width}×{height}: {[geometry(pic) for pic in pictures]}')
            capture(app.root, destination / f'baseline_{width}.png')
            if width == 1920:
                app.baseline_views['front']['scan_button'].invoke()
                capture(app.baseline_scan_dialogs['front']['window'], destination / 'scan_popup.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', type=Path)
    args = parser.parse_args()
    if args.shots is not None:
        screenshots(args.shots)
    else:
        unittest.main()
