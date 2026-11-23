import argparse
from .app import App

def add_view_arguments(parser):
    parser.add_argument('--page', default=None, help='页面名称或 main/home、camcal/camera、offset/tool-offset、pickacc/pick-accuracy、baseline、toolcal/tooltip')
    parser.add_argument('--geometry', help='窗口尺寸，例如 1920x1080')
    parser.add_argument('--screenshot', '--screenshot-path', dest='screenshot', help='截图输出文件，例如 reports/ui/home.png')
    parser.add_argument('--screenshot-after-ms', '--screenshot-delay-ms', dest='screenshot_after_ms', type=int, help='启动后等待多少毫秒截图并退出')

def validate_view_arguments(parser, args):
    if (args.screenshot is None) != (args.screenshot_after_ms is None):
        parser.error('--screenshot 和 --screenshot-after-ms 必须一起指定')
    if args.screenshot_after_ms is not None and args.screenshot_after_ms < 0:
        parser.error('--screenshot-after-ms 不能为负数')

def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', default='runs'); p.add_argument('--fullscreen', action='store_true'); p.add_argument('--refresh-ms', type=int, default=300); add_view_arguments(p)
    a = p.parse_args(); validate_view_arguments(p, a)
    app = App(a.root, a.refresh_ms, a.fullscreen, a.page, a.screenshot, a.screenshot_after_ms)
    if a.geometry:
        try:
            app.root.state('normal')
            app.root.attributes('-zoomed', False)
        except Exception:
            pass
        app.root.geometry(a.geometry)
    app.run()
if __name__ == '__main__': main()
