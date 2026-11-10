#!/usr/bin/env python3
"""Repeatable, non-gating 4024x3036 hair detector timing."""
from statistics import median
from time import perf_counter
import cv2
import numpy as np
from nest_clear.vision.hair import HairDetectConfig, detect_candidates, scaled


def main():
    width, height = 4024, 3036
    rng = np.random.default_rng(20261110)
    gray = np.clip(np.linspace(170, 210, width)[None, :] + rng.normal(0, 2, (height, width)), 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(12):
        cv2.ellipse(frame, (250 + (i % 4) * 950, 400 + (i // 4) * 1000),
                    (90, 18), 12 * i, 0, 360, (30, 30, 30), -1)
    cfg = scaled(HairDetectConfig())
    detect_candidates(frame, cfg)  # warm-up
    samples, stages = [], []
    for _ in range(5):
        timing = {}
        start = perf_counter()
        result = detect_candidates(frame, cfg, timings_ms=timing)
        samples.append(1000 * (perf_counter() - start))
        stages.append(timing)
    print(f'4024x3036, 5 次中位数: {median(samples):.2f} ms, 目标 {len(result)}')
    for key in stages[0]:
        print(f'{key}: {median(s[key] for s in stages):.2f} ms')

if __name__ == '__main__':
    main()
