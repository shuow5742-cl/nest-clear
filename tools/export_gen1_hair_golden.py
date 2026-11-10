#!/usr/bin/env python3
"""Run from ~/pluck-hair/backend with its venv; export synthetic and real goldens."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import dataclasses
import json
import math
from pathlib import Path
import random
import sys
import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(REPO / 'tests'))
from src.pluck import detect, detect_steps
from hair_scenes import SCENES, frame_for
OUTPUT = REPO / 'tests/data/hair_parity'
PATCHES = Path.home() / 'pluck-ds/patches'

def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return value

def sample_patches():
    groups = defaultdict(list)
    for path in sorted(PATCHES.glob('*.png')): groups['_'.join(path.name.split('_')[:2])].append(path)
    rng = random.Random(20261110)
    runs = sorted(groups); rng.shuffle(runs)
    chosen = []
    while len(chosen) < 60 and runs:
        for run in runs[:]:
            if not groups[run]: runs.remove(run); continue
            chosen.append(groups[run].pop(rng.randrange(len(groups[run]))))
            if len(chosen) == 60: break
    return chosen

def snapshot(clumps, relabel=False):
    def key(c):
        r = (tuple(map(tuple,c.polygon_xy)), dataclasses.astuple(c.bbox), c.mask_area, c.pick_point_xy, c.pick_angle_deg)
        if relabel: r += (c.eligible_for_pick,c.reject_reason,c.object_type)
        return repr(r)
    return sorted(key(c) for c in clumps)

def export(frame, counts):
    ctx = detect_steps._new_context(frame, dark_max=detect._HAIR_DARK_MAX)
    for i, step in enumerate(detect_steps.build_detect_steps()):
        name = f'{i+1:02d}_{type(step).__name__}'
        before = snapshot(ctx.detections, type(step).__name__ == 'RelabelStep')
        ctx = step.process(ctx)
        after = snapshot(ctx.detections, type(step).__name__ == 'RelabelStep')
        counts[name] += int(before != after)
    hairs, clumps = detect_steps.detect_hairs(None, frame)
    assert hairs == ctx.metadata['hairs'] and clumps == ctx.detections
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return {'hairs':[dataclasses.asdict(h) for h in hairs], 'clumps':[dataclasses.asdict(c) for c in clumps], 'paleness':[detect.clump_paleness(c,gray) for c in clumps]}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode',choices=('synthetic','real','all'),default='all'); args=parser.parse_args()
    counts=Counter()
    if args.mode in ('synthetic','all'):
        for name in SCENES:
            write(OUTPUT / f'{name}.json',export(frame_for(name),counts)); print(name,flush=True)
    if args.mode in ('real','all'):
        paths=sample_patches()
        if len(paths)!=60: raise RuntimeError(f'Expected 60 images in {PATCHES}, found {len(paths)}')
        for path in paths:
            frame=cv2.imread(str(path))
            if frame is None: raise ValueError(f'Cannot read {path}')
            write(OUTPUT/'real'/f'{path.stem}.json',{'filename':path.name,**export(frame,counts)}); print(path.name,flush=True)
        print(f'{len(paths)} patches, {len({"_".join(p.name.split("_")[:2]) for p in paths})} runs')
    write(OUTPUT/f'coverage_{args.mode}.json',dict(counts)); print(dict(counts))
if __name__=='__main__': main()
