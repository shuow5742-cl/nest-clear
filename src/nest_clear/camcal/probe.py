"""只读 PLC 坐标诊断及会话汇总。"""
import itertools
import math


def make_probe(sequence, target, before, after, actual):
    target = list(target)
    def enrich(samples):
        return [{**s, 'delta_target_mm': [v - t for v, t in zip(s['xyz'], target)]} for s in (samples or [])]
    return {'sequence': sequence, 'target_xyz': target, 'algorithm_actual_xyz': list(actual),
            'before': enrich(before), 'after': enrich(after)}


def summarize(probes):
    shots = []
    for p in probes:
        before, after = p['before'], p['after']
        jitter = [max(s['xyz'][i] for s in before) - min(s['xyz'][i] for s in before) for i in range(3)] if before else None
        delta = [after[0]['xyz'][i] - before[-1]['xyz'][i] for i in range(3)] if before and after else None
        shots.append({'sequence': p['sequence'], 'target_xyz': p['target_xyz'], 'algorithm_actual_xyz': p['algorithm_actual_xyz'], 'before_jitter_mm': jitter, 'after_minus_before_mm': delta})
    groups = {}
    for p in probes: groups.setdefault(tuple(p['target_xyz']), []).append(p)
    repeated = []
    for target, group in groups.items():
        if len(group) < 2: continue
        pairs = []
        for a, b in itertools.combinations(group, 2):
            delta = [y-x for x, y in zip(a['algorithm_actual_xyz'], b['algorithm_actual_xyz'])]
            pairs.append({'sequences': [a['sequence'], b['sequence']], 'delta_xyz_mm': delta, 'delta_xy_mm': math.hypot(*delta[:2])})
        repeated.append({'target_xyz': list(target), 'pairs': pairs})
    return {'version': 1, 'shots': shots, 'repeated_targets': repeated}
