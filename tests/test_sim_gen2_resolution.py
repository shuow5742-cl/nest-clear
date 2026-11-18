from nest_clear.tools.sim_gen2_resolution import _stats, _touches


def test_group_statistics_and_distinct_error_measures():
    rows = [{
        'ref': [{'truncated': False}, {'truncated': True}],
        'gen': [{'truncated': False}, {'truncated': True}],
        'pairs': [{'ref': 0, 'gen': 0, 'on_hair_mm': 0.0, 'along_mm': 0.4}],
    }]
    whole = _stats(rows, False)
    cut = _stats(rows, True)
    assert whole['reference'] == 1 and whole['recall'] == 1
    assert whole['on_hair_mm']['p95'] == 0.0
    assert whole['along_hair_mm']['p95'] == 0.4
    assert cut['reference'] == 1 and cut['recall'] == 0 and cut['extra'] == 1
    assert _touches([(0, 3), (1, 4), (2, 3)], 20, 20)
    assert not _touches([(1, 3), (1, 4), (2, 3)], 20, 20)
