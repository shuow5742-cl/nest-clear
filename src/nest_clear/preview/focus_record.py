"""Persist focus records without requiring image analysis libraries."""
from datetime import datetime
import json
import uuid
from pathlib import Path


def write_record(root, data, image_bytes, note):
    now = datetime.now().astimezone()
    directory = Path(root) / 'focus_records' / now.date().isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    name = now.strftime('%H%M%S_%f') + '_' + uuid.uuid4().hex[:8] + '.jpg'
    (directory / name).write_bytes(image_bytes)
    record = dict(data, time=now.isoformat(), note=note, thumbnail=name)
    with (directory / 'records.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    return record
