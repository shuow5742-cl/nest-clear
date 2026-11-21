from nest_clear.tooloffset.commands import CommandReader, send


class Events:
    def __init__(self): self.rows = []
    def emit(self, event, **data): self.rows.append((event, data))


def test_command_deduplicates_and_rejects_unexpected(tmp_path):
    events = Events()
    reader = CommandReader(tmp_path, events)
    seq = send(tmp_path, 'tool_in_place')
    assert reader.poll({'tool_in_place'}) == ('tool_in_place', {})
    assert reader.poll({'tool_in_place'}) is None
    assert events.rows[-1][1]['seq'] == seq
    send(tmp_path, 'abort')
    assert reader.poll({'tool_in_place'}) is None
    assert not events.rows[-1][1]['accepted']
