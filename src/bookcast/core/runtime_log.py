from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Callable, Iterator


LogSink = Callable[[str], None]
_sink_lock = threading.RLock()
_sinks: list[LogSink] = []


def _snapshot_sinks() -> list[LogSink]:
    with _sink_lock:
        return list(_sinks)


def _emit_to_sinks(text: str) -> None:
    if not text:
        return
    for sink in _snapshot_sinks():
        sink(text)


def register_sink(sink: LogSink) -> None:
    with _sink_lock:
        if sink not in _sinks:
            _sinks.append(sink)


def unregister_sink(sink: LogSink) -> None:
    with _sink_lock:
        if sink in _sinks:
            _sinks.remove(sink)


@contextmanager
def use_sink(sink: LogSink) -> Iterator[None]:
    register_sink(sink)
    try:
        yield
    finally:
        unregister_sink(sink)


def console_print(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file=None,
    flush: bool = True,
) -> None:
    text = sep.join(str(value) for value in values) + end
    target = sys.__stdout__ if file is None else file
    target.write(text)
    if flush and hasattr(target, "flush"):
        target.flush()
    _emit_to_sinks(text)
