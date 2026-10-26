"""Wire protocol: length-prefixed frames with a JSON header and an optional tensor.

Frame layout (all integers big-endian):

    u32 total_len | u32 header_len | header (JSON, utf-8) | payload (raw tensor bytes)

``total_len`` counts everything after itself. The header is a JSON object with
at least a ``type`` key. When a tensor rides along, the header also carries
``dtype`` (numpy dtype string, usually ``float16``) and ``shape`` (list of ints),
and the payload holds the array's C-contiguous bytes.

Message types (``type`` field):

control (server <-> host)
    hello, welcome, profile, profile_result, probe_link, probe_result,
    load, loaded, start, feed, outputs, stats, pause, resume, drain, drained,
    heartbeat, error, bye
data (host <-> host, around the ring)
    peer_hello, hidden (carries a tensor), tokens, link_probe, link_probe_ack
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

HEADER_STRUCT = struct.Struct("!II")
MAX_FRAME = 1 << 31  # 2 GB; a sanity bound, not a design limit
TENSOR_DTYPE = np.float16


@dataclass
class Frame:
    """One message: a JSON-able header plus an optional numpy tensor."""

    type: str
    body: dict[str, Any] = field(default_factory=dict)
    tensor: Optional[np.ndarray] = None

    def __getitem__(self, key: str) -> Any:
        return self.body[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.body.get(key, default)

    def encode(self) -> bytes:
        header = dict(self.body)
        header["type"] = self.type
        payload = b""
        if self.tensor is not None:
            arr = np.ascontiguousarray(self.tensor)
            header["dtype"] = str(arr.dtype)
            header["shape"] = list(arr.shape)
            payload = arr.tobytes()
        hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        total = 4 + len(hbytes) + len(payload)
        if total > MAX_FRAME:
            raise ValueError(f"frame too large: {total} bytes")
        return HEADER_STRUCT.pack(total, len(hbytes)) + hbytes + payload

    @classmethod
    def decode(cls, data: bytes | memoryview) -> "Frame":
        """Decode the bytes after the ``total_len`` prefix (i.e. header_len onward)."""
        mv = memoryview(data)
        (hlen,) = struct.unpack("!I", mv[:4])
        header = json.loads(bytes(mv[4 : 4 + hlen]).decode("utf-8"))
        payload = mv[4 + hlen :]
        typ = header.pop("type")
        tensor = None
        if "dtype" in header and "shape" in header:
            dtype = np.dtype(header.pop("dtype"))
            shape = tuple(header.pop("shape"))
            tensor = np.frombuffer(payload, dtype=dtype).reshape(shape)
        elif len(payload):
            raise ValueError("payload without tensor metadata")
        return cls(typ, header, tensor)

    def nbytes(self) -> int:
        return len(self.encode())


def pack(type: str, tensor: Optional[np.ndarray] = None, **body: Any) -> Frame:
    return Frame(type, body, tensor)


class FrameParser:
    """Incremental parser for a byte stream of frames (for non-asyncio uses)."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self._buf.extend(data)
        out: list[Frame] = []
        while True:
            if len(self._buf) < 4:
                break
            (total,) = struct.unpack("!I", self._buf[:4])
            if len(self._buf) < 4 + total:
                break
            chunk = bytes(self._buf[4 : 4 + total])
            del self._buf[: 4 + total]
            out.append(Frame.decode(chunk))
        return out


async def read_frame(reader) -> Optional[Frame]:
    """Read one frame from an asyncio StreamReader; None on clean EOF."""
    try:
        prefix = await reader.readexactly(4)
    except Exception:
        return None
    (total,) = struct.unpack("!I", prefix)
    try:
        data = await reader.readexactly(total)
    except Exception:
        return None
    return Frame.decode(data)


async def write_frame(writer, frame: Frame) -> int:
    data = frame.encode()
    writer.write(data)
    await writer.drain()
    return len(data)
