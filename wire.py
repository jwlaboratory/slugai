"""Frames between the controller and hosts: a JSON message, optionally with one numpy array.

On the socket a frame is ``u32 header length | u32 data length | JSON header |
array bytes``. The header is the message, plus ``array: [dtype, shape]`` when
an array comes with it.
"""

from __future__ import annotations

import asyncio
import json
import struct

import numpy as np


async def connect(address: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a connection to ``host:port``."""
    host, port = address.rsplit(":", 1)
    return await asyncio.open_connection(host, int(port))


async def send(writer: asyncio.StreamWriter, msg: dict, array: np.ndarray | None = None) -> None:
    data = b""
    if array is not None:
        msg = {**msg, "array": [array.dtype.str, array.shape]}
        data = np.ascontiguousarray(array).tobytes()
    header = json.dumps(msg).encode()
    writer.write(struct.pack("<II", len(header), len(data)) + header + data)
    await writer.drain()


async def recv(reader: asyncio.StreamReader) -> tuple[dict, np.ndarray | None]:
    """The next message and its array (None without one). Raises ``asyncio.IncompleteReadError`` once the peer closes."""
    header_len, data_len = struct.unpack("<II", await reader.readexactly(8))
    msg = json.loads(await reader.readexactly(header_len))
    data = await reader.readexactly(data_len)
    if "array" not in msg:
        return msg, None
    dtype, shape = msg.pop("array")
    return msg, np.frombuffer(data, dtype=dtype).reshape(shape)
