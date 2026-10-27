"""asyncio connections carrying :mod:`slugai.net.protocol` frames.

A :class:`Connection` wraps a StreamReader/StreamWriter pair, counts bytes in
both directions, and answers heartbeats transparently. Control connections
(server <-> host) send a heartbeat every few seconds so a vanished peer is
noticed; data connections around the ring are busy enough not to need them.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from typing import Optional

from .protocol import Frame, pack, read_frame, write_frame

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 3.0
HEARTBEAT_TIMEOUT = 20.0


class ConnectionClosed(Exception):
    pass


class Connection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, name: str = ""):
        self.reader = reader
        self.writer = writer
        self.name = name
        self.bytes_sent = 0
        self.last_seen = time.monotonic()
        self._send_lock = asyncio.Lock()
        self._hb_task: Optional[asyncio.Task] = None
        self.closed = False
        try:
            sock = writer.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    @property
    def peer(self) -> str:
        try:
            host, port = self.writer.get_extra_info("peername")[:2]
            return f"{host}:{port}"
        except Exception:
            return "?"

    async def send(self, frame: Frame) -> int:
        if self.closed:
            raise ConnectionClosed(self.name)
        async with self._send_lock:
            try:
                n = await write_frame(self.writer, frame)
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                self.closed = True
                raise ConnectionClosed(f"{self.name}: {e}") from e
        self.bytes_sent += n
        return n

    async def recv(self) -> Frame:
        """Next non-heartbeat frame; raises ConnectionClosed on EOF."""
        while True:
            f = await read_frame(self.reader)
            if f is None:
                self.closed = True
                raise ConnectionClosed(self.name)
            self.last_seen = time.monotonic()
            if f.type == "heartbeat":
                continue
            return f

    def start_heartbeat(self, interval: Optional[float] = None) -> None:
        if self._hb_task is None:
            self._hb_task = asyncio.create_task(self._heartbeat(interval or HEARTBEAT_INTERVAL))

    async def _heartbeat(self, interval: float) -> None:
        try:
            while not self.closed:
                await asyncio.sleep(interval)
                try:
                    await self.send(pack("heartbeat"))
                except ConnectionClosed:
                    return
                if time.monotonic() - self.last_seen > HEARTBEAT_TIMEOUT:
                    log.warning("%s: no traffic for %.0fs, closing", self.name, HEARTBEAT_TIMEOUT)
                    await self.close()
                    return
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        self.closed = True
        if self._hb_task is not None:
            self._hb_task.cancel()
            self._hb_task = None
        try:
            self.writer.close()
            await asyncio.wait_for(self.writer.wait_closed(), 2.0)
        except Exception:
            pass


async def connect(host: str, port: int, name: str = "", timeout: float = 10.0) -> Connection:
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port, limit=1 << 26), timeout)
    return Connection(reader, writer, name or f"{host}:{port}")


async def serve(handler, host: str = "0.0.0.0", port: int = 0) -> asyncio.base_events.Server:
    """Start a TCP server; ``handler(conn)`` is awaited per connection."""

    async def on_conn(reader, writer):
        conn = Connection(reader, writer)
        conn.name = conn.peer
        try:
            await handler(conn)
        except ConnectionClosed:
            pass
        except Exception:
            log.exception("connection handler failed for %s", conn.peer)
        finally:
            await conn.close()

    return await asyncio.start_server(on_conn, host, port, limit=1 << 26)


def local_ip(probe: str = "8.8.8.8") -> str:
    """Best-effort LAN address of this machine (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((probe, 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
