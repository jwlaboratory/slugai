"""Bonjour discovery of hosts (zeroconf ``_slugai._tcp.local.``).

Hosts advertise themselves; the server browses until it has enough of them or
a deadline passes. ``--hosts a:port,b:port`` on the server skips all of this.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass
from typing import Optional

from .transport import local_ip

log = logging.getLogger(__name__)
SERVICE_TYPE = "_slugai._tcp.local."
DEFAULT_PORT = 47111


@dataclass(frozen=True)
class HostAddr:
    name: str
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.name}@{self.host}:{self.port}"


class Advertiser:
    """Register this host with Bonjour; ``async with`` or start()/stop()."""

    def __init__(self, name: str, port: int, properties: Optional[dict] = None, ip: Optional[str] = None):
        self.name = name
        self.port = port
        self.properties = properties or {}
        self.ip = ip or local_ip()
        self._zc = None
        self._info = None

    async def start(self) -> None:
        from zeroconf import ServiceInfo
        from zeroconf.asyncio import AsyncZeroconf

        self._zc = AsyncZeroconf()
        props = {k: str(v) for k, v in self.properties.items()}
        props["name"] = self.name
        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"{self.name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(self.ip)],
            port=self.port,
            properties=props,
            server=f"{self.name}.local.",
        )
        await self._zc.async_register_service(self._info)
        log.info("advertising %s at %s:%d", self.name, self.ip, self.port)

    async def stop(self) -> None:
        if self._zc is not None:
            try:
                await self._zc.async_unregister_service(self._info)
                await self._zc.async_close()
            except Exception:
                pass
            self._zc = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()


async def browse(timeout: float = 30.0, min_hosts: int = 1, settle: float = 2.0) -> list[HostAddr]:
    """Collect advertised hosts until ``min_hosts`` are seen (then wait ``settle`` s more) or ``timeout``."""
    from zeroconf import ServiceBrowser, ServiceListener
    from zeroconf.asyncio import AsyncZeroconf

    found: dict[str, HostAddr] = {}
    zc = AsyncZeroconf()
    loop = asyncio.get_running_loop()

    class Listener(ServiceListener):
        def add_service(self, zc_, type_, name):
            asyncio.run_coroutine_threadsafe(resolve(name), loop)

        def update_service(self, zc_, type_, name):
            asyncio.run_coroutine_threadsafe(resolve(name), loop)

        def remove_service(self, zc_, type_, name):
            found.pop(name, None)

    async def resolve(name: str):
        info = await zc.async_get_service_info(SERVICE_TYPE, name, timeout=3000)
        if info is None or not info.addresses:
            return
        host = socket.inet_ntoa(info.addresses[0])
        props = {k.decode(): (v.decode() if isinstance(v, bytes) else v) for k, v in (info.properties or {}).items()}
        hname = props.get("name") or name.split(".")[0]
        found[name] = HostAddr(hname, host, info.port)
        log.info("found host %s", found[name])

    browser = ServiceBrowser(zc.zeroconf, SERVICE_TYPE, Listener())
    deadline = time.monotonic() + timeout
    try:
        reached_at: Optional[float] = None
        while time.monotonic() < deadline:
            if len(found) >= min_hosts:
                if reached_at is None:
                    reached_at = time.monotonic()
                elif time.monotonic() - reached_at >= settle:
                    break
            await asyncio.sleep(0.2)
    finally:
        browser.cancel()
        await zc.async_close()
    return sorted(found.values(), key=lambda h: h.name)


def parse_hosts(spec: str, default_port: int = DEFAULT_PORT) -> list[HostAddr]:
    out = []
    for i, item in enumerate(x.strip() for x in spec.split(",") if x.strip()):
        if ":" in item:
            h, p = item.rsplit(":", 1)
            out.append(HostAddr(f"host{i}", h, int(p)))
        else:
            out.append(HostAddr(f"host{i}", item, default_port))
    return out
