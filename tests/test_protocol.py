import numpy as np
import pytest

from slugai.net.protocol import Frame, FrameParser, pack


def test_roundtrip_no_tensor():
    f = pack("hello", name="mac-a", port=1234, nested={"a": [1, 2]})
    g = Frame.decode(f.encode()[4:])
    assert g.type == "hello"
    assert g.body == {"name": "mac-a", "port": 1234, "nested": {"a": [1, 2]}}
    assert g.tensor is None


def test_roundtrip_tensor_fp16():
    t = (np.random.randn(4, 7, 16) * 3).astype(np.float16)
    f = pack("hidden", tensor=t, slot=2, phase="decode")
    g = Frame.decode(f.encode()[4:])
    assert g.type == "hidden"
    assert g["slot"] == 2 and g["phase"] == "decode"
    assert g.tensor.dtype == np.float16 and g.tensor.shape == (4, 7, 16)
    np.testing.assert_array_equal(g.tensor, t)
    # dtype/shape never leak into body
    assert "dtype" not in g.body and "shape" not in g.body


def test_roundtrip_int_tensor():
    t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    g = Frame.decode(pack("tokens", tensor=t).encode()[4:])
    np.testing.assert_array_equal(g.tensor, t)


def test_parser_handles_split_and_concatenated_frames():
    frames = [pack("a", i=i, tensor=np.ones((i + 1,), np.float16)) for i in range(5)]
    blob = b"".join(f.encode() for f in frames)
    parser = FrameParser()
    out = []
    # feed in awkward chunks
    for i in range(0, len(blob), 7):
        out.extend(parser.feed(blob[i : i + 7]))
    assert [f["i"] for f in out] == [0, 1, 2, 3, 4]
    assert out[4].tensor.shape == (5,)


def test_payload_without_metadata_rejected():
    bad = Frame("x").encode()[4:] + b"junk"
    with pytest.raises(ValueError):
        Frame.decode(bad)


@pytest.mark.asyncio
async def test_asyncio_stream_roundtrip():
    import asyncio
    from slugai.net.protocol import read_frame, write_frame

    got = []

    async def handle(reader, writer):
        while (f := await read_frame(reader)) is not None:
            got.append(f)
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    big = np.zeros((64, 1, 4096), np.float16)
    await write_frame(writer, pack("hidden", tensor=big, slot=0))
    await write_frame(writer, pack("bye"))
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    server.close()
    await server.wait_closed()
    assert [f.type for f in got] == ["hidden", "bye"]
    assert got[0].tensor.shape == (64, 1, 4096)
