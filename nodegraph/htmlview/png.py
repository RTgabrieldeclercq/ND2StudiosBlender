"""PNG packing for the standalone HTML viewer — numpy and zlib, nothing else.

A browser decodes PNG natively and fast, but a canvas hands back 8 bits per sample, so a
16-bit plane is packed as one opaque GREY image twice the plane's height: the top half holds
the high bytes, the bottom half the low bytes (``value = top * 256 + bottom``), and the
viewer's JavaScript rebuilds the 16-bit values from ``getImageData``. Two stacked halves
rather than the bytes interleaved in an RGB image because PNG's row filter then runs along
a byte plane that is smooth (the high bytes) or merely noisy (the low bytes) instead of
alternating between them — measured on a 1024² 12-bit plane with shot noise: 1.20 MB in 136
ms against 1.51 MB in 315 ms for the interleaved layout (2026-11-05). Opaque and grey on
purpose: a browser premultiplies alpha and would corrupt the bytes of any pixel with alpha
< 255, and no ``gAMA`` / ``sRGB`` / ``iCCP`` chunk is written, so no colour management
touches the samples either.

Rows are filtered with PNG's ``Up`` filter (each byte minus the byte above, modulo 256),
which costs one numpy subtraction. The output is deterministic for a given array and zlib
level, which is what lets a test compare two exports byte for byte.
"""
from __future__ import annotations

import base64
import struct
import zlib
from typing import Tuple

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"
_CT = {"L": 0, "RGB": 2, "RGBA": 6}

#: zlib level: 6 is the knee — level 9 is slower for under a percent, level 3 is a quarter
#: faster for 8 % more bytes on the noisy half (measured 2026-11-05)
LEVEL = 6


def _chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def encode_png(arr: np.ndarray, mode: str = "", level: int = LEVEL) -> bytes:
    """``arr`` (``(H, W)`` uint8 → ``L``; ``(H, W, 3)`` → ``RGB``; ``(H, W, 4)`` → ``RGBA``)
    as PNG bytes, Up-filtered, deflated at ``level``."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8))
    if a.ndim == 2:
        mode = mode or "L"
        a = a[:, :, None]
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        mode = mode or ("RGB" if a.shape[2] == 3 else "RGBA")
    else:
        raise ValueError(f"encode_png: an (H, W), (H, W, 3) or (H, W, 4) uint8 array, not {a.shape}")
    if mode not in _CT or _CT[mode] != {1: 0, 3: 2, 4: 6}[a.shape[2]]:
        raise ValueError(f"encode_png: mode {mode!r} does not fit {a.shape}")
    h, w, n = a.shape
    rows = a.reshape(h, w * n)
    filt = np.empty((h, 1 + w * n), dtype=np.uint8)
    filt[:, 0] = 2                                       # Up
    filt[0, 1:] = rows[0]
    if h > 1:
        np.subtract(rows[1:], rows[:-1], out=filt[1:, 1:])   # uint8 wraps: modulo 256
    ihdr = struct.pack(">IIBBBBB", w, h, 8, _CT[mode], 0, 0, 0)
    return (_SIG + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(filt.tobytes(), level))
            + _chunk(b"IEND", b""))


def decode_png(data: bytes) -> np.ndarray:
    """The inverse (tests and the headless check): the array an ``encode_png`` call made.
    Reads only what this module writes — 8-bit, non-interlaced."""
    if data[:8] != _SIG:
        raise ValueError("not a PNG")
    pos, idat, w = 8, b"", 0
    h, ct = 0, 0
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            w, h, _bits, ct, _c, _f, _i = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
    chans = {0: 1, 2: 3, 6: 4}[ct]
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * chans)
    out = np.empty((h, w * chans), dtype=np.uint8)
    prev = np.zeros(w * chans, dtype=np.uint8)
    for y in range(h):
        f, row = int(raw[y, 0]), raw[y, 1:]
        if f == 0:
            cur = row
        elif f == 2:
            cur = (row + prev).astype(np.uint8)          # wraps back
        else:
            raise ValueError(f"decode_png: filter {f} is not one this module writes")
        out[y] = cur
        prev = cur
    return out.reshape(h, w) if chans == 1 else out.reshape(h, w, chans)


def pack_u16(plane: np.ndarray) -> np.ndarray:
    """A ``(H, W)`` uint16 plane as the ``(2H, W)`` uint8 grey image the viewer unpacks:
    the high bytes above the low bytes."""
    a = np.ascontiguousarray(np.asarray(plane, dtype=np.uint16))
    return np.concatenate([(a >> 8).astype(np.uint8), (a & 0xFF).astype(np.uint8)], axis=0)


def unpack_u16(stacked: np.ndarray) -> np.ndarray:
    a = np.asarray(stacked, dtype=np.uint16)
    h = a.shape[0] // 2
    return (a[:h] << 8) | a[h:]


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def plane_png(plane: np.ndarray) -> Tuple[bytes, str]:
    """``(png bytes, encoding)`` for an integer plane: ``"u8"`` (a grey PNG of the plane)
    when every value fits a byte, else ``"u16"`` (the stacked high / low halves)."""
    a = np.asarray(plane)
    if a.dtype == np.uint8 or (a.size and int(a.max()) <= 255 and int(a.min()) >= 0):
        return encode_png(a.astype(np.uint8)), "u8"
    return encode_png(pack_u16(a)), "u16"


__all__ = ["LEVEL", "encode_png", "decode_png", "pack_u16", "unpack_u16", "data_uri",
           "plane_png"]
