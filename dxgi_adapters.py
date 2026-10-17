"""Match torch-directml adapters to ORT's DXGI EnumAdapters1 indices."""
from __future__ import annotations

import ctypes
import uuid
from ctypes import wintypes as w


class LUID(ctypes.Structure):
    _fields_ = [("low", w.DWORD), ("high", w.LONG)]


class AdapterDesc(ctypes.Structure):
    _fields_ = [("name", w.WCHAR * 128), ("vendor", w.UINT), ("device", w.UINT),
                ("subsystem", w.UINT), ("revision", w.UINT),
                ("dedicated_video", ctypes.c_size_t), ("dedicated_system", ctypes.c_size_t),
                ("shared_system", ctypes.c_size_t), ("luid", LUID), ("flags", w.UINT)]


def _method(pointer, slot, result, *args):
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[slot])


def enumerate_adapters():
    dxgi = ctypes.WinDLL("dxgi.dll")
    d3d12 = ctypes.WinDLL("d3d12.dll")
    device_guid = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID("189819f1-1db6-4b57-be54-1821339b85f7").bytes_le)
    check_device = d3d12.D3D12CreateDevice
    check_device.argtypes = [ctypes.c_void_p, w.UINT, ctypes.c_void_p, ctypes.c_void_p]
    check_device.restype = ctypes.c_long
    guid = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    factory = ctypes.c_void_p()
    create = dxgi.CreateDXGIFactory1
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(guid), ctypes.byref(factory)) < 0:
        raise RuntimeError("Cannot enumerate DXGI adapters")
    result = []
    try:
        for index in range(64):
            adapter = ctypes.c_void_p()
            status = _method(factory, 12, ctypes.c_long, w.UINT, ctypes.POINTER(ctypes.c_void_p))(
                factory, index, ctypes.byref(adapter))
            if status & 0xffffffff == 0x887a0002:
                break
            if status < 0:
                raise RuntimeError(f"DXGI EnumAdapters1 failed: {status:#x}")
            try:
                desc = AdapterDesc()
                if _method(adapter, 10, ctypes.c_long, ctypes.POINTER(AdapterDesc))(adapter, ctypes.byref(desc)) < 0:
                    raise RuntimeError("DXGI GetDesc1 failed")
                result.append(dict(index=index, name=desc.name, vendor_id=desc.vendor,
                                   luid=f"{desc.luid.high & 0xffffffff:08x}:{desc.luid.low:08x}",
                                   dedicated_bytes=desc.dedicated_video, software=bool(desc.flags & 2),
                                   d3d12=check_device(adapter, 0xb000, ctypes.byref(device_guid), None) >= 0))
            finally:
                _method(adapter, 2, w.ULONG)(adapter)
    finally:
        _method(factory, 2, w.ULONG)(factory)
    return result


def match_adapter(torch_name, adapters=None, torch_device=None):
    def normalized(name):
        return " ".join(name.replace("\x00", "").lower().split())
    adapters = enumerate_adapters() if adapters is None else adapters
    matches = [a for a in adapters if not a['software'] and a.get('d3d12', True) and normalized(a['name']) == normalized(torch_name)]
    if len(matches) > 1 and torch_device is not None:
        from gpu_adapter_probe import torch_adapter_luid
        luid = torch_adapter_luid(str(torch_device))
        matches = [a for a in matches if a['luid'] == luid]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot uniquely match DirectML adapter {torch_name!r} to DXGI: {adapters}")
    return matches[0]
