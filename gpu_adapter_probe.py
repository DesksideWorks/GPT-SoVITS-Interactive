"""Identify a torch-directml allocation by its Windows adapter LUID."""
import ctypes
import os
import re
import time
from ctypes import wintypes as w
from functools import lru_cache


class Value(ctypes.Structure):
    _fields_ = [('status', w.DWORD), ('value', ctypes.c_double)]


class Item(ctypes.Structure):
    _fields_ = [('name', ctypes.c_wchar_p), ('formatted', Value)]


@lru_cache(maxsize=8)
def torch_adapter_luid(device):
    import torch
    pdh = ctypes.WinDLL('pdh.dll')
    pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(w.HANDLE)]
    pdh.PdhAddEnglishCounterW.argtypes = [w.HANDLE, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(w.HANDLE)]
    pdh.PdhCollectQueryData.argtypes = [w.HANDLE]
    pdh.PdhGetFormattedCounterArrayW.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.POINTER(w.DWORD), ctypes.c_void_p]
    pdh.PdhCloseQuery.argtypes = [w.HANDLE]
    query = w.HANDLE()
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
        raise RuntimeError('GPU adapter identity counters unavailable')
    counters = []
    try:
        for kind in ('Dedicated Usage', 'Shared Usage', 'Running Time'):
            counter = w.HANDLE()
            path = ('\\GPU Engine(*)\\' if kind == 'Running Time' else '\\GPU Process Memory(*)\\')+kind
            if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) != 0:
                raise RuntimeError('GPU adapter memory counter unavailable')
            counters.append((kind,counter))
        def read():
            pdh.PdhCollectQueryData(query)
            result = {}
            for kind,counter in counters:
                size = w.DWORD(); count = w.DWORD()
                pdh.PdhGetFormattedCounterArrayW(counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
                if not size.value:
                    continue
                buffer = ctypes.create_string_buffer(size.value)
                if pdh.PdhGetFormattedCounterArrayW(counter, 0x200, ctypes.byref(size), ctypes.byref(count), buffer) != 0:
                    continue
                items = ctypes.cast(buffer, ctypes.POINTER(Item))
                for i in range(count.value):
                    name = (items[i].name or '').lower()
                    match = re.search(r'luid_0x([0-9a-f]+)_0x([0-9a-f]+)', name)
                    if f'pid_{os.getpid()}_' in name and match and items[i].formatted.status in (0, 1):
                        luid = f'{int(match[1],16):08x}:{int(match[2],16):08x}'
                        if kind == 'Running Time':
                            luid = 'engine:'+luid
                        result[luid] = result.get(luid, 0) + items[i].formatted.value
            return result
        before = read()
        allocation = torch.ones(16 * 1024 * 1024, device=device)
        allocation[0].cpu().item()
        for _ in range(10):
            after = read()
            changed = [luid for luid, value in after.items() if not luid.startswith('engine:') and value-before.get(luid, 0) >= 32*1024**2]
            if len(changed) == 1:
                return changed[0]
            busy = [luid[7:] for luid,value in after.items() if luid.startswith('engine:') and value-before.get(luid,0)>0]
            if len(busy) == 1:
                return busy[0]
            allocation.add_(1)
            allocation[0].cpu().item()
            time.sleep(.05)
        raise RuntimeError('Cannot uniquely identify torch-directml adapter allocation')
    finally:
        pdh.PdhCloseQuery(query)
