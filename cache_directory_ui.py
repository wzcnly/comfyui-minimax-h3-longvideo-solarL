"""Local-only native directory picker, invoked explicitly by the node button."""
import asyncio
import ipaddress
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlsplit

_picker_lock = asyncio.Lock()
_NATIVE_SCRIPT = r'''
import ctypes as c
from ctypes import wintypes as w
import os
shell = c.WinDLL('shell32')
ole = c.OleDLL('ole32')
ole.OleUninitialize.restype = None
ole.CoTaskMemFree.restype = None
user = c.WinDLL('user32')
ole.OleInitialize(None)
CALLBACK = c.WINFUNCTYPE(c.c_int, w.HWND, w.UINT, w.LPARAM, w.LPARAM)
user.SendMessageW.argtypes = [w.HWND,w.UINT,w.WPARAM,w.LPARAM]
user.SetForegroundWindow.argtypes = [w.HWND]
user.SetWindowTextW.argtypes = [w.HWND,w.LPCWSTR]
user.SetWindowPos.argtypes = [w.HWND,w.HWND,c.c_int,c.c_int,c.c_int,c.c_int,w.UINT]
initial = c.create_unicode_buffer(os.environ.get('SOLARL_PICKER_START',''))
def initialize(hwnd, message, param, data):
    if message == 1:
        user.SetWindowTextW(hwnd, '选择缓存文件夹 - solarL')
        user.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x43)
        if initial.value:
            user.SendMessageW(hwnd, 0x467, 1, c.cast(initial,c.c_void_p).value)
        user.SetForegroundWindow(hwnd)
    return 0
callback = CALLBACK(initialize)
class BrowseInfo(c.Structure):
    _fields_ = [('owner',w.HWND),('root',c.c_void_p),('display',w.LPWSTR),
                ('title',w.LPCWSTR),('flags',w.UINT),('callback',CALLBACK),
                ('data',w.LPARAM),('image',c.c_int)]
display = c.create_unicode_buffer(32768)
info = BrowseInfo(None,None,c.cast(display,w.LPWSTR),'选择缓存文件夹；取消保留原设置',0x51,callback,0,0)
shell.SHBrowseForFolderW.argtypes=[c.POINTER(BrowseInfo)]
shell.SHBrowseForFolderW.restype=c.c_void_p
ole.CoTaskMemFree.argtypes=[c.c_void_p]
pidl = shell.SHBrowseForFolderW(c.byref(info))
try:
    if pidl:
        path = c.create_unicode_buffer(32768)
        shell.SHGetPathFromIDListEx.argtypes=[c.c_void_p,w.LPWSTR,w.DWORD,w.DWORD]
        if not shell.SHGetPathFromIDListEx(pidl,path,len(path),0):
            raise OSError('无法读取所选文件夹路径')
        print(path.value,end='')
finally:
    if pidl: ole.CoTaskMemFree(pidl)
    ole.OleUninitialize()
'''


def is_local_request(remote, origin, host):
    try:
        address = ipaddress.ip_address(remote)
        address = getattr(address, 'ipv4_mapped', None) or address
        if not address.is_loopback:
            return False
    except ValueError:
        return False
    hostname = urlsplit('http://' + host).hostname
    if hostname != 'localhost':
        try:
            if not ipaddress.ip_address(hostname or '').is_loopback:
                return False
        except ValueError:
            return False
    # Reject a browser page on a different origin, including loopback proxies.
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme not in ('http', 'https') or parsed.netloc != host:
            return False
    return True


def writable_directory(value):
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError('所选路径不是文件夹')
    with tempfile.TemporaryFile(dir=path):
        pass
    return str(path)


async def choose_directory(initial):
    if os.name != 'nt':
        raise RuntimeError('系统目录选择目前支持Windows；请手动填写执行服务器目录')
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-c', _NATIVE_SCRIPT,
        env=dict(os.environ, SOLARL_PICKER_START=initial, PYTHONIOENCODING='utf-8'),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        creationflags=0x08000000)  # Hide the helper console; show only the picker.
    try:
        output, error = await asyncio.wait_for(process.communicate(), timeout=180)
        if process.returncode:
            detail = error.decode('utf-8-sig', errors='replace').strip()
            raise RuntimeError('无法打开系统目录窗口：' + (detail or '目录窗口进程启动失败'))
        path = output.decode('utf-8-sig').strip()
        return writable_directory(path) if path else None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def register_routes():
    server = sys.modules.get('server')
    PromptServer = getattr(server, 'PromptServer', None) if server is not None else None
    if PromptServer is None:
        return
    instance = getattr(PromptServer, 'instance', None)
    if instance is None:
        return
    from aiohttp import web
    import folder_paths

    @instance.routes.post('/solarL/cache-directory/choose')
    async def choose(request):
        if not is_local_request(request.remote or '', request.headers.get('Origin'), request.host):
            return web.json_response({'error':'远程连接请手动填写ComfyUI服务器上的目录；系统选择仅支持本机连接。'}, status=403)
        if request.content_type != 'application/json':
            return web.json_response({'error':'请求格式错误'}, status=400)
        if _picker_lock.locked():
            return web.json_response({'error':'目录选择窗口已打开，请先完成或取消。'}, status=409)
        async with _picker_lock:
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError('请求格式错误')
                initial = body.get('initial') or folder_paths.get_output_directory()
                if not isinstance(initial, str):
                    raise ValueError('路径必须是文本')
                path = await choose_directory(initial)
                return web.json_response({'path':path, 'cancelled':path is None})
            except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
                return web.json_response({'error':str(exc) or '目录选择超时，请重试'}, status=400)
