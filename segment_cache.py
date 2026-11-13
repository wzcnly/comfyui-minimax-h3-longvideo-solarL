"""Atomic, tensor-only segment storage. No arbitrary object deserialization."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import inspect
import platform
import functools

import torch

SESSION = uuid.uuid4().hex
_RUNTIME_OBJECTS = {}  # Bounded identity registry; eviction conservatively invalidates.


def exclusive_task(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        directory = kwargs.get('cache_directory', '')
        if not directory:
            return fn(*args, **kwargs)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        with open(root / '.task.lock', 'a+b') as lock:
            lock.seek(0)
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b'0')
            lock.flush()
            lock.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('该缓存任务正在运行，请为并行任务选择不同目录') from exc
            try:
                manifest = root / 'manifest.json'
                data = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {'version':2,'stages':{}}
                for entry in data.get('stages', {}).values():
                    entry['status'] = 'pending_validation'
                data['task'] = {'state':'running', 'sampler':type(args[0]).__name__,
                    'segment_plan':kwargs.get('segment_plan'), 'target_frames':kwargs.get('target_frames'),
                    'segment_frames':kwargs.get('segment_frames'), 'seed':kwargs.get('seed')}
                atomic_json(manifest, data)
                result = fn(*args, **kwargs)
                data = json.loads(manifest.read_text(encoding='utf-8'))
                data['task']['state'] = 'complete'
                atomic_json(manifest, data)
                return result
            except BaseException as exc:
                if (root / 'manifest.json').exists():
                    data = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
                    data.setdefault('task', {})['state'] = 'stopped'
                    data['task']['error_type'] = type(exc).__name__
                    atomic_json(root / 'manifest.json', data)
                raise
            finally:
                lock.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock, fcntl.LOCK_UN)
    return run


def fingerprint(value):
    """Opaque runtime objects deliberately invalidate after engine restart."""
    h = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            t = item.detach().cpu().contiguous()
            h.update(str((t.shape, t.dtype)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif getattr(item, 'is_nested', False):
            for part in item.unbind(): add(part)
        elif isinstance(item, dict):
            h.update(b'{')
            for key in sorted(item): add(key); add(item[key])
            h.update(b'}')
        elif isinstance(item, (list, tuple)):
            h.update(str(type(item)).encode())
            for part in item: add(part)
            h.update(b']')
        elif item is None or isinstance(item, (str, int, float, bool)):
            h.update(repr((type(item).__name__, item)).encode())
        else:
            if id(item) not in _RUNTIME_OBJECTS:
                if len(_RUNTIME_OBJECTS) >= 64:
                    _RUNTIME_OBJECTS.pop(next(iter(_RUNTIME_OBJECTS)))
                _RUNTIME_OBJECTS[id(item)] = (item, uuid.uuid4().hex)
            h.update(f'{SESSION}:{type(item)}:{_RUNTIME_OBJECTS[id(item)][1]}'.encode())
            # Comfy model patcher exposes a UUID that changes with patches.
            h.update(str(getattr(item, 'patches_uuid', '')).encode())
    add(value)
    return h.hexdigest()


def workflow_identity(prompt, unique_id, ports):
    """Fingerprint upstream graph, node source and resolved local model files."""
    if not prompt or str(unique_id) not in prompt:
        return None
    import nodes
    import folder_paths
    visited, graph, files = set(), {}, {}
    def visit(node_id):
        node_id = str(node_id)
        if node_id in visited:
            return
        visited.add(node_id)
        row = prompt[node_id]
        cls = nodes.NODE_CLASS_MAPPINGS[row['class_type']]
        source = inspect.getsourcefile(cls)
        if source and Path(source).is_file():
            files[str(Path(source).resolve())] = digest_file(source)
        graph[node_id] = row
        for value in row.get('inputs', {}).values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
                visit(value[0])
            elif isinstance(value, str):
                for roots, _ in folder_paths.folder_names_and_paths.values():
                    for root in roots:
                        candidate = Path(root) / value
                        try:
                            if candidate.is_file():
                                resolved = str(candidate.resolve())
                                if resolved not in files:
                                    files[resolved] = digest_file(candidate)
                        except OSError:
                            continue  # Literal prompts and URLs need not be paths.
    root = prompt[str(unique_id)]['inputs']
    for port in ports:
        value = root.get(port)
        if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
            visit(value[0])
    code = {str(p.relative_to(Path(__file__).parent)): digest_file(p)
            for p in Path(__file__).parent.rglob('*.py') if 'tests' not in p.parts}
    return fingerprint([graph, files, code, platform.python_version(), torch.__version__])


def atomic_json(path, value):
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with open(pending, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def cached_sample(directory, sample, *, model, cond, negative, latent,
                  sampler, sigmas, seed, cfg, stage, identity=None, index=0, details=None, mask=None):
    def execute():
        args = (model, cond, negative, latent, sampler, sigmas, seed, cfg)
        return sample(*args, mask=mask) if mask is not None else sample(*args)
    if not directory:
        return execute(), False
    from comfy.nested_tensor import NestedTensor
    runtime = [model, sampler] if identity is None else identity
    signature = fingerprint([stage, runtime, cond, negative, latent, sigmas, seed, cfg, mask, details])
    cache = SegmentCache(directory)
    manifest_path = cache.directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'version':2, 'stages':{}}
    if manifest.get('version') != 2:
        raise ValueError('任务清单版本不兼容，请使用新缓存目录')
    entry = {'signature':signature, 'identity':identity, 'stage':stage, 'index':index,
             'details':details or {}, 'status':'checking'}
    slot = f'{stage}:{index}'
    manifest['stages'][slot] = entry
    data = cache.load(signature)
    if data is not None:
        result = {'samples': NestedTensor((data['video'], data['audio']))}
        if 'clean_video' in data:
            result['_denoised_samples'] = NestedTensor((data['clean_video'], data['clean_audio']))
        entry['status'] = 'reused'
        atomic_json(manifest_path, manifest)
        return result, True
    entry['status'] = 'sampling'
    atomic_json(manifest_path, manifest)
    try:
        result = execute()
    except BaseException as exc:
        entry['status'] = 'interrupted' if not isinstance(exc, Exception) else 'failed'
        entry['error_type'] = type(exc).__name__
        atomic_json(manifest_path, manifest)
        raise
    video, audio = result['samples'].unbind()
    data = {'video': video, 'audio': audio}
    if '_denoised_samples' in result:
        data['clean_video'], data['clean_audio'] = result['_denoised_samples'].unbind()
    cache.save(signature, data)
    entry['status'] = 'complete'
    atomic_json(manifest_path, manifest)
    return result, False


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class SegmentCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, signature, tensors):
        if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in tensors.items()):
            raise TypeError('缓存只接受命名Tensor')
        key = hashlib.sha256(signature.encode('utf-8')).hexdigest()
        token = uuid.uuid4().hex
        data = self.directory / f'{key}.{token}.pt'
        temporary = data.with_suffix('.tmp')
        try:
            with open(temporary, 'wb') as output:
                torch.save({k: v.detach().cpu() for k, v in tensors.items()}, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, data)
            metadata = {'version': 1, 'signature': signature, 'file': data.name,
                        'sha256': digest_file(data)}
            pending = self.directory / f'{key}.{token}.json.tmp'
            with open(pending, 'w', encoding='utf-8') as output:
                json.dump(metadata, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(pending, self.directory / f'{key}.json')
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, signature):
        key = hashlib.sha256(signature.encode('utf-8')).hexdigest()
        manifest = self.directory / f'{key}.json'
        if not manifest.exists():
            return None
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('version') != 1 or metadata.get('signature') != signature:
            raise ValueError('缓存版本或签名不匹配')
        filename = metadata.get('file', '')
        if Path(filename).name != filename or not filename.startswith(key+'.'):
            raise ValueError('缓存文件路径非法')
        data = self.directory / filename
        if digest_file(data) != metadata.get('sha256'):
            raise ValueError('片段缓存完整性校验失败')
        tensors = torch.load(data, map_location='cpu', weights_only=True)
        if not isinstance(tensors, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in tensors.items()):
            raise ValueError('片段缓存数据类型非法')
        return tensors
