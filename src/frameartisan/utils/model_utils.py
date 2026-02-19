import hashlib
import io
import logging
import os

import xxhash


logger = logging.getLogger(__name__)

# Known wrapper prefixes that should be stripped for canonical tensor key comparison.
_TENSOR_KEY_PREFIXES = ("model.", "encoder.model.", "text_model.")

# Config fields that differ only due to wrapper metadata and should be ignored
# when comparing component configs for identity.
_CONFIG_IGNORE_FIELDS = frozenset({"architectures", "_name_or_path", "model_type", "auto_map", "torch_dtype"})


def calculate_file_hash(filepath):
    partial_hash = calculate_partial_file_hash(filepath)
    xxhash_hash = calculate_file_hash_xxhash(filepath)
    return f"{partial_hash}-{xxhash_hash}"


def calculate_file_hash_xxhash(filepath):
    hasher = xxhash.xxh64()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(io.DEFAULT_BUFFER_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_partial_file_hash(filepath, chunk_size=io.DEFAULT_BUFFER_SIZE, offset=0):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        file.seek(offset)
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            break  # Only read one chunk
    return hasher.hexdigest()


def _normalize_tensor_key(key: str) -> str:
    """Strip known wrapper prefixes for canonical comparison."""
    for prefix in _TENSOR_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _tensor_to_bytes(tensor) -> bytes:
    """Convert a torch tensor to raw bytes, handling BFloat16 and other dtypes."""
    import torch

    t = tensor.contiguous()
    if t.dim() == 0:
        t = t.unsqueeze(0)
    return t.view(torch.uint8).numpy().tobytes()


def _hash_safetensors_dir(component_dir: str) -> str:
    """Hash safetensors files using canonical tensor key ordering.

    Reads tensor metadata (key names, shapes, dtypes) and tensor data bytes
    through memory-mapped access. Keys are normalized to strip wrapper prefixes
    so that e.g. Qwen3ForCausalLM and Qwen3Model produce the same hash.
    """
    from safetensors import safe_open

    shard_files = sorted(f for f in os.listdir(component_dir) if f.endswith(".safetensors"))

    if not shard_files:
        return _hash_directory_contents(component_dir)

    # First pass: collect metadata only (no tensor data reads)
    # tensor_entries: norm_key -> (shard_file, orig_key, shape_str, dtype_str)
    tensor_entries: dict[str, tuple[str, str, str, str]] = {}

    for shard_file in shard_files:
        shard_path = os.path.join(component_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                norm_key = _normalize_tensor_key(key)
                tensor = f.get_tensor(key)
                tensor_entries[norm_key] = (
                    shard_file,
                    key,
                    str(list(tensor.shape)),
                    str(tensor.dtype),
                )

    # Hash in canonical (sorted normalized key) order
    hasher = xxhash.xxh64()

    for norm_key in sorted(tensor_entries.keys()):
        shard_file, orig_key, shape_str, dtype_str = tensor_entries[norm_key]
        # Feed key identity
        hasher.update(norm_key.encode("utf-8"))
        hasher.update(shape_str.encode("utf-8"))
        hasher.update(dtype_str.encode("utf-8"))

        # Feed tensor data bytes
        shard_path = os.path.join(component_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(orig_key)
            hasher.update(_tensor_to_bytes(tensor))

    return hasher.hexdigest()


def _hash_directory_contents(component_dir: str) -> str:
    """Hash all files in a directory by their content (for tokenizer dirs etc)."""
    hasher = xxhash.xxh64()
    files = sorted(f for f in os.listdir(component_dir) if os.path.isfile(os.path.join(component_dir, f)))
    for filename in files:
        filepath = os.path.join(component_dir, filename)
        hasher.update(filename.encode("utf-8"))
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(io.DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
    return hasher.hexdigest()


def calculate_component_hash(component_dir: str) -> str:
    """Canonical hash of a model component directory.

    For safetensors: normalizes tensor keys (strips 'model.' prefix etc),
    sorts canonically, hashes metadata + tensor data bytes.

    For non-safetensors (tokenizer files etc): hashes file contents directly.
    """
    if not os.path.isdir(component_dir):
        raise ValueError(f"Component directory does not exist: {component_dir}")

    has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(component_dir))

    if has_safetensors:
        return _hash_safetensors_dir(component_dir)
    else:
        return _hash_directory_contents(component_dir)


def calculate_structural_hash(component_dir: str) -> str:
    """Quick structural hash using only safetensors headers (tensor names + shapes + dtypes).

    Much faster than calculate_component_hash() since it doesn't read tensor data.
    Useful for preliminary matching / compatibility checks.
    """
    from safetensors import safe_open

    shard_files = sorted(f for f in os.listdir(component_dir) if f.endswith(".safetensors"))

    if not shard_files:
        # For non-safetensors dirs, fall back to full content hash
        return _hash_directory_contents(component_dir)

    hasher = xxhash.xxh64()
    entries: list[tuple[str, list[int], str]] = []

    for shard_file in shard_files:
        shard_path = os.path.join(component_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                norm_key = _normalize_tensor_key(key)
                entries.append((norm_key, list(tensor.shape), str(tensor.dtype)))

    for norm_key, shape, dtype in sorted(entries, key=lambda e: e[0]):
        hasher.update(norm_key.encode("utf-8"))
        hasher.update(str(shape).encode("utf-8"))
        hasher.update(dtype.encode("utf-8"))

    return hasher.hexdigest()
