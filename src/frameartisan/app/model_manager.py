import contextlib
import contextvars
import gc
import hashlib
import logging
import threading
from collections.abc import Generator
from typing import Any, Literal

import torch


logger = logging.getLogger(__name__)

OFFLOAD_STRATEGIES = ("auto", "no_offload", "model_offload", "sequential_group_offload", "group_offload")


_SCOPED_DEVICE: contextvars.ContextVar[Any] = contextvars.ContextVar("frameartisan_scoped_device", default=None)
_SCOPED_DTYPE: contextvars.ContextVar[Any] = contextvars.ContextVar("frameartisan_scoped_dtype", default=None)


ModelComponent = Literal[
    "tokenizer",
    "text_encoder",
    "transformer",
    "vae",
    "controlnet",
    "preprocessor",
]


class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._components: dict[ModelComponent, Any] = {}
        self._model_id: str | None = None

        self._component_hashes: dict[ModelComponent, str] = {}
        self._lora_sources: dict[str, str] = {}
        self._compiled_components: dict[tuple[str | None, ModelComponent, str, str, str], Any] = {}
        self._attention_backend: str | None = None
        self._component_cache: dict[str, Any] = {}

        self._offload_strategy: str = "auto"
        self._group_offload_use_stream: bool = False
        self._group_offload_low_cpu_mem: bool = False
        self._managed_components: dict[str, Any] = {}
        self._applied_strategy: str | None = None

    @property
    def attention_backend(self) -> str:
        with self._lock:
            return self._attention_backend or "native"

    @attention_backend.setter
    def attention_backend(self, value: str | None) -> None:
        with self._lock:
            self._attention_backend = value if value and value != "native" else None

    @property
    def offload_strategy(self) -> str:
        with self._lock:
            return self._offload_strategy

    @offload_strategy.setter
    def offload_strategy(self, value: str) -> None:
        if value not in OFFLOAD_STRATEGIES:
            raise ValueError(f"Unknown offload strategy {value!r}. Must be one of {OFFLOAD_STRATEGIES}")
        with self._lock:
            self._offload_strategy = value

    @property
    def group_offload_use_stream(self) -> bool:
        with self._lock:
            return self._group_offload_use_stream

    @group_offload_use_stream.setter
    def group_offload_use_stream(self, value: bool) -> None:
        with self._lock:
            self._group_offload_use_stream = bool(value)

    @property
    def group_offload_low_cpu_mem(self) -> bool:
        with self._lock:
            return self._group_offload_low_cpu_mem

    @group_offload_low_cpu_mem.setter
    def group_offload_low_cpu_mem(self, value: bool) -> None:
        with self._lock:
            self._group_offload_low_cpu_mem = bool(value)

    @property
    def applied_strategy(self) -> str | None:
        with self._lock:
            return self._applied_strategy

    def resolve_offload_strategy(self, device: torch.device | str) -> str:
        """Resolve ``"auto"`` to a concrete strategy based on available VRAM."""
        strategy = self.offload_strategy
        if strategy != "auto":
            return strategy

        device = torch.device(device) if isinstance(device, str) else device
        if device.type != "cuda":
            return "group_offload"

        try:
            total_mem = torch.cuda.get_device_properties(device).total_mem
            total_gb = total_mem / (1024**3)
        except Exception:
            return "group_offload"

        if total_gb >= 20:
            return "no_offload"
        if total_gb >= 12:
            return "model_offload"
        if total_gb >= 8:
            return "sequential_group_offload"
        return "group_offload"

    def register_component(self, name: str, module: Any) -> None:
        """Register a named component for lifecycle management.

        If the component differs from the currently registered one, resets
        ``_applied_strategy`` so the next ``apply_offload_strategy`` call
        re-applies placement/hooks to the new module.
        """
        with self._lock:
            existing = self._managed_components.get(name)
            if existing is not module:
                self._applied_strategy = None
            self._managed_components[name] = module

    def get_component(self, name: str) -> Any | None:
        """Retrieve a managed component by name."""
        with self._lock:
            return self._managed_components.get(name)

    @staticmethod
    def remove_offload_hooks(module: Any) -> None:
        """Remove diffusers group-offloading hooks from *module* and all submodules."""
        hook_names = ("group_offloading", "layer_execution_tracker", "lazy_prefetch_group_offloading")
        for submodule in module.modules():
            if hasattr(submodule, "_diffusers_hook"):
                registry = submodule._diffusers_hook
                for hook_name in hook_names:
                    try:
                        registry.remove_hook(hook_name, recurse=False)
                    except Exception:
                        pass

    def prepare_strategy_transition(self, new_strategy: str, device: torch.device | str) -> None:
        """Clean up the old offload strategy before applying *new_strategy*."""
        with self._lock:
            old = self._applied_strategy
            if old == new_strategy:
                return

            if old in ("group_offload", "sequential_group_offload"):
                for name, mod in self._managed_components.items():
                    self.remove_offload_hooks(mod)
                    if hasattr(mod, "to"):
                        mod.to("cpu")
                    logger.debug("Removed offload hooks from %s, moved to CPU", name)
            elif old == "no_offload":
                for name, mod in self._managed_components.items():
                    if hasattr(mod, "to"):
                        mod.to("cpu")
                        logger.debug("Moved %s to CPU (leaving no_offload)", name)
            elif old == "model_offload":
                for name, mod in self._managed_components.items():
                    if hasattr(mod, "to"):
                        mod.to("cpu")
                        logger.debug("Ensured %s on CPU (leaving model_offload)", name)

            self._applied_strategy = new_strategy

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reapply_group_offload(self, component_name: str, device: torch.device | str) -> None:
        """Re-apply group offload hooks to a single component.

        Call this after modifying a component's module structure (e.g. loading
        LoRA adapters) so that new submodules get offload hooks.
        """
        if self._applied_strategy != "group_offload":
            return

        from diffusers.hooks.group_offloading import apply_group_offloading

        with self._lock:
            mod = self._managed_components.get(component_name)
        if mod is None:
            return

        use_stream = self._group_offload_use_stream
        offload_kwargs = dict(
            onload_device=torch.device(device) if isinstance(device, str) else device,
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=use_stream,
        )
        if use_stream and self._group_offload_low_cpu_mem:
            offload_kwargs["low_cpu_mem_usage"] = True

        self.remove_offload_hooks(mod)
        apply_group_offloading(mod, **offload_kwargs)
        logger.info("Re-applied group offload hooks for %s", component_name)

    def apply_offload_strategy(self, device: torch.device | str) -> str:
        """Resolve and apply the current offload strategy to all managed components.

        Returns the resolved (concrete) strategy name.  If the resolved strategy
        is already applied this is a no-op.
        """
        strategy = self.resolve_offload_strategy(device)

        with self._lock:
            if not self._managed_components:
                self._applied_strategy = strategy
                return strategy
            if self._applied_strategy == strategy:
                return strategy

        # Clean up the old strategy before applying the new one
        self.prepare_strategy_transition(strategy, device)

        if strategy == "no_offload":
            with self._lock:
                components = list(self._managed_components.items())
            for name, mod in components:
                if hasattr(mod, "to"):
                    mod.to(device)
                    logger.info("no_offload: moved %s to %s", name, device)
        elif strategy == "group_offload":
            from diffusers.hooks.group_offloading import (
                apply_group_offloading,
            )

            use_stream = self._group_offload_use_stream
            offload_kwargs = dict(
                onload_device=torch.device(device) if isinstance(device, str) else device,
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=use_stream,
            )
            if use_stream and self._group_offload_low_cpu_mem:
                offload_kwargs["low_cpu_mem_usage"] = True
            with self._lock:
                components = list(self._managed_components.items())
            for name, mod in components:
                try:
                    # Remove any existing hooks first to make re-application
                    # idempotent (e.g. when a new component registration resets
                    # _applied_strategy and triggers re-application).
                    self.remove_offload_hooks(mod)
                    apply_group_offloading(mod, **offload_kwargs)
                    logger.info("Group offload enabled for %s", name)
                except Exception as e:
                    logger.warning("Failed to enable group offload for %s: %s", name, e)
        elif strategy == "sequential_group_offload":
            # Keep on CPU with no hooks; use_components() applies hooks on demand
            with self._lock:
                components = list(self._managed_components.items())
            for name, mod in components:
                if hasattr(mod, "to"):
                    mod.to("cpu")
            logger.info("sequential_group_offload: all components on CPU, hooks deferred")
        elif strategy == "model_offload":
            # Keep on CPU; use_components() will handle placement
            logger.info("model_offload: all components remain on CPU")

        return strategy

    @contextlib.contextmanager
    def use_components(
        self,
        *names: str,
        device: torch.device | str,
        strategy_override: str | None = None,
    ) -> Generator[None, None, None]:
        """Context manager that places named components on *device*.

        Behaviour depends on the active offload strategy:
        - ``no_offload`` / ``group_offload``: no-op yield.
        - ``model_offload``: bulk CPU ↔ GPU move.
        - ``sequential_group_offload``: apply group-offload hooks on enter,
          remove hooks + move to CPU on exit.

        *strategy_override* lets callers force a different strategy for these
        components (e.g. ``"model_offload"`` for small models like VAE that
        are too granular for leaf-level hook offloading).
        """
        strategy = self._applied_strategy
        if strategy is None:
            strategy = self.apply_offload_strategy(device)
        if strategy_override is not None:
            strategy = strategy_override

        if strategy in ("no_offload", "group_offload"):
            yield
            return

        modules = []
        with self._lock:
            for n in names:
                mod = self._managed_components.get(n)
                if mod is not None and hasattr(mod, "to"):
                    modules.append((n, mod))

        if strategy == "model_offload":
            for name, mod in modules:
                # Remove any stale group-offload hooks that would block .to()
                self.remove_offload_hooks(mod)
                mod.to(device)
                logger.debug("model_offload: moved %s to %s", name, device)
            try:
                yield
            finally:
                for name, mod in modules:
                    mod.to("cpu")
                    logger.debug("model_offload: moved %s back to CPU", name)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elif strategy == "sequential_group_offload":
            from diffusers.hooks.group_offloading import apply_group_offloading

            use_stream = self._group_offload_use_stream
            offload_kwargs = dict(
                onload_device=torch.device(device) if isinstance(device, str) else device,
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=use_stream,
            )
            if use_stream and self._group_offload_low_cpu_mem:
                offload_kwargs["low_cpu_mem_usage"] = True

            for name, mod in modules:
                try:
                    apply_group_offloading(mod, **offload_kwargs)
                    logger.debug("sequential_group_offload: hooks applied to %s", name)
                except Exception as e:
                    logger.warning("sequential_group_offload: failed to hook %s: %s", name, e)
            try:
                yield
            finally:
                for name, mod in modules:
                    self.remove_offload_hooks(mod)
                    if hasattr(mod, "to"):
                        mod.to("cpu")
                    logger.debug("sequential_group_offload: cleaned up %s", name)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            yield

    def get_available_attention_backends(self) -> list[tuple[str, str]]:
        """Return list of (backend_id, display_name) tuples for available backends.

        Only returns backends that are actually usable on this system.
        Detects both locally installed packages and HuggingFace Hub kernels.
        """
        available: list[tuple[str, str]] = [("native", "Auto (PyTorch SDPA)")]

        if not torch.cuda.is_available():
            return available

        # Check for HuggingFace kernels package (provides hub variants)
        has_kernels = False
        try:
            import kernels  # noqa: F401

            has_kernels = True
        except ImportError:
            pass

        # Check GPU compute capability for Hopper detection
        is_hopper_gpu = False
        try:
            is_hopper_gpu = torch.cuda.get_device_capability()[0] >= 9
        except Exception:
            pass

        # Check for Flash Attention 2 (local)
        has_flash_local = False
        try:
            import flash_attn  # noqa: F401

            has_flash_local = True  # noqa: F841
            available.append(("flash", "Flash Attention 2"))
        except ImportError:
            pass

        # Flash Attention 2 from Hub (if kernels installed)
        if has_kernels:
            available.append(("flash_hub", "Flash Attention 2 (Hub)"))

        # Flash Attention 3 - only for Hopper GPUs (SM 9.0+)
        if is_hopper_gpu:
            # Local FA3 requires building from source, so we only check for hub variant
            if has_kernels:
                available.append(("_flash_3_hub", "Flash Attention 3 (Hub)"))

        # Check for Sage Attention (local)
        has_sage_local = False
        try:
            import sageattention  # noqa: F401

            has_sage_local = True  # noqa: F841
            available.append(("sage", "Sage Attention"))
        except ImportError:
            pass

        # Sage Attention from Hub (if kernels installed)
        if has_kernels:
            available.append(("sage_hub", "Sage Attention (Hub)"))

        # Check for xFormers (local only, no hub variant)
        try:
            import xformers  # noqa: F401

            available.append(("xformers", "xFormers"))
        except ImportError:
            pass

        return available

    @staticmethod
    def component_hash(identifier: str) -> str:
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def get_cached(self, hash_key: str) -> Any | None:
        with self._lock:
            return self._component_cache.get(hash_key)

    def set_cached(self, hash_key: str, obj: Any) -> None:
        with self._lock:
            self._component_cache[hash_key] = obj

    @contextlib.contextmanager
    def device_scope(self, *, device, dtype=None):
        dev_token = _SCOPED_DEVICE.set(device)
        dtype_token = _SCOPED_DTYPE.set(dtype)
        try:
            yield
        finally:
            _SCOPED_DEVICE.reset(dev_token)
            _SCOPED_DTYPE.reset(dtype_token)

    def clear(self):
        with self._lock:
            self._components.clear()
            self._model_id = None
            self._component_hashes.clear()
            self._lora_sources.clear()
            self._compiled_components.clear()
            self._component_cache.clear()
            self._managed_components.clear()
            self._applied_strategy = None
            self._group_offload_use_stream = False
            self._group_offload_low_cpu_mem = False
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


_MODEL_MANAGER_SINGLETON: ModelManager | None = None
_MODEL_MANAGER_SINGLETON_LOCK = threading.Lock()
_CURRENT_MODEL_MANAGER: contextvars.ContextVar[ModelManager | None] = contextvars.ContextVar(
    "frameartisan_current_model_manager",
    default=None,
)


def set_global_model_manager(manager: ModelManager | None) -> None:
    global _MODEL_MANAGER_SINGLETON
    with _MODEL_MANAGER_SINGLETON_LOCK:
        _MODEL_MANAGER_SINGLETON = manager


@contextlib.contextmanager
def use_model_manager(manager: ModelManager):
    token = _CURRENT_MODEL_MANAGER.set(manager)
    try:
        yield manager
    finally:
        _CURRENT_MODEL_MANAGER.reset(token)


@contextlib.contextmanager
def model_scope(*, device: torch.device | str | None, dtype: torch.dtype | None = None):
    mm = get_model_manager()
    with mm.device_scope(device=device, dtype=dtype):
        yield mm


def get_model_manager() -> ModelManager:
    global _MODEL_MANAGER_SINGLETON

    current = _CURRENT_MODEL_MANAGER.get()
    if current is not None:
        return current

    if _MODEL_MANAGER_SINGLETON is None:
        with _MODEL_MANAGER_SINGLETON_LOCK:
            if _MODEL_MANAGER_SINGLETON is None:
                _MODEL_MANAGER_SINGLETON = ModelManager()
    return _MODEL_MANAGER_SINGLETON
