import importlib

__all__ = ["GAT", "HRN", "HRNClassifier"]


class _LazyClassProxy:
    def __init__(self, module_name: str, attr_name: str):
        self._module_name = module_name
        self._attr_name = attr_name
        self._real = None

    def _load(self):
        if self._real is None:
            mod = importlib.import_module(self._module_name, __package__)
            self._real = getattr(mod, self._attr_name)

    def __call__(self, *args, **kwargs):
        self._load()
        return self._real(*args, **kwargs)

    def __getattr__(self, item):
        self._load()
        return getattr(self._real, item)


def __getattr__(name: str):
    if name == "GAT":
        return _LazyClassProxy(".gat", "GAT")
    if name == "HRN":
        return _LazyClassProxy(".hrn", "HRN")
    if name == "HRNClassifier":
        return _LazyClassProxy(".hrn_classifier", "HRNClassifier")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__