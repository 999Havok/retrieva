"""Shared test setup.

The tests only exercise raptor.py's pure logic, so they don't need the
embedding model, FAISS, or Gemini clients. When those packages aren't
installed (e.g. a lightweight CI env), stand-in modules are registered so
`import raptor` still works. Nothing here touches the network.
"""

import importlib
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stub(module_name, **attrs):
    if importlib.util.find_spec(module_name.split(".")[0]) is not None:
        try:
            importlib.import_module(module_name)
            return
        except ImportError:
            pass
    parts = module_name.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        sys.modules.setdefault(name, types.ModuleType(name))
    for key, value in attrs.items():
        setattr(sys.modules[module_name], key, value)


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("not available in offline unit tests")


_stub("langchain_huggingface", HuggingFaceEmbeddings=_Unavailable)
_stub("langchain_community.vectorstores", FAISS=_Unavailable)
_stub("langchain_google_genai", ChatGoogleGenerativeAI=_Unavailable)
