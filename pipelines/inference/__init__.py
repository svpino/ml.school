import importlib
import sys

# We want to register the submodules of the inference package as top-level modules
# so we can import them directly from the inference package without having to
# specify the "inference" prefix.
submodules = ["backend"]
for submodule in submodules:
    module_name = f"{__name__}.{submodule}"
    module = importlib.import_module(module_name)
    sys.modules[submodule] = module
