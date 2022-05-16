import builtins

try:
    builtins.profile  # type: ignore
except AttributeError:
    builtins.profile = lambda x: x  # type: ignore

