from . import pickup_ring_models  # noqa: F401

try:
    from . import overcooked_models  # noqa: F401
except ImportError:
    pass  # Might fail if Overcooked isn't installed.
