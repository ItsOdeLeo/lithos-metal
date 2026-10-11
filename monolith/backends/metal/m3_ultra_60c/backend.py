from ..base import MetalBackend


class Backend(MetalBackend):
    """Native backend for the 60-core M3 Ultra (two 30-core M3 Max dies joined by UltraFusion).

    Does not inherit the 30-core M3 Max profile: twice the cores across two dies is a different device for the
    crew geometry and the bus. With 256 GB the target and its drafter fit at the full context.
    """
    id = "m3_ultra_60c"
