from slowapi import Limiter  # pyre-ignore[21]
from slowapi.util import get_remote_address  # pyre-ignore[21]

# Keep limiter in a dedicated module so app setup and route decorators share
# a single instance without circular imports.
limiter = Limiter(key_func=get_remote_address, default_limits=[])
