from slowapi import Limiter
from slowapi.util import get_remote_address

# In local/dev mode we use the default in-memory storage so Redis is not required.
# For production with multiple workers, set storage_uri="redis://localhost:6379/0".
limiter = Limiter(key_func=get_remote_address)
