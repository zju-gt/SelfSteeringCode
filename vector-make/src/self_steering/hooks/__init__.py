"""Native PyTorch residual-stream hooks."""

from self_steering.hooks.capture import capture_last_token
from self_steering.hooks.intervention import add_steering_vector

__all__ = ["add_steering_vector", "capture_last_token"]

