"""Stub for HuggingFace datasets compatibility layer.

This module is imported for side effects (registering dataset formats, etc.).
When the actual module is unavailable, this stub provides a no-op fallback so
that config/checkpoint loading can proceed for inference-only use cases.
"""

# The original module likely registers custom dataset builders or feature types
# with HuggingFace `datasets`. For inference-only usage (policy evaluation,
# rollout collection, etc.), we don't need these registrations.
pass
