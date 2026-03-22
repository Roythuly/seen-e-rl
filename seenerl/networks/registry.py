"""
Network registry for extensible architecture support.

Register custom networks (e.g., VLA, Transformer-based) using decorators:

    @register_actor("my_vla_actor")
    class VLAActor(BaseActor):
        ...

Then build from config:
    actor = build_actor("my_vla_actor", **kwargs)
"""

from typing import Any, Dict, Type

ACTOR_REGISTRY: Dict[str, Type] = {}
CRITIC_REGISTRY: Dict[str, Type] = {}


def register_actor(name: str):
    """Decorator to register an actor network class."""
    def decorator(cls):
        ACTOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_critic(name: str):
    """Decorator to register a critic network class."""
    def decorator(cls):
        CRITIC_REGISTRY[name] = cls
        return cls
    return decorator


def build_actor(name: str, **kwargs) -> Any:
    """Build an actor network by registered name."""
    if name not in ACTOR_REGISTRY:
        raise ValueError(
            f"Actor '{name}' not found. Available: {list(ACTOR_REGISTRY.keys())}"
        )
    return ACTOR_REGISTRY[name](**kwargs)


def build_critic(name: str, **kwargs) -> Any:
    """Build a critic network by registered name."""
    if name not in CRITIC_REGISTRY:
        raise ValueError(
            f"Critic '{name}' not found. Available: {list(CRITIC_REGISTRY.keys())}"
        )
    return CRITIC_REGISTRY[name](**kwargs)
