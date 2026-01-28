"""Experimental models that extend the baseline implementations."""

from .minimal_spike_head import (
    MinimalSpikeHeadLinear,
    MinimalSpikeHeadMLP,
    MinimalSpikeHeadLiveRecMLP,
)

__all__ = [
    "MinimalSpikeHeadLinear",
    "MinimalSpikeHeadMLP",
    "MinimalSpikeHeadLiveRecMLP",
]

