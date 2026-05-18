"""
MagBridge-Battery — bridge architecture (v1.3).

Public API:
    BridgeV13           main bridge class
    BridgeConfig        bridge hyperparameters dataclass
    MorphologyBank      OSF anchor statistics
    Regime              regime enum (GROUNDED / EXTRAPOLATION / UNSUPPORTED)
    classify_regime     regime classifier for a voltage
    QuantumRecurrentReservoir10q  the 10-qubit QRec reservoir
"""

from magbridge.bridge.bridge import BridgeV13, BridgeConfig
from magbridge.bridge.morphology import (
    MorphologyBank,
    Regime,
    classify_regime,
    OSF_ANCHORS,
    GROUNDED_LOW,
    GROUNDED_HIGH,
    EXTRAPOLATION_LOW,
)
from magbridge.bridge.qrec import (
    QuantumRecurrentReservoir10q,
    QREC_DEFAULTS,
    POOLING_DEFAULT,
    pool_reservoir_states,
    angle_scale_sequences,
)

__all__ = [
    "BridgeV13",
    "BridgeConfig",
    "MorphologyBank",
    "Regime",
    "classify_regime",
    "QuantumRecurrentReservoir10q",
    "QREC_DEFAULTS",
    "POOLING_DEFAULT",
    "pool_reservoir_states",
    "angle_scale_sequences",
    "OSF_ANCHORS",
    "GROUNDED_LOW",
    "GROUNDED_HIGH",
    "EXTRAPOLATION_LOW",
]
