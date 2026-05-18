"""
QRec reservoir for OSF magnetic-sensing sequences.

This module is a faithful Python port of the QRec reservoir code from the
validated v1.2/v1.3 ablation notebook. It encodes (T=100, C=6) magnetic
sequences into 171-dimensional pooled feature vectors via a fixed 10-qubit
quantum recurrent reservoir.

Architecture:
    n_memory=4 qubits + n_processor=6 qubits = 10 qubits total
    Two reservoir layers of random RY/RZ rotations + CNOT entanglement
    Per-timestep:
      - encode 6 inputs into 6 processor qubits via RY/RZ
      - encode memory state into 4 memory qubits via RY/RZ
      - entangle memory chain, processor ring, memory-processor pairs
      - apply reservoir layers
      - measure: 18 single-qubit Pauli expectations (X,Y,Z on each proc)
                 + 15 processor-pair ZZ correlations
                 + 24 memory-processor ZZ correlations
                 + 4 memory Z expectations (for memory feedback)
    Outputs: 57 retained features per timestep × 3 pooling modes = 171
    
Output dimensions:
    n_outputs = 18 + 15 + 24 = 57 retained per-timestep features
    pooled    = 3 × 57 = 171 (concatenate last + mean + std across time)

The reservoir is FIXED (random parameters seeded once with seed=42). No
training happens here; downstream classifiers are trained on the 171-D
embeddings.

This implementation matches the original QRec-Mag work and the v1.2 bridge
exactly (same seed, same circuit, same pooling).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Defaults (locked — must not change without invalidating cached embeddings)
# =============================================================================
QREC_DEFAULTS = dict(
    n_memory=4,
    n_processor=6,
    n_reservoir_layers=2,
    n_inputs=6,
    seed=42,
    coupling_strength=0.3,
)

POOLING_DEFAULT = ("last", "mean", "std")


# =============================================================================
# QuantumRecurrentReservoir10q
# =============================================================================
class QuantumRecurrentReservoir10q:
    """10-qubit quantum recurrent reservoir for time-series embedding.

    Fixed reservoir: parameters seeded at construction, never trained.
    Used to map (T, n_inputs=6) magnetic sequences to per-timestep
    n_outputs=57-dimensional feature vectors.

    The memory register provides recurrent state (4 qubits); the
    processor register receives the time-step input (6 qubits). Memory
    is updated via measured Z expectations.
    """

    def __init__(
        self,
        n_memory: int = 4,
        n_processor: int = 6,
        n_reservoir_layers: int = 2,
        n_inputs: int = 6,
        seed: int = 42,
        coupling_strength: float = 0.3,
    ):
        self.n_memory = n_memory
        self.n_processor = n_processor
        self.n_qubits = n_memory + n_processor
        self.n_reservoir_layers = n_reservoir_layers
        self.n_inputs = n_inputs
        self.seed = seed
        self.coupling_strength = coupling_strength

        self.memory_wires = list(range(n_memory))
        self.proc_wires = list(range(n_memory, self.n_qubits))
        self.proc_pairs = [
            (i, j) for i in self.proc_wires for j in self.proc_wires if i < j
        ]
        self.cross_pairs = [
            (i, j) for i in self.memory_wires for j in self.proc_wires
        ]

        # Output dimension breakdown
        self.n_single = n_processor * 3                       # X, Y, Z per proc qubit
        self.n_proc_pairs = len(self.proc_pairs)              # ZZ pairs in processor
        self.n_cross_pairs = len(self.cross_pairs)            # ZZ memory-processor pairs
        self.n_memory_out = n_memory                          # Z on memory (for feedback)
        self.n_outputs = self.n_single + self.n_proc_pairs + self.n_cross_pairs

        # Seed once, sample all reservoir parameters deterministically
        rng = np.random.RandomState(seed)
        self.reservoir_params = rng.uniform(
            0, 2 * np.pi, size=(n_reservoir_layers, self.n_qubits, 2)
        )
        self.input_projection = rng.randn(n_inputs, n_processor) * 0.5
        self.memory_rotation = rng.uniform(0, 2 * np.pi, size=(n_memory, 2))

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.memory_angles = np.zeros(n_memory)
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Build the parameterised quantum circuit as a QNode.

        Layout of returned observables:
            [0:18]   single-qubit Pauli X/Y/Z on each of 6 processor qubits
            [18:33]  ZZ correlations within processor (15 pairs)
            [33:57]  ZZ correlations memory-processor (24 pairs)
            [57:61]  Z on memory qubits (used for recurrent feedback only)
        """
        n_q = self.n_qubits
        n_m = self.n_memory
        n_l = self.n_reservoir_layers
        rp = self.reservoir_params
        mw = self.memory_wires
        pw = self.proc_wires
        pp = self.proc_pairs
        cp = self.cross_pairs
        c = self.coupling_strength
        mr = self.memory_rotation

        @qml.qnode(self.dev)
        def circuit(proc_input, memory_angles):
            # 1. Encode memory state on memory qubits
            for i, w in enumerate(mw):
                qml.RY(memory_angles[i], wires=w)
                qml.RZ(mr[i, 0], wires=w)

            # 2. Encode input on processor qubits
            for i, w in enumerate(pw):
                if i < len(proc_input):
                    qml.RY(proc_input[i], wires=w)
                    qml.RZ(0.5 * proc_input[i], wires=w)

            # 3. Memory chain (CNOT m0->m1->m2->m3)
            for i in range(n_m - 1):
                qml.CNOT(wires=[mw[i], mw[i + 1]])

            # 4. Processor ring (CNOT p0->p1->...->p5->p0)
            for i in range(len(pw) - 1):
                qml.CNOT(wires=[pw[i], pw[i + 1]])
            qml.CNOT(wires=[pw[-1], pw[0]])

            # 5. Memory-processor coupling (controlled rotations)
            for i in range(n_m):
                qml.CRY(c * np.pi, wires=[mw[i], pw[i]])
                qml.CRY(c * 0.5 * np.pi, wires=[pw[i], mw[i]])
            # Skip connections (m_i -> p_{i+2})
            for i in range(n_m):
                qml.CRY(c * 0.3 * np.pi, wires=[mw[i], pw[(i + 2) % len(pw)]])

            # 6. Reservoir layers (random rotations + CNOT entanglement)
            for layer in range(n_l):
                for i in range(n_q):
                    qml.RY(rp[layer, i, 0], wires=i)
                    qml.RZ(rp[layer, i, 1], wires=i)
                for i in range(n_q):
                    qml.CNOT(wires=[i, (i + 1) % n_q])
                for i in range(0, n_q - 2, 2):
                    qml.CNOT(wires=[i, i + 2])

            # 7. Measurements
            obs = []
            for w in pw:
                obs.extend([
                    qml.expval(qml.PauliX(w)),
                    qml.expval(qml.PauliY(w)),
                    qml.expval(qml.PauliZ(w)),
                ])
            for (i, j) in pp:
                obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            for (i, j) in cp:
                obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            # Memory observables for recurrent feedback (NOT retained in pooled output)
            for w in mw:
                obs.append(qml.expval(qml.PauliZ(w)))
            return obs

        self._circuit = circuit

    def reset(self) -> None:
        """Reset memory to zero state. Call before processing each new sequence."""
        self.memory_angles = np.zeros(self.n_memory)

    def process_single(self, x_t: np.ndarray) -> np.ndarray:
        """Process a single timestep input vector x_t of shape (n_inputs,).

        Returns the n_outputs=57 retained features. Internally updates
        the memory_angles state from the measured memory observables.
        """
        proc_input = np.clip(x_t @ self.input_projection, -np.pi, np.pi)
        all_outputs = np.array(self._circuit(proc_input, self.memory_angles), dtype=float)
        # Memory observables are the LAST n_memory entries (used for recurrent feedback)
        memory_obs = all_outputs[self.n_outputs:self.n_outputs + self.n_memory]
        # Map [-1, 1] -> [0, 2pi] for the next step's memory angles
        self.memory_angles = (memory_obs + 1.0) * np.pi
        return all_outputs[:self.n_outputs]

    def process_sequence(self, X: np.ndarray) -> np.ndarray:
        """Process a full sequence of shape (T, n_inputs).

        Returns reservoir states of shape (T, n_outputs=57).
        Each call to process_sequence resets the memory.
        """
        self.reset()
        states = np.zeros((X.shape[0], self.n_outputs), dtype=float)
        for t in range(X.shape[0]):
            states[t] = self.process_single(X[t])
        return states


# =============================================================================
# Pooling and preprocessing
# =============================================================================
def pool_reservoir_states(
    states: np.ndarray,
    pooling: tuple = POOLING_DEFAULT,
) -> np.ndarray:
    """Pool per-timestep reservoir states into a single feature vector.

    Default pooling concatenates [last, mean, std] across the time axis,
    producing a 3*57 = 171-dimensional embedding.
    """
    parts = []
    if "last" in pooling:
        parts.append(states[-1])
    if "mean" in pooling:
        parts.append(states.mean(axis=0))
    if "std" in pooling:
        parts.append(states.std(axis=0))
    return np.concatenate(parts, axis=0)


def angle_scale_sequences(
    X_seq: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Standard-scale + clip + rescale to angle range [-pi/3, pi/3].

    The scaler is fitted to the input if not provided. Critical: the
    SAME scaler must be used for all subsequent embeddings, otherwise
    they live in different angle spaces and become incomparable.

    Args:
        X_seq: shape (N, T, C) or (1, T, C) for a single sequence.
        scaler: optional pre-fitted scaler. If None, fit on flattened input.

    Returns:
        (scaled_X, scaler) — scaled_X has same shape as input.
    """
    N, T, C = X_seq.shape
    flat = X_seq.reshape(-1, C)
    if scaler is None:
        scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(N, T, C)
    scaled = np.clip(scaled, -3, 3) * (np.pi / 3.0)
    return scaled, scaler
