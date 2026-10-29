"""A small, dependency-light neural-network learning project.

Run with ``python -m src.weather_nn`` from the project directory.
"""

from __future__ import annotations

import numpy as np


MAP_SIZE = 4
HISTORY = 3
INPUTS = MAP_SIZE * MAP_SIZE * HISTORY
CLASSES = ("clear", "rain", "storm")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash a neuron's weighted input into the range (0, 1)."""
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def make_dataset(samples: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create moving 4x4 rain patches and binary/multiclass targets.

    X contains three consecutive maps flattened into 48 pixel inputs. Labels
    describe the next map, which keeps the exercise forecast-shaped without
    requiring downloaded observations.
    """
    rng = np.random.default_rng(seed)
    features = np.zeros((samples, INPUTS), dtype=np.float64)
    binary = np.zeros(samples, dtype=np.int64)
    labels = np.zeros(samples, dtype=np.int64)

    for i in range(samples):
        row, col = rng.integers(0, MAP_SIZE, size=2)
        drift_row, drift_col = rng.choice((-1, 0, 1), size=2)
        intensity = rng.uniform(0.35, 1.0)
        maps = []
        for step in range(HISTORY + 1):
            grid = rng.uniform(0.0, 0.08, size=(MAP_SIZE, MAP_SIZE))
            r = (row + step * drift_row) % MAP_SIZE
            c = (col + step * drift_col) % MAP_SIZE
            grid[r, c] = intensity
            # A small neighboring cell makes the patterns less trivial.
            grid[r, (c + 1) % MAP_SIZE] = intensity * 0.55
            maps.append(grid)

        features[i] = np.asarray(maps[:HISTORY]).reshape(-1)
        next_mean = float(maps[-1].mean())
        binary[i] = int(next_mean > 0.12)
        labels[i] = 2 if next_mean > 0.14 else (1 if next_mean > 0.08 else 0)

    return features, binary, labels


class Perceptron:
    """Single linear threshold neuron for binary rain/no-rain classification."""

    def __init__(self, inputs: int, learning_rate: float = 0.1, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.weights + self.bias >= 0.0).astype(np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> None:
        for _ in range(epochs):
            for xi, target in zip(X, y):
                guess = int(xi @ self.weights + self.bias >= 0.0)
                error = int(target) - guess
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error


class MLP:
    """One-hidden-layer network with vectorized mini-batch backpropagation."""

    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        learning_rate: float = 0.8,
        seed: int = 2,
    ):
        rng = np.random.default_rng(seed)
        # Heuristic-scaled initialization keeps early activations well behaved.
        self.W1 = rng.normal(0.0, np.sqrt(1.0 / inputs), (inputs, hidden))
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, outputs))
        self.b2 = np.zeros((1, outputs))
        self.learning_rate = learning_rate

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Feed a whole batch through both layers using matrix operations."""
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        scores = a1 @ self.W2 + self.b2
        return a1, scores, self.softmax(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[2].argmax(axis=1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        batch_size: int = 32,
        seed: int = 3,
    ) -> list[float]:
        """Train with cross-entropy and mini-batch gradient descent."""
        rng = np.random.default_rng(seed)
        history: list[float] = []
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                indices = order[start : start + batch_size]
                xb, yb = X[indices], y[indices]
                a1, _, probabilities = self.forward(xb)
                batch_count = len(xb)

                # Softmax + cross-entropy derivative at the output layer.
                delta2 = probabilities.copy()
                delta2[np.arange(batch_count), yb] -= 1.0
                delta2 /= batch_count
                grad_W2 = a1.T @ delta2
                grad_b2 = delta2.sum(axis=0, keepdims=True)

                # Chain rule through sigmoid hidden neurons.
                delta1 = (delta2 @ self.W2.T) * a1 * (1.0 - a1)
                grad_W1 = xb.T @ delta1
                grad_b1 = delta1.sum(axis=0, keepdims=True)

                self.W2 -= self.learning_rate * grad_W2
                self.b2 -= self.learning_rate * grad_b2
                self.W1 -= self.learning_rate * grad_W1
                self.b1 -= self.learning_rate * grad_b1

            _, _, probs = self.forward(X)
            loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()
            history.append(float(loss))
        return history


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(predictions == targets))


def print_map(flat: np.ndarray) -> None:
    """Render one normalized weather image as a compact text pixel grid."""
    grid = flat.reshape(MAP_SIZE, MAP_SIZE)
    symbols = " .:+#"
    for row in grid:
        print(" ".join(symbols[min(int(value * len(symbols)), len(symbols) - 1)] for value in row))


def main() -> None:
    X, binary_y, labels = make_dataset(600, seed=7)
    rng = np.random.default_rng(11)
    order = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = order[:split], order[split:]
    X_train, X_test = X[train_idx], X[test_idx]

    perceptron = Perceptron(INPUTS)
    perceptron.fit(X_train, binary_y[train_idx])
    binary_score = accuracy(perceptron.predict(X_test), binary_y[test_idx])

    network = MLP(INPUTS, hidden=12, outputs=len(CLASSES))
    losses = network.fit(X_train, labels[train_idx], epochs=250, batch_size=32)
    predictions = network.predict(X_test)

    print("Weather Pixels — neural network learning demo")
    print(f"Input: {HISTORY} maps × {MAP_SIZE} × {MAP_SIZE} pixels = {INPUTS} values")
    print(f"Weight shapes: W1={network.W1.shape}, W2={network.W2.shape}")
    print(f"Perceptron rain/no-rain test accuracy: {binary_score:.1%}")
    print(f"MLP {CLASSES} test accuracy: {accuracy(predictions, labels[test_idx]):.1%}")
    print(f"Cross-entropy: {losses[0]:.3f} → {losses[-1]:.3f}")
    print("\nExample past weather pixels (bright = heavier rain):")
    print_map(X_test[0][-MAP_SIZE * MAP_SIZE :])
    probabilities = network.forward(X_test[:1])[2][0]
    print("\nNext-map class probabilities:")
    for name, probability in zip(CLASSES, probabilities):
        print(f"  {name:>5}: {probability:.1%}")
    print("\nThis is a synthetic teaching demo, not an operational forecast.")


if __name__ == "__main__":
    main()
