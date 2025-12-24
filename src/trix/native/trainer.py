"""
TriX Native Trainer

No PyTorch. No TensorFlow. Pure CuPy training loop.

Components:
- Loss functions (MSE, cross-entropy)
- AdamOptimizer
- Trainer class for training loops
"""

import numpy as np
import time
from typing import Dict, Tuple, Callable, Optional, List

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def mse_loss(pred: "cp.ndarray", target: "cp.ndarray") -> Tuple["cp.ndarray", "cp.ndarray"]:
    """
    Mean Squared Error loss with gradient.

    Args:
        pred: Predictions [batch, ...]
        target: Targets [batch, ...]

    Returns:
        (loss, gradient) tuple
    """
    diff = pred - target
    loss = (diff ** 2).mean()
    grad = 2.0 * diff / diff.size
    return loss, grad


def cross_entropy_loss(
    logits: "cp.ndarray",
    targets: "cp.ndarray"
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Raw logits [batch, num_classes]
        targets: Class indices [batch]

    Returns:
        (loss, gradient) tuple
    """
    # Softmax
    exp_logits = cp.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Loss
    batch_size = logits.shape[0]
    log_probs = cp.log(probs + 1e-10)
    loss = -log_probs[cp.arange(batch_size), targets].mean()

    # Gradient
    grad = probs.copy()
    grad[cp.arange(batch_size), targets] -= 1.0
    grad /= batch_size

    return loss, grad


def binary_cross_entropy_loss(
    pred: "cp.ndarray",
    target: "cp.ndarray"
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """
    Binary cross-entropy loss.

    Args:
        pred: Predictions in [0, 1] range [batch, ...]
        target: Binary targets [batch, ...]

    Returns:
        (loss, gradient) tuple
    """
    eps = 1e-7
    pred = cp.clip(pred, eps, 1 - eps)

    loss = -(target * cp.log(pred) + (1 - target) * cp.log(1 - pred)).mean()
    grad = (pred - target) / (pred * (1 - pred) + eps) / pred.size

    return loss, grad


# =============================================================================
# ADAM OPTIMIZER
# =============================================================================

class AdamOptimizer:
    """
    Pure CuPy Adam optimizer.

    Args:
        params: Dictionary of parameter name -> array
        lr: Learning rate
        beta1: First moment decay
        beta2: Second moment decay
        eps: Numerical stability epsilon

    Example:
        >>> optimizer = AdamOptimizer({'weights': w, 'bias': b}, lr=0.001)
        >>> optimizer.step({'weights': dw, 'bias': db})
    """

    def __init__(
        self,
        params: Dict[str, "cp.ndarray"],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # Moment buffers
        self.m = {k: cp.zeros_like(v) for k, v in params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, "cp.ndarray"]):
        """
        Update parameters with gradients.

        Args:
            grads: Dictionary of parameter name -> gradient array
        """
        self.t += 1
        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t

        for name, param in self.params.items():
            if name not in grads:
                continue

            g = grads[name]
            m = self.m[name]
            v = self.v[name]

            # Update moments
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = m / (1 - beta1_t)
            v_hat = v / (1 - beta2_t)

            # Update parameter
            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset for next iteration (moments persist)."""
        pass  # Moments persist across iterations in Adam

    def state_dict(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            't': self.t,
            'm': {k: cp.asnumpy(v) for k, v in self.m.items()},
            'v': {k: cp.asnumpy(v) for k, v in self.v.items()},
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
        }

    def load_state_dict(self, state: Dict):
        """Load optimizer state from checkpoint."""
        self.t = state['t']
        self.lr = state['lr']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']

        for k, v in state['m'].items():
            if k in self.m:
                self.m[k] = cp.asarray(v)

        for k, v in state['v'].items():
            if k in self.v:
                self.v[k] = cp.asarray(v)


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    Native training loop. No frameworks.

    Args:
        model: Model with forward(), backward(), step() methods
        loss_fn: Loss function name ('mse', 'cross_entropy', 'bce')
                 or callable (pred, target) -> (loss, grad)

    Example:
        >>> trainer = Trainer(model, loss_fn='mse')
        >>> history = trainer.train(train_x, train_y, epochs=100)
    """

    def __init__(
        self,
        model,
        loss_fn: str = 'mse',
    ):
        self.model = model

        if isinstance(loss_fn, str):
            self.loss_fn = {
                'mse': mse_loss,
                'cross_entropy': cross_entropy_loss,
                'bce': binary_cross_entropy_loss,
            }.get(loss_fn, mse_loss)
        else:
            self.loss_fn = loss_fn

        self.history = {
            'loss': [],
            'time': [],
        }

    def train_step(
        self,
        x: "cp.ndarray",
        target: "cp.ndarray",
        positions: Optional["cp.ndarray"] = None,
    ) -> float:
        """
        Single training step.

        Args:
            x: Input batch
            target: Target batch
            positions: Optional position indices

        Returns:
            Loss value as float
        """
        # Forward
        output = self.model.forward(x, positions, save_for_backward=True)

        # Loss
        loss, d_loss = self.loss_fn(output, target)

        # Backward
        self.model.backward(d_loss)

        # Update
        self.model.step()

        return float(loss)

    def train(
        self,
        train_x: "cp.ndarray",
        train_y: "cp.ndarray",
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
        log_every: int = 10,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_x: Training inputs
            train_y: Training targets
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to print progress
            log_every: Print every N epochs

        Returns:
            Training history dictionary
        """
        n_samples = train_x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        if verbose:
            print(f"Training on {n_samples:,} samples")
            print(f"  Epochs: {epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Batches per epoch: {n_batches}")
            print()

        start_time = time.perf_counter()

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle
            indices_np = np.random.permutation(n_samples)
            indices = cp.asarray(indices_np)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]

                loss = self.train_step(batch_x, batch_y)
                epoch_loss += loss

            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)
            self.history['time'].append(time.perf_counter() - start_time)

            if verbose and (epoch + 1) % log_every == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Epoch {epoch+1:4d}: loss = {epoch_loss:.6f} ({elapsed:.1f}s)")

        total_time = time.perf_counter() - start_time

        if verbose:
            print(f"\nTraining complete in {total_time:.2f}s")
            print(f"Final loss: {self.history['loss'][-1]:.6f}")

        return self.history

    def evaluate(
        self,
        test_x: "cp.ndarray",
        test_y: "cp.ndarray",
        batch_size: int = 256,
    ) -> float:
        """
        Evaluate model on test data.

        Args:
            test_x: Test inputs
            test_y: Test targets
            batch_size: Batch size

        Returns:
            Average loss
        """
        n_samples = test_x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        total_loss = 0.0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            batch_x = test_x[start_idx:end_idx]
            batch_y = test_y[start_idx:end_idx]

            output = self.model.forward(batch_x, save_for_backward=False)
            loss, _ = self.loss_fn(output, batch_y)
            total_loss += float(loss)

        return total_loss / n_batches
