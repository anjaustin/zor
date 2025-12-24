# Formal Notation for ZIT-1 Plasticity Rules

## Definitions

### State Variables

For each node $i \in \{0, 1, \ldots, N-1\}$:

- $S_i(t) \in \{0, 1, \ldots, 255\}$ — State value at cycle $t$
- $F_i(t) \in \{0, 1, \ldots, 255\}$ — Frustration counter at cycle $t$
- $R_i(t) \in \{0, 1\}$ — Resonance indicator at cycle $t$
- $\mathcal{N}_i(t) = \{n_0, n_1, n_2, n_3, n_4, n_5\}$ — Neighbor set at cycle $t$
- $W_i(t) \in \{0, 1\}$ — Rewiring active flag

### Global Parameters

- $\theta$ — Frustration threshold (default: 4)
- $\tau$ — Evaluation period (default: 8 cycles)
- $\gamma$ — Decay shift (default: 1, meaning halving)

---

## The Frozen Shape: Comparator Swap

For each phase $p \in \{0, 1, 2, 3, 4, 5\}$:

$$
S_i(t, p+1) = \begin{cases}
S_{n_p}(t, p) & \text{if } \text{swap}_p(i, t) \\
S_i(t, p) & \text{otherwise}
\end{cases}
$$

where the swap condition is:

$$
\text{swap}_p(i, t) = \begin{cases}
S_i(t, p) > S_{n_p}(t, p) & \text{if } p \text{ is even (positive direction)} \\
S_{n_p}(t, p) > S_i(t, p) & \text{if } p \text{ is odd (negative direction)}
\end{cases}
$$

---

## Resonance

A node is resonant if it completed all 6 phases without swapping:

$$
R_i(t) = \prod_{p=0}^{5} \neg\text{swap}_p(i, t)
$$

Or equivalently:

$$
R_i(t) = 1 \iff \forall p \in \{0..5\}: S_i(t, p+1) = S_i(t, p)
$$

---

## Frustration Dynamics

### Without Active Rewiring ($W_i(t) = 0$)

$$
F_i(t+1) = \begin{cases}
\min(F_i(t) + 1, 255) & \text{if } R_i(t) = 0 \\
\lfloor F_i(t) / 2^\gamma \rfloor & \text{if } R_i(t) = 1
\end{cases}
$$

### During Rewiring Evaluation ($W_i(t) = 1$)

$$
F_i(t+1) = \begin{cases}
\min(F_i(t) + 1, 255) & \text{if } R_i(t) = 0 \\
F_i(t) & \text{if } R_i(t) = 1
\end{cases}
$$

(No decay during evaluation to get accurate comparison)

---

## Plasticity Rules

### Initiate Rewiring

When $W_i(t) = 0$ and $F_i(t) \geq \theta$:

1. Set $W_i(t+1) = 1$
2. Store old neighbor: $\text{old}_i = n_d$ where $d$ is current rewire direction
3. Store pre-rewire frustration: $F^{\text{pre}}_i = F_i(t)$
4. Select random target: $n'_d \sim \text{Uniform}(\{0..N-1\} \setminus \{i\})$
5. Update neighbor: $n_d \leftarrow n'_d$
6. Reset frustration: $F_i(t+1) = 0$
7. Reset evaluation counter: $e_i = 0$

### Evaluation Period

While $W_i(t) = 1$ and $e_i < \tau$:

$$
e_i \leftarrow e_i + 1
$$

### Decision After Evaluation

When $e_i = \tau$:

$$
n_d(t+1) = \begin{cases}
\text{old}_i & \text{if } F_i(t) \geq F^{\text{pre}}_i \text{ (revert)} \\
n'_d & \text{otherwise (keep)}
\end{cases}
$$

Then:
- $W_i(t+1) = 0$
- $d \leftarrow (d + 1) \mod 6$ (rotate to next direction)
- $F_i(t+1) = 0$

---

## Convergence Criterion

The system has converged at cycle $T$ if:

$$
\sum_{i=0}^{N-1} R_i(T) = N
$$

That is, all nodes are resonant simultaneously.

---

## The Scaling Law (Empirical)

Let $C(N)$ denote cycles to convergence for $N$ nodes:

$$
C(N) = O(\log^k N) \text{ for some } k \approx 1
$$

Observed data suggests sublinear scaling with "sweet spots" where certain $N$ values converge faster due to topological properties.

| $N$ | $\log_2 N$ | $C(N)$ | $C(N)/\log_2 N$ |
|-----|------------|--------|-----------------|
| 64 | 6 | 82 | 13.7 |
| 512 | 9 | 102 | 11.3 |
| 4,096 | 12 | 168 | 14.0 |
| 32,768 | 15 | 145 | 9.7 |
| 262,144 | 18 | 166 | 9.2 |
| 2,097,152 | 21 | 540 | 25.7 |
| 16,777,216 | 24 | 1,063 | 44.3 |
| 56,623,104 | 25.7 | 570 | 22.2 |

---

## Invariants

### Topology Invariant

At all times:
$$
\forall i, d: n_d^{(i)} \in \{0..N-1\} \setminus \{i\}
$$

(Neighbors are valid node indices, no self-loops)

### Conservation

The multiset of state values is preserved under swap operations:
$$
\{S_i(t) : i \in \{0..N-1\}\} = \{S_i(0) : i \in \{0..N-1\}\}
$$

(Values are permuted, not created or destroyed)

---

## Thermodynamic Interpretation

### Frustration as Entropy Gradient

$$
\nabla H_i = F_i - \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} F_j
$$

Nodes rewire to reduce local entropy gradients.

### Resonance as Equilibrium

$$
R_i = 1 \iff \nabla H_i = 0
$$

Full convergence represents global thermodynamic equilibrium:

$$
\sum_i R_i = N \iff \text{System at minimum free energy}
$$
