# Providence

*Content-Addressed Memory with Frozen Shapes*

> *"In Providence, addresses find you."*

---

## What Is Providence?

Providence is content-addressed memory. Instead of looking up by index ("give me item #47"), you look up by content ("give me the item most similar to this").

Traditional memory:
```c
value = memory[index];  // You need to know the address
```

Providence:
```c
value = providence_lookup(query);  // You describe what you want
```

---

## The Core Operations

Everything in Providence is built from frozen shapes.

### 1. Hamming Distance (The Foundation)

How similar are two binary vectors? Count the bits that differ.

```c
/**
 * Hamming Distance
 *
 * d(a, b) = popcount(a XOR b)
 *
 * For binary inputs, this is exact.
 * For continuous inputs, we use L1 as a differentiable proxy.
 */
static inline float trix_hamming_distance(
    const float* a,
    const float* b,
    int len
) {
    float dist = 0.0f;
    for (int i = 0; i < len; i++) {
        // XOR for binary: a[i] + b[i] - 2*a[i]*b[i]
        // L1 for continuous: |a[i] - b[i]|
        dist += fabsf(a[i] - b[i]);
    }
    return dist;
}
```

**Why Hamming?**
- Fast (just XOR and popcount on binary)
- Works with binary embeddings (32x compression)
- Natural distance metric for bit patterns

### 2. Soft Lookup (The Query)

Find the k nearest neighbors and blend their values.

```c
static inline void trix_providence_lookup(
    const trix_providence_t* prov,
    const float* query,     // What we're looking for
    float* output,          // What we find
    int top_k,
    float temperature
) {
    // Step 1: Compute distance to all keys
    float* distances = alloca(memory_size * sizeof(float));
    for (int i = 0; i < memory_size; i++) {
        distances[i] = trix_hamming_distance(query, &keys[i * d], d);
    }

    // Step 2: Find top-k nearest
    int* top_k_idx = alloca(top_k * sizeof(int));
    // ... (selection sort or partial sort)

    // Step 3: Compute attention weights (frozen softmax)
    float* weights = alloca(top_k * sizeof(float));
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        weights[k] = expf(-distances[top_k_idx[k]] / temperature);
        sum += weights[k];
    }
    for (int k = 0; k < top_k; k++) {
        weights[k] /= sum;
    }

    // Step 4: Weighted blend of values
    memset(output, 0, d * sizeof(float));
    for (int k = 0; k < top_k; k++) {
        int idx = top_k_idx[k];
        for (int i = 0; i < d; i++) {
            output[i] += weights[k] * values[idx * d + i];
        }
    }
}
```

---

## Precision Awareness

Providence can use different precisions for keys and values:

```c
typedef struct {
    int d_model;
    int memory_size;
    float* keys;              // Can be FP8 for compression
    float* values;            // Can be FP16/FP32 for precision
    trix_precision_t key_precision;
    trix_precision_t value_precision;
} trix_providence_t;
```

**Why?**
- Keys just need to be "similar enough" - low precision is fine
- Values need to be accurate - higher precision
- 4:1 or 8:1 memory savings on keys

---

## The C API

### Initialization

```c
#include <trixc/providence.h>

trix_providence_t prov;
trix_providence_init(
    &prov,
    1024,       // memory_size: number of entries
    64,         // d_model: dimension
    TRIX_FP8,   // key_precision
    TRIX_FP16   // value_precision
);
```

### Lookup

```c
float query[64];
float result[64];

// Single query
trix_providence_lookup(&prov, query, result, 16, 1.0f);

// Batch query
float batch_query[4 * 64];
float batch_result[4 * 64];
trix_providence_lookup_batch(&prov, batch_query, batch_result, 4, 16, 1.0f);
```

### Update (for training)

```c
// Write to specific indices
int indices[] = {0, 5, 10};
float new_keys[3 * 64];
float new_values[3 * 64];
trix_providence_update(&prov, new_keys, new_values, indices, 3);
```

### Cleanup

```c
trix_providence_free(&prov);
```

---

## Hierarchical Providence

For large memories, use a hierarchy:

```
                    Query
                      │
                      ▼
              ┌───────────────┐
              │   Level 0    │  Coarse index (256 entries)
              │   Lookup     │
              └───────┬───────┘
                      │ top-k candidates
                      ▼
              ┌───────────────┐
              │   Level 1    │  Fine index (within candidates)
              │   Lookup     │
              └───────┬───────┘
                      │
                      ▼
                   Result
```

```c
trix_hierarchical_providence_t hprov;
trix_hierarchical_providence_init(
    &hprov,
    65536,      // total_entries
    2,          // num_levels
    (int[]){256, 256},  // entries_per_level
    64,         // d_model
    TRIX_FP4,   // coarse_precision
    TRIX_FP16   // fine_precision
);

trix_hierarchical_providence_lookup(&hprov, query, result, 16);
```

**Why hierarchical?**
- Full lookup is O(N) in memory size
- Hierarchical is O(√N) or O(log N)
- Enables million-entry memories

---

## Comparison with Other Systems

### vs. Faiss / Annoy / ScaNN

| Feature | Providence | Faiss |
|---------|------------|-------|
| Distance | Hamming (frozen) | Various |
| Precision | Mixed (APU) | Fixed |
| Soft lookup | Built-in | External |
| Binary | Native | Plugin |
| Size | <10 KB | >10 MB |
| Dependencies | None | MKL/OpenMP |

### vs. Key-Value Memory in Transformers

| Feature | Providence | Transformer KV |
|---------|------------|----------------|
| Lookup | Content-based | Position-based |
| Distance | Hamming | Dot product |
| Sparse | Yes (top-k) | Dense (softmax) |
| Interpretable | Yes | Somewhat |

### vs. Hash Tables

| Feature | Providence | Hash Table |
|---------|------------|------------|
| Lookup | Soft (blended) | Hard (exact) |
| Similar keys | Blend results | Collision |
| Missing keys | Nearest neighbor | Error/default |

---

## Use Cases

### 1. Embedding Lookup

Replace learned embeddings with content-addressed lookup:

```c
// Instead of: embedding = E[token_id]
// Do:
float token_query[64] = one_hot_encode(token_id);
trix_providence_lookup(&embedding_memory, token_query, embedding, 8, 1.0f);
```

### 2. FFN Replacement (via Sparse Octave)

Providence is the core of Sparse Octave Lookup - see [SPARSE_OCTAVE.md](SPARSE_OCTAVE.md).

### 3. Retrieval-Augmented Generation

```c
// Encode query
float query_embedding[768];
encode(query_text, query_embedding);

// Retrieve relevant documents
float doc_embedding[768];
trix_providence_lookup(&doc_memory, query_embedding, doc_embedding, 5, 1.0f);

// doc_embedding is now a blend of the 5 most relevant documents
```

### 4. Learned Index

Replace B-trees with Providence:

```c
// Key: The thing you're searching for
// Value: Where to find it (offset, pointer, etc.)

float search_key[32];
float location_hint[32];
trix_providence_lookup(&index, search_key, location_hint, 1, 0.1f);
// location_hint points to the right neighborhood
```

---

## Why "Providence"?

Providence, Rhode Island. Where addresses find you.

Also: Divine providence - the idea that things work out as they should.

In Providence memory, you don't need to know the address. You describe what you want, and the right thing finds you.

---

## The Frozen Shape Breakdown

| Operation | Shape | Frozen? |
|-----------|-------|---------|
| Key-query comparison | Hamming distance | Yes |
| Top-k selection | Comparison sort | Yes |
| Attention weights | Softmax | Yes |
| Value blending | Weighted sum | Yes |
| Key storage | Memory | Trained → Frozen |
| Value storage | Memory | Trained → Frozen |

**Total learned parameters:** Memory contents
**Total frozen operations:** All the compute

---

## Performance

### Single Lookup (d=64, k=16)

| Memory Size | Time (C, O3) |
|-------------|--------------|
| 128 | 0.02 ms |
| 1024 | 0.15 ms |
| 8192 | 1.2 ms |

### Binary Size

```bash
$ size build/test_apu  # Includes Providence

   text    data     bss     dec     hex   filename
   5688     776       8    6472    1948   test_apu
```

Providence adds ~1 KB to the binary.

---

## The Principle

> *"Traditional memory: you ask for #47, you get #47."*
>
> *"Providence: you ask for 'something like this', you get the best match."*

Providence is memory that understands similarity. The lookup is a frozen shape. The storage is data. The result is exactly what you were looking for (approximately).

---

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   "How do you find something when you don't know            │
│    where it is?"                                            │
│                                                             │
│   "You describe it. Providence finds it for you."           │
│                                                             │
│   Content-addressed memory. Hamming distance lookup.        │
│   Soft attention blending. All frozen shapes.               │
│                                                             │
│   In Providence, addresses find you.                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
