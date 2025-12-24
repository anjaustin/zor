"""
MDS Proof: One hash. Same result.
"""

import hashlib
from lattice import build_adder_lattice

# Build lattice ONCE
lattice = build_adder_lattice(4)

# Hash all navigation results
nav_results = []
for a in range(16):
    for b in range(16):
        inputs = {f'a{i}': (a >> i) & 1 for i in range(4)}
        inputs.update({f'b{i}': (b >> i) & 1 for i in range(4)})
        result = lattice.navigate(inputs)
        nav_results.append(sum(result[f's{i}'] << i for i in range(5)))

nav_hash = hashlib.md5(str(nav_results).encode()).hexdigest()

# Hash all arithmetic results
arith_results = [a + b for a in range(16) for b in range(16)]
arith_hash = hashlib.md5(str(arith_results).encode()).hexdigest()

# Proof
print(f"Navigation: {nav_hash}")
print(f"Arithmetic: {arith_hash}")
print(f"Match: {nav_hash == arith_hash}")
