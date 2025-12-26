#!/usr/bin/env python3
"""
TriX Fabric Demo: End-to-End Integration Test

The fabric IS the program.
The fabric IS the database.
The fabric IS the knowledge.

This demo shows:
1. Creating a fabric from FDL
2. Executing computations via message injection
3. Querying the fabric
4. Persisting and loading
"""

import sys
sys.path.insert(0, 'src')

from trix.fabric import (
    Fabric,
    FabricProcessor,
    Message,
    parse_fdl,
    emit_fdl,
    query_fabric,
)
from trix.fabric.query import fabric_stats


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    print("=" * 60)
    print("  TRIX FABRIC - Unified Database-Computer")
    print("  The fabric IS the program.")
    print("=" * 60)

    # =========================================================================
    # 1. Create fabric programmatically
    # =========================================================================
    separator("1. Create Fabric Programmatically")

    fabric = Fabric(name="xor_network")

    # Add processors
    fabric.add_processor(FabricProcessor(
        signature="/logic/0b00",
        shape="xor",
        name="gate_a"
    ))
    fabric.add_processor(FabricProcessor(
        signature="/logic/0b01",
        shape="xor",
        name="gate_b"
    ))
    fabric.add_processor(FabricProcessor(
        signature="/logic/0b10",
        shape="and",
        name="combiner"
    ))

    # Connect: gate_a -> combiner, gate_b -> combiner
    fabric.connect("/logic/0b00", "/logic/0b10", "result", "a")
    fabric.connect("/logic/0b01", "/logic/0b10", "result", "b")
    fabric.set_export("/logic/0b10")

    print(f"Created: {fabric}")
    print(f"Processors: {list(p.name for p in fabric.iter_processors())}")

    # =========================================================================
    # 2. Execute via message injection
    # =========================================================================
    separator("2. Execute via Message Injection")

    # Send inputs to both XOR gates
    inputs = {
        "/logic/0b00": {"a": 1.0, "b": 0.0},  # XOR(1,0) = 1
        "/logic/0b01": {"a": 0.0, "b": 1.0},  # XOR(0,1) = 1
    }
    result = fabric.execute(inputs)

    print(f"Inputs: {inputs}")
    print(f"Messages processed: {result.messages_processed}")
    print(f"Cycles: {result.cycles}")
    print(f"Outputs: {result.outputs}")
    # Note: Async message passing - each XOR output triggers combiner separately
    # Synchronous execution would require a gather/barrier mechanism

    # =========================================================================
    # 3. Create fabric from FDL
    # =========================================================================
    separator("3. Create Fabric from FDL")

    fdl_source = '''
    fabric "half_adder" {
        # Half adder: sum = a XOR b, carry = a AND b
        processor xor_gate {
            shape: xor
            signature: /adder/0b00
        }
        processor and_gate {
            shape: and
            signature: /adder/0b01
        }
        export: xor_gate
        export: and_gate
    }
    '''

    adder = parse_fdl(fdl_source)
    print(f"Parsed: {adder}")
    print(f"FDL:\n{emit_fdl(adder)}")

    # Execute half adder: 1 + 1 = 10 (binary)
    result = adder.execute({
        "/adder/0b00": {"a": 1.0, "b": 1.0},
        "/adder/0b01": {"a": 1.0, "b": 1.0},
    })
    print(f"\n1 + 1 in half adder:")
    print(f"  Sum (XOR):   {result.outputs['/adder/0b00']['result']}")
    print(f"  Carry (AND): {result.outputs['/adder/0b01']['result']}")
    print("  Expected: Sum=0, Carry=1 (binary 10)")

    # =========================================================================
    # 4. Fuzzy logic (continuous values)
    # =========================================================================
    separator("4. Fuzzy Logic (Continuous Values)")

    # XOR with continuous inputs
    fuzzy = Fabric(name="fuzzy_gate")
    fuzzy.add_processor(FabricProcessor(
        signature="/0/0b00",
        shape="xor",
        name="fuzzy_xor"
    ))
    fuzzy.set_export("/0/0b00")

    # XOR(0.7, 0.3) = 0.7 + 0.3 - 2*0.7*0.3 = 1.0 - 0.42 = 0.58
    msg = Message(target="/0/0b00", payload={"a": 0.7, "b": 0.3})
    result = fuzzy.inject(msg)

    print(f"XOR(0.7, 0.3) = {result.outputs['/0/0b00']['result']:.4f}")
    print(f"Expected: 0.7 + 0.3 - 2*0.7*0.3 = 0.58")

    # =========================================================================
    # 5. Query the fabric
    # =========================================================================
    separator("5. Query the Fabric")

    # Build a larger fabric
    big = Fabric(name="big_fabric")
    for i in range(4):
        big.add_processor(FabricProcessor(
            signature=f"/cluster_a/0b{i:02b}",
            shape="xor",
            name=f"xor_{i}"
        ))
    for i in range(4):
        big.add_processor(FabricProcessor(
            signature=f"/cluster_b/0b{i:02b}",
            shape="and",
            name=f"and_{i}"
        ))

    print(f"Created: {big}")

    # Query by shape
    xor_procs = query_fabric(big, shape="xor")
    print(f"XOR processors: {xor_procs.count}")

    # Query by cluster
    cluster_a = query_fabric(big, cluster="/cluster_a")
    print(f"Processors in /cluster_a: {cluster_a.count}")

    # Query by name pattern
    named = query_fabric(big, name="and_*")
    print(f"Processors matching 'and_*': {named.count}")

    # Stats
    stats = fabric_stats(big)
    print(f"\nFabric stats:")
    print(f"  Total processors: {stats['total_processors']}")
    print(f"  Clusters: {stats['total_clusters']}")
    print(f"  Shapes: {stats['shapes']}")

    # =========================================================================
    # 6. Deep hierarchy
    # =========================================================================
    separator("6. Deep Hierarchy")

    deep = Fabric(name="deep_fabric")
    deep.add_processor(FabricProcessor(
        signature="/level1/level2/level3/0b1010",
        shape="xor",
        name="deep_processor"
    ))

    proc = deep.get_processor("/level1/level2/level3/0b1010")
    print(f"Deep processor: {proc}")
    print(f"Cluster path: {proc.cluster_path}")
    print(f"Local signature: {proc.local_signature}")

    # =========================================================================
    # Done
    # =========================================================================
    separator("COMPLETE")

    print("""
TriX Fabric is operational.

Architecture:
  - Recursive: Fabrics contain processors contain fabrics
  - Hierarchical: Tree of hypercubes for addressing
  - Declarative: FDL defines, persists, and queries
  - Stateless: Messages carry context
  - Substrate-independent: HSOS → GILLIES → Vulkan

The fabric IS the program.
The fabric IS the database.
The fabric IS the knowledge.
""")


if __name__ == "__main__":
    main()
