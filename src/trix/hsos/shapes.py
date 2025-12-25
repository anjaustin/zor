"""
Frozen Shapes for HSOS Processors.

Pure Python. No dependencies. Mathematical truths.

Each shape is a function: (state, inputs) -> (state, outputs)
Shapes are frozen - they never change.
"""

from typing import Dict, Any, Tuple, Callable

# Type alias for shape functions
ShapeFunc = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]


class Shape:
    """
    A frozen shape - pure computation with no learnable parameters.

    Shapes transform (state, inputs) -> (state, outputs).
    The shape IS the computation.
    """

    def __init__(self, name: str, func: ShapeFunc, doc: str = ""):
        self.name = name
        self.func = func
        self.doc = doc

    def __call__(self, state: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.func(state, inputs)

    def __repr__(self):
        return f"Shape({self.name})"


# =============================================================================
# PRIMITIVE SHAPES - The Axioms
# =============================================================================

def _xor_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """XOR: a + b - 2ab (exact on binary)"""
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)
    result = a + b - 2 * a * b
    return state, {'result': result}


def _and_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """AND: ab"""
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)
    result = a * b
    return state, {'result': result}


def _or_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """OR: a + b - ab"""
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)
    result = a + b - a * b
    return state, {'result': result}


def _not_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """NOT: 1 - a"""
    a = inputs.get('a', 0)
    result = 1 - a
    return state, {'result': result}


def _identity_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """Identity: pass through"""
    return state, inputs.copy()


# =============================================================================
# COMPOSITE SHAPES - Built from Axioms
# =============================================================================

def _full_adder_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """
    Full adder: (a, b, c_in) -> (sum, c_out)

    sum = a XOR b XOR c
    c_out = (a AND b) OR ((a XOR b) AND c)
    """
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)
    c = inputs.get('c', 0)

    # XOR chain for sum
    p = a + b - 2 * a * b  # a XOR b
    s = p + c - 2 * p * c  # p XOR c

    # Carry
    ab = a * b
    pc = p * c
    c_out = ab + pc - ab * pc  # ab OR pc

    return state, {'sum': s, 'carry': c_out}


def _half_adder_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """Half adder: (a, b) -> (sum, carry)"""
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)

    s = a + b - 2 * a * b  # XOR
    c = a * b              # AND

    return state, {'sum': s, 'carry': c}


def _accumulator_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """
    Accumulator: adds input to state, outputs current value.

    State: {'acc': current_value}
    Input: {'value': delta}
    Output: {'result': new_value}
    """
    acc = state.get('acc', 0)
    delta = inputs.get('value', 0)
    new_acc = (acc + delta) & 0xFF  # 8-bit wrap

    new_state = {'acc': new_acc}
    return new_state, {'result': new_acc}


def _counter_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """
    Counter: increments on each call.

    State: {'count': current}
    Input: {} (ignored)
    Output: {'count': current}
    """
    count = state.get('count', 0)
    new_count = (count + 1) & 0xFF

    return {'count': new_count}, {'count': count}


def _latch_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """
    Latch: stores value when 'set' is 1.

    State: {'value': stored}
    Input: {'data': value, 'set': 0/1}
    Output: {'value': stored}
    """
    stored = state.get('value', 0)
    data = inputs.get('data', 0)
    set_flag = inputs.get('set', 0)

    # If set, store new value
    new_value = data if set_flag else stored

    return {'value': new_value}, {'value': new_value}


def _compare_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """
    Compare: outputs routing signal based on comparison.

    Input: {'a': val1, 'b': val2}
    Output: {'lt': a<b, 'eq': a==b, 'gt': a>b}
    """
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)

    # Soft comparison (works on continuous values too)
    diff = a - b
    lt = 1 if diff < 0 else 0
    eq = 1 if diff == 0 else 0
    gt = 1 if diff > 0 else 0

    return state, {'lt': lt, 'eq': eq, 'gt': gt}


def _mux_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """
    Multiplexer: selects one of two inputs.

    Input: {'a': val1, 'b': val2, 'sel': 0/1}
    Output: {'result': selected}
    """
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)
    sel = inputs.get('sel', 0)

    # sel=0 -> a, sel=1 -> b
    result = a * (1 - sel) + b * sel

    return state, {'result': result}


def _hamming_shape(state: Dict, inputs: Dict) -> Tuple[Dict, Dict]:
    """
    Hamming distance: popcount(a XOR b).

    For 8-bit values.
    """
    a = inputs.get('a', 0)
    b = inputs.get('b', 0)

    # XOR and count bits
    xored = a ^ b
    count = bin(xored).count('1')

    return state, {'distance': count}


# =============================================================================
# SHAPE REGISTRY
# =============================================================================

SHAPES: Dict[str, Shape] = {
    # Primitives
    'xor': Shape('xor', _xor_shape, "XOR: a + b - 2ab"),
    'and': Shape('and', _and_shape, "AND: ab"),
    'or': Shape('or', _or_shape, "OR: a + b - ab"),
    'not': Shape('not', _not_shape, "NOT: 1 - a"),
    'identity': Shape('identity', _identity_shape, "Pass through"),

    # Arithmetic
    'full_adder': Shape('full_adder', _full_adder_shape, "Full adder with carry"),
    'half_adder': Shape('half_adder', _half_adder_shape, "Half adder"),

    # Stateful
    'accumulator': Shape('accumulator', _accumulator_shape, "Running sum"),
    'counter': Shape('counter', _counter_shape, "Increment counter"),
    'latch': Shape('latch', _latch_shape, "Store on set"),

    # Routing
    'compare': Shape('compare', _compare_shape, "Compare two values"),
    'mux': Shape('mux', _mux_shape, "Select one of two"),
    'hamming': Shape('hamming', _hamming_shape, "Hamming distance"),
}


def get_shape(name: str) -> Shape:
    """Get a shape by name."""
    if name not in SHAPES:
        raise ValueError(f"Unknown shape: {name}. Available: {list(SHAPES.keys())}")
    return SHAPES[name]


def register_shape(name: str, func: ShapeFunc, doc: str = "") -> Shape:
    """Register a custom shape."""
    shape = Shape(name, func, doc)
    SHAPES[name] = shape
    return shape
