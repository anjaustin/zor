"""
DISCOVERY

Run this. Don't read ahead.
"""

def xor(a, b):
    return a + b - 2*a*b

print("0 XOR 0 =", int(xor(0, 0)))
print("0 XOR 1 =", int(xor(0, 1)))
print("1 XOR 0 =", int(xor(1, 0)))
print("1 XOR 1 =", int(xor(1, 1)))

# ^^^^^ That just happened.
# XOR from addition and multiplication. No ^ operator.


# ============================================================
# TRY THESE (modify the code above, run again)
# ============================================================
#
# 1. Change the 2 to a 3. What breaks?
#
# 2. Change the 2 to a 1. What do you get instead of XOR?
#
# 3. Try xor(0.5, 0.5). What happens between 0 and 1?


# ============================================================
# CHALLENGE
# ============================================================
#
# Write AND(a, b) using only +, -, *
# It's simpler than you think.
#
# Test it:
#   print(AND(0, 0))  # should be 0
#   print(AND(0, 1))  # should be 0
#   print(AND(1, 0))  # should be 0
#   print(AND(1, 1))  # should be 1


# ============================================================
# YOU GOT IT WHEN
# ============================================================
#
# You can explain to a friend why a + b - 2ab equals XOR.
# (Hint: think about what happens at each corner of the truth table)


# ============================================================
# SOLUTION (no peeking until you've tried)
# ============================================================
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def AND(a, b):
#     return a * b
#
# That's it. Multiplication IS AND for binary inputs:
#   0 * 0 = 0
#   0 * 1 = 0
#   1 * 0 = 0
#   1 * 1 = 1
#
# BONUS: What's OR?
#   OR(a, b) = a + b - a*b
#
# Now go to: python build.py
