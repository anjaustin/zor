# Reflect: Freshman Onboarding

## Core Insight

**The current tutorials TELL. They should ASK.**

Every explanation should be a question the freshman answers themselves:
- "What happens if you change 2 to 3?" → They discover the formula matters
- "Can you make a NAND gate?" → They discover composition
- "Why is the router picking XOR?" → They discover pattern matching

## The Pedagogy

Good tutorial structure:
```
1. HOOK (10 lines, surprise)
2. TRY (modify something, see effect)
3. BUILD (make something work)
4. CHALLENGE (harder problem, no solution given)
5. REFLECT (one-line "you got it when...")
```

Bad tutorial structure:
```
1. Explain concept
2. Show example
3. Explain why it matters
4. Tell them what to do next
```

## The Five-File Curriculum

Looking at the nodes, the right structure is:

### File 1: The Hook
- XOR from math. 10 lines. No explanation.
- TRY: Change 2 to 3. What breaks?
- CHALLENGE: Can you write AND the same way?

### File 2: The Build
- Full adder, then 8-bit adder
- TRY: Add two numbers. Verify.
- CHALLENGE: Can you make it 16-bit?

### File 3: The Compose
- Stack shapes. See emergent behavior.
- TRY: Rearrange the order. What changes?
- CHALLENGE: Build a multiplexer.

### File 4: The Route
- Multiple shapes, input picks the right one
- TRY: Change the signatures. Watch routing change.
- CHALLENGE: Add a new shape and make inputs route to it.

### File 5: The System
- Everything together. Train routing. Run fast.
- TRY: Break the training. See it fail.
- CHALLENGE: Invent your own shape and integrate it.

## What Dies in This Revision

- "So what" file (telling, not showing)
- "Why fast" file (merge into Build)
- "What is a shape" file (too abstract)
- All files 06-10 (consolidate into Compose and Route)
- All explanatory prose at the end of files

## What Emerges

Each file is:
- **Self-contained**: Run it, learn something, done
- **Active**: Must modify code to complete challenges
- **Progressive**: Each harder than the last
- **Satisfying**: Build real things, not toy examples

## The Real Test

A skeptical freshman who completes all 5 files should:
1. Know XOR can be math (verified by changing constants)
2. Have built an adder (can show a friend)
3. Understand composition (solved multiplexer challenge)
4. Grok routing (trained their own router)
5. See the whole system (integrated their own shape)

If they can't do #5, we failed.

## The Freshman UX Bar

**Production-grade means:**
- Zero dependency issues (just Python)
- Zero unexplained errors
- Every challenge has a hidden solution they can find if stuck
- Feels like a game, not homework
