# TriX Onramp

A guided journey from zero to mastery.

## Start Here

```bash
cd /workspace/ZOR
python onramp/00_witness.py
```

Watch. Then follow the prompts.

## The Path

| Script | What You'll Learn | Time |
|--------|-------------------|------|
| `00_witness` | See the 6502 run | 2 min |
| `01_touch` | Run your first shape | 3 min |
| `02_truth` | Verify the math | 5 min |
| `03_build` | Create a shape | 10 min |
| `04_compose` | Build an ALU | 15 min |
| `05_route` | Add learned routing | 20 min |
| `06_export` | Generate C code | 10 min |
| `07_deploy` | Run on Thor | 10 min |
| `08_explore` | Build something | ∞ |

**Total time to mastery: ~75 minutes**

## Requirements

- Python 3.10+
- gcc (for `07_deploy.py`)
- Thor/aarch64 (for native deployment, but works on any platform)

No PyTorch. No TensorFlow. No external ML frameworks.
TriX Native uses CuPy (GPU) or NumPy (CPU) internally.

## Philosophy

This isn't a tutorial. It's a journey.

Each script answers a question you're already asking.
The code is minimal. The comments teach.
By the end, you won't just know how. You'll know why.

*Computation is geometry. Learning is routing.*

## If You Get Stuck

Each script is self-contained. You can always:
- Re-read the comments
- Run individual sections
- Check the expected output

There's no rush. Mastery over speed.
