# Differential Growth Pattern Generator

A Python implementation of the differential growth algorithm for generating organic, coral-like patterns. The algorithm simulates nodes along a curve that repel each other while being attracted to neighbors, creating naturally branching structures.

## Features

- Multiple starting shapes (circle, rectangle, triangle, star, line)
- Bounding box constraints with boundary repulsion
- Smooth Bezier curve output (Catmull-Rom interpolation)
- Spatial hashing for O(n) performance
- Self-intersection detection and prevention
- PyPy compatible for 8-10x speedup

## Requirements

- Python 3.8+ (CPython or PyPy)
- No external dependencies (uses only standard library)

## Quick Start

```bash
# Basic usage
python diffgrowth.py --steps 1000 --output growth.svg

# Faster with PyPy (recommended)
pypy3 diffgrowth.py --steps 2000 --output growth.svg

# Show 30 example command lines with different patterns
pypy3 diffgrowth.py --examples
```

## Parameters

### Force Parameters (0-1 scale)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--growth` | 0.5 | Outward force along curve normals |
| `--repulsion` | 0.5 | Force pushing nodes/edges apart |
| `--attraction` | 0.5 | Force pulling neighbors together |
| `--alignment` | 0.5 | Smoothing force toward neighbor midpoint |
| `--noise` | 0.1 | Random perturbation per step |
| `--damping` | 0.5 | Velocity dampening (0=instant stop, 1=no friction) |

### Shape & Constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--shape` | circle | Starting shape: `circle`, `rectangle`, `triangle`, `star`, `line` |
| `--bounds` | none | Bounding region: `MIN_X MIN_Y MAX_X MAX_Y` |
| `--bound-shape` | rectangle | Shape of bound: `rectangle`, `circle`, `star` |
| `--boundary-repulsion` | 0 | Force pushing away from bounds (0-1) |

### Other Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steps` | 200 | Number of simulation steps |
| `--initial-nodes` | 20 | Starting node count |
| `--seed` | random | Random seed for reproducibility |
| `--width` | 800 | Canvas width |
| `--height` | 800 | Canvas height |
| `--no-intersection-check` | off | Disable intersection checking (faster) |
| `--safe-mode` | off | Auto-adjust repulsion for safe fast mode |
| `--output` | growth.svg | Output filename |
| `--examples` | off | Print 30 example command lines and exit |

## Style Recipes

### Dense Branching
```bash
pypy3 diffgrowth.py --steps 2000 --growth 0.7 --repulsion 0.8 --noise 0.15 --alignment 0.5 --no-intersection-check
```

### Regular/Symmetric
```bash
pypy3 diffgrowth.py --steps 2000 --growth 0.5 --repulsion 0.6 --noise 0.05 --alignment 0.7 --no-intersection-check
```

### Organic/Natural
```bash
pypy3 diffgrowth.py --steps 2000 --growth 0.6 --repulsion 0.65 --noise 0.18 --alignment 0.5 --no-intersection-check
```

### Sparse/Minimal
```bash
pypy3 diffgrowth.py --steps 2000 --growth 0.4 --repulsion 0.4 --noise 0.1 --alignment 0.5 --no-intersection-check
```

## Shape Examples

```bash
# Rectangle starting shape
pypy3 diffgrowth.py --shape rectangle --steps 1000 --output rectangle.svg

# Triangle
pypy3 diffgrowth.py --shape triangle --steps 1000 --output triangle.svg

# Star
pypy3 diffgrowth.py --shape star --steps 1500 --growth 0.7 --repulsion 0.8 --output star.svg

# Line (open curve that grows outward)
pypy3 diffgrowth.py --shape line --steps 1000 --output line.svg
```

## Bounding Shape Examples

```bash
# Constrained to rectangular bounds
pypy3 diffgrowth.py --bounds 100 100 700 700 --bound-shape rectangle --boundary-repulsion 0.8 --steps 1000 --output bound_rect.svg

# Constrained to circular bounds
pypy3 diffgrowth.py --bounds 100 100 700 700 --bound-shape circle --boundary-repulsion 0.8 --steps 1000 --output bound_circle.svg

# Constrained to star-shaped bounds (use higher repulsion to avoid intersections)
pypy3 diffgrowth.py --bounds 50 50 750 750 --bound-shape star --boundary-repulsion 1.0 --repulsion 0.9 --steps 1000 --output bound_star.svg
```

## Understanding Parameters

### Growth vs Repulsion Ratio

This is the key driver of pattern complexity:

| Ratio | Effect |
|-------|--------|
| growth < repulsion | Expands freely, many branches |
| growth ≈ repulsion | Balanced branching |
| growth > repulsion | Compact, limited growth |

### Noise Effect

- **0.05**: Smooth, symmetric patterns
- **0.10**: Slightly varied, natural-looking
- **0.15**: Noticeable asymmetry, diverse branch sizes
- **0.25**: Chaotic, highly irregular (may cause intersections)

### Safe Parameter Constraint

For intersection-free results without checking:
```
repulsion > 0.2 * growth + 3 * noise
```

Use `--safe-mode` to auto-adjust repulsion, or check the warning message for recommended values.

## Performance

| Interpreter | 1000 steps | Relative |
|-------------|------------|----------|
| CPython 3.x | ~30s | 1x |
| PyPy 3.x | ~3s | 10x |

Use PyPy for best performance. Install with: `brew install pypy3` (macOS) or `apt install pypy3` (Linux).

## Output

The generator produces SVG files with:
- Smooth Bezier curves (Catmull-Rom to cubic Bezier conversion)
- Black fill with red stroke (configurable in code)
- Auto-fitted viewBox

## Algorithm Overview

1. **Initialize** nodes along starting shape
2. **Each step**:
   - Calculate forces: attraction, repulsion, alignment, growth, noise, boundary
   - Apply velocity with damping
   - Optionally check for self-intersections
   - Split edges that exceed max length
3. **Export** smooth curve as SVG

## License

MIT
