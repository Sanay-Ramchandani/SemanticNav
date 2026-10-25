# Research inspiration, limitations, and reproduction

**Inspiration:** Gervet et al. (2023), [Navigating to objects in the real world](https://doi.org/10.1126/scirobotics.adf6991), *Science Robotics*. In that study, a modular system achieved approximately 90% success versus 23% for an end-to-end learned system in real homes. Those are **published results from a very different experiment**, not baseline measurements for our program.

**Original engineering here:** A self-contained procedural 2D grid-world with synthetic colour-keyed RGB sensor, noisy local vision, occlusion, occupancy memory, A* goal-directed search, BFS frontier exploration, dynamic obstacle replanning, and a reactive random-walk comparison agent. No neural network, RGB-D camera, pretrained vision model or physical robot.

## How to reproduce our stricter experiment

```bash
python -m pip install -e '.[dev,viz]'
python -m pytest -q
python -m semanticnav bench --seed-start 4000 --episodes 250 --max-steps 80 --output results/strict_80_step
```

Expected for seeds 4000–4249: modular 245/250 (98.00%), reactive 58/250 (23.20%). These are paired on the same procedural seed set. The experiment's simulator is artificially simple and the reactive baseline weak; success cannot be extrapolated to the source paper's real-world benchmark. The 230-step default experiment in `results/` produced modular 250/250 and reactive 103/250 on seeds 1000–1249.

## Research roadmap

- Use more realistic camera observations and probabilistic mapping.
- Compare against stronger classical frontier explorers and published modular baselines.
- Validate on Habitat or AI2-THOR and document domain-shift failures.
- Report SPL/path length, success with fixed difficulty bins and randomization sensitivity.

## Attribution and authorship

Written with AI assistance for this project and inspired by public, established algorithms and cited research. No paper repository code was copied into the implementation. A third-party code-similarity review has not been performed, and anyone presenting this project should be precise about their own contributions.
