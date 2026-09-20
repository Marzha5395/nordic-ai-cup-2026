# Survival simulator — how to run

Serves the agent controller on port **9052**; submit `http://<your-host>:9052/predict`.

```sh
cd survival-simulator
source .venv-sim/bin/activate
python agent_server.py
```

Run one worker only: the controller keeps per-simulation state and resets itself when a new run
starts. The policy is CPU-only and needs no GPU; it decides in about 5 ms per tick, well inside the
1200 s of accumulated wait allowed over a 30000-tick game.

## Requirements

```sh
python -m venv .venv-sim && source .venv-sim/bin/activate
pip install -r requirements.txt
```

Python 3.12. The policy lives in `src/utils/controllers/survival_policy.py`, with its defaults in
`DEFAULT_POPULATION` and `DEFAULT_TUNING`; the endpoint and the local runner use the same ones.

## Checking it locally

```sh
python validate.py --seeds 1 2 3 4 --workers 4      # headless local games, no network submission
python validate.py --seeds 1 2 3 4 --workers 4 --policy dummy   # the supplied baseline
python -m unittest -v test_survival_policy
```

`validate.py` reports score, survival time and `decision_ms_mean` per seed. Results vary between
runs of the same seed, so compare averages over several seeds rather than single games.

To watch a game against the reference simulator instead, start the endpoint and run
`python simulation_server.py`.
