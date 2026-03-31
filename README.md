# RL Warehouse Navigation (SARSA vs Q-Learning vs Expected SARSA)

## Setup
Create a virtual environment
```bash
python3 -m venv RL_ENV
```
```bash
source RL_ENV/bin/activate 
```
Install necessary dependencies
```bash
pip3 install -r requirements.txt
```

## Run the code

### Run a Single model
```bash
python3 train.py --model q_learning --seeds 42 28 61 --episodes 12000
```
```bash
python3 train.py --model sarsa --seeds 42 28 61 --episodes 12000
```
```bash
python3 train.py --model expected_sarsa --seeds 42 28 61 --episodes 12000
```

### Run all 3 models at a time
```bash
python3 train.py --model all --seeds 42 28 61 --episodes 12000
```

## Outputs
- `results/plots/*.png` : learning curves per model/seed, step comparission, success comparission, and path visualisation. 
- `results/json/*` : per-seed metrics and `summary.json` (mean ± std)

## Metrics
- successRate
- avgSteps (delivery time)
- collisions
- energy

## Notes
- Environment includes pickup location, charging station, static obstacles, and delivery location. 

