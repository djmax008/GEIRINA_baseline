# GEIRINA_baseline
Grid2op baseline by GEIRINA

## Train it (optional)
As no weights are provided for this baselines by default (yet), you will first need to train such a baseline:

```python
import grid2op
from l2rpn_baselines.GEIRINA import train
env = grid2op.make()
res = train(env, save_path="THE/PATH/TO/SAVE/IT", iterations=100)
```

You can have more information about extra argument of the "train" function in the 
[CONTRIBUTE](/CONTRIBUTE.md) file.

## Evaluate it
Once trained, you can reload it and evaluate its performance with the provided "evaluate" function:

```python
import grid2op
from l2rpn_baselines.GEIRINA import evaluate
env = grid2op.make()
res = evaluate(env, load_path="THE/PATH/TO/LOAD/IT.h5", nb_episode=10)
```

## Mian Entrance

python geirina_baseline.py
