# Yahtzee
Building an On Policy RL algorithm to play Yahtzee

## Setup
 - pip install -r requirements.txt

## Core Functionality
### Training a Model

To train a model, run:
```
python trainer.py
```

### Running the App

To run the interactive Yahtzee app, use:
```
python app.py
```

## Optimal Results Model Configuration:
```
python trainer.py \
    --batch_size 8192 \
    --num_steps 10000 \
    --policy_loss_coefficient 100.0 \
    --value_loss_coefficient 0.01 \
    --entropy_loss_coefficient 1.0 \
```

## Example Runs

Here are example runs using the optimal configuration:

- [Run 1](https://wandb.ai/ashikari123-shikari-projects/yahtzee-rl/runs/td9oegs6?nw=nwuserashikari123) - Example training run with optimal parameters
- [Run 2](https://wandb.ai/ashikari123-shikari-projects/yahtzee-rl/runs/3puuyyes?nw=nwuserashikari123) - Alternative training run with optimal parameters

All experiments can be compared on the [Weights & Biases project page](https://wandb.ai/ashikari123-shikari-projects/yahtzee-rl?nw=nwuserashikari123).


## TODO
- [ ] Clean up the State class to group features into dicts
- [ ] Implement UI for calculation mode
- [ ] Model saving / Loading
- [ ] Model Store via hugging face
- [ ] Experiment Management / Comparison Improvements

## References
- [Reinforcement Learning for Yahtzee](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf) - Explores using Deep Q-Learning and Policy Gradient methods to train an AI agent to play Yahtzee
- [Optimal Play in Yahtzee](http://www.yahtzee.org.uk/optimal_yahtzee_TV.pdf) - Mathematical analysis of optimal Yahtzee strategies and expected values
- [Yahtzee Q-Learning Implementation](https://github.com/marcchen2/yahtzee_qlearning) - Example implementation of Q-learning applied to Yahtzee


