**Status:** Work in progress. SocialGameEnv is working and ready-to-use.

TODO:
* DR Implementations:
    -[ ] Review Noise Model implementation
    -[ ] Algorithmic DR implementation(s) ?
- [ ] Hourly Environment (finish implementing hourly agents)
- [ ] DR Environments for Hourly and Monthly variants


# gym-socialgame
OpenAI Gym environment for a social game.

## Usage
This environment can be directly plugged into existing OpenAI-Gym compatible RL libraries (e.g. [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html)).
For this case, try the following:
    
    pip install -e .

Otherwise, this package can be used as a standalone environment for other RL algorithms.
