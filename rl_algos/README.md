# RL Algorithms for Social Games
This directory contains existing SOTA DeepRL algorithms for incentive design in a social game. 
For ease of use, this setup uses [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) for implementations of said algorithms.

## Setup Instructions
You will need to install TensorFlow, TensorBoard, our gym-socialgame environment, & stable-baselines.

Run the following command to install our gym environment within your virtual environment
    
    cd ../gym-socialgame/
    pip install -e .
    
Then, run the following command to install tensorflow & TensorBoard, stable-baselines: 

    cd ../rl_algos
    pip install --upgrade tensorflow
    pip install stable-baselines

For parallelization capabilities with [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html), you will need OpenMPI.
To utilize OpenMPI with [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) run:

    pip install stable-baselines[mpi]

