Rules:
- same random start points

Goal:
- need less running time to get a better/same result when using perturbation to a 2 or so hidden layers only.


Sep. 4th:
- Define a neural network model with two and three hidden layers.
- Set training parameters: number of epochs, number of trials, and number of epochs per trial.
- Perturbation logic: Calculate the distance between the initial weights of the previous trial set and the final output weights. Multiply the scale (0.1) to create the +-perturbation range (upper bound and lower bound). After obtaining the upper bound and lower bound, add a new input weight for the next trial set by choosing a random value within the specified range.

- `benchmark_2layers.py`: Total time taken for all trials: 113.90 seconds; avg. accuracy: 0.5103; avg. train loss: 1.4080; avg. test loss: 1.4476; avg. std. train loss: 0.1366; avg. std. 
 test loss: 0.0413
- `perturbation_2layers.py`: Total time taken for all trials: 117.47 seconds; avg. accuracy: 0.5094; avg. train loss: 1.4073; avg. test loss: 1.4445; avg. std. train loss: 0.1361; avg. std. test loss: 0.0435
- `benchmark_3layers.py`: Total time taken for all trials: 234.04 seconds; avg. accuracy: 0.5233; avg. train loss: 1.3799; avg. test loss: 1.4162; avg. std. train loss: 0.1580; avg. std. test loss: 0.0503
- `perturbation_3layers.py`: Total time taken for all trials: 234.55 seconds; avg. accuracy: 0.5232; avg. train loss: 1.3837; avg. test loss: 1.4172; avg. std. train loss: 0.1579; avg. std. test loss: 0.0527 

Summary:
- The total time taken for all trials is slightly higher in the perturbation experiments.
- For both benchmark and perturbation experiments, the 3-layer models exhibit slightly lower average train and test losses compared to the 2-layer models. 
- The perturbation experiments show very minor changes in the metrics compared to the benchmark experiments. This indicates that the perturbations applied to the weights were relatively small and did not cause significant deviations in the training process.
- The results are generally consistent between benchmark and perturbation experiments. This suggests that the introduced perturbations did not lead to significant changes in the overall model performance or behavior.

Future: 
- How many nodes? ; 1 layer & 5 layers; variability (quality); 30 trials
- Experiment with different hidden layer node combinations
- Adaptive perturbation from larger scale to smaller scale
