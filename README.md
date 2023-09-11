Rules:
- same random start points

Goal:
- need less running time to get a better/same result when using perturbation to a 2 or so hidden layers only.


Sep. 4th:
- Define a neural network model with different (1-3, 5) layers.
- Set training parameters: 10 epochs and 30 trial.
- Perturbation logic: Calculate the distance between the initial weights of the previous trial set and the final output weights. Multiply the scale (0.1) to create the +-perturbation range (upper bound and lower bound). After obtaining the upper bound and lower bound, add a new input weight for the next trial set by choosing a random value within the specified range.

## Benchmark
# of Layers | Total time taken for all trials | avg. acc. | avg. train loss | avg. test loss | avg. std. train loss | avg. std. test loss
---|---|---|---|---|---|---|
1 | 1921.74 | 0.3877 | 2.1772 | 2.2148 | 0.1698 | 0.0943 |
2 | 1797.44 | 0.3877 | 2.1772 | 2.2148 | 0.1698 | 0.0943 |
3 | 1991.36 | 0.5304 | 1.2152 | 1.3972 | 0.2040 | 0.0435 |
5 | 2028.03 | 0.5316 | 1.2249 | 1.3935 | 0.2204 | 0.0567 |

## Perturbation  
# of Layers | Total time taken for all trials | avg. acc. | avg. train loss | avg. test loss | avg. std. train loss | avg. std. test loss
---|---|---|---|---|---|---|
1 | **1917.43** | 0.3877 | 2.1772 | 2.2146 | 0.1694 | 0.0951 |
2 | **1505.05** | **0.5162** | 1.2741 | 1.4319 | 0.1686 | 0.0346 |
3 | **1497.41** | 0.5294 | 1.2151 | 1.3974 | 0.2040 | 0.0437 |
5 | **1957.75** | 0.5315 | 1.2242 | 1.3920 | 0.2209 | 0.0559 |

Summary:
- The total time taken for all trials is slightly higher in the perturbation experiments.
- For both benchmark and perturbation experiments, the 3-layer models exhibit slightly lower average train and test losses compared to the 2-layer models. 
- The perturbation experiments show very minor changes in the metrics compared to the benchmark experiments. This indicates that the perturbations applied to the weights were relatively small and did not cause significant deviations in the training process.
- The results are generally consistent between benchmark and perturbation experiments. This suggests that the introduced perturbations did not lead to significant changes in the overall model performance or behavior.

Future: 
- How many nodes? ; 1 layer & 5 layers; variability (quality); 30 trials
- Experiment with different hidden layer node combinations
- Adaptive perturbation from larger scale to smaller scale
