Sep. 4th:

- Define a neural network model with two and three hidden layers.
- Set training parameters: number of epochs, number of trials, and number of epochs per trial.
- Perturbation logic: Calculate the distance between the initial weights of the previous trial set and the final output weights. Multiply the scale (0.1) to create the +-perturbation range (upper bound and lower bound). After obtaining the upper bound and lower bound, add a new input weight for the next trial set by choosing a random value within the specified range.


`benchmark_2layers.py`: Total time taken for all trials: 225.48 seconds 
`perturbation_2layers.py`: Total time taken for all trials: 234.42 seconds
`benchmark_3layers.py`: Total time taken for all trials: 244.26 seconds 
`perturbation_3layers.py`: Total time taken for all trials: 243.17 seconds   