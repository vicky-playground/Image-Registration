The selected random weights for the next epoch should differ by at least 10% of the distance between the output weights of the current epoch and the input weights used in the same epoch. These input weights are the ones without any perturbation. Additionally, the chosen random weights for the next epoch must fall within a certain range: [original input weights of the next epoch - distance, original input weights of the next epoch + distance]. If a random weight goes beyond this range, it will be adjusted to be exactly 10% of the calculated distance away from the original input weights of the next epoch, ensuring controlled and meaningful changes to the learning process.

08/20: 
- The perturbed model is struggling to learn the patterns in the data, leading to a poor fit.

Next Step:
- get the basic baseline results regarding running time and see the metrics change (avg. loss, std. loss, and accuracy) for using only one/two hidden layers 
- move the perturbation to restart another whole set of trial only.

Goal:
- need less running time to get a better/same result when using perturbation to a 2 or so hidden layers only.

Future: 
- Adaptive perturbation from larger scale to smaller scale
