The selected random weights for the next epoch should differ by at least 10% of the distance between the output weights of the current epoch and the input weights used in the same epoch. These input weights are the ones without any perturbation. Additionally, the chosen random weights for the next epoch must fall within a certain range: [original input weights of the next epoch - distance, original input weights of the next epoch + distance]. If a random weight goes beyond this range, it will be adjusted to be exactly 10% of the calculated distance away from the original input weights of the next epoch, ensuring controlled and meaningful changes to the learning process.

08/20: 
- The perturbed model is struggling to learn the patterns in the data, leading to a poor fit.

Next Step:
- Reduce perturbation scale and bounds
- Experiment with other optimizers (gradients first)
- Lower learning rate


Future: 
- Adaptive perturbation from larger scale to smaller scale