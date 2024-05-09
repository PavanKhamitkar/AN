class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match the number of weights")

        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return 1 if weighted_sum >= self.threshold else 0

def main():
    # Define weights and threshold for ANDNOT function
    weights = [1, -1]
    threshold = 0

    # Create a McCulloch-Pitts neuron with the defined weights and threshold
    neuron = McCullochPittsNeuron(weights, threshold)

    # Test the neuron with different inputs
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for input_pair in inputs:
        result = neuron.activate(input_pair)
        print(f"ANDNOT({input_pair[0]}, {input_pair[1]}) = {result}")

if __name__ == "__main__":
    main()
