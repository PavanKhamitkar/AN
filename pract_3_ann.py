def step_function(x):
    return 1 if x >= 0 else 0

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def binary_representation(num):
    return [int(x) for x in f'{num:06b}']

def learn_parity(num_epochs=100):
    weights = [0, 0, 0, 0, 0, 1]
    
    for epoch in range(num_epochs):
        for number in range(10):
            input_data = binary_representation(number)
            target = 1 if number % 2 == 0 else 0
            output = step_function(dot_product(input_data, weights))
            error = target - output

            for i in range(len(weights)):
                weights[i] += input_data[i] * error

    return weights

def predict_parity(num, weights):
    input_data = binary_representation(num)
    output = step_function(dot_product(input_data, weights))
    return "odd" if output == 0 else "even"

if __name__ == "__main__":
    weights = learn_parity()
    while True:
        num = input("Enter a number (0-9) or 'q' to quit: ")
        if num.lower() == 'q':
            break
        try:
            number = int(num)
            if not 0 <= number <= 9:
                raise ValueError("Number must be between 0 and 9.")
            prediction = predict_parity(number, weights)
            print(number, "is", prediction)
        except ValueError as e:
            print(e)
