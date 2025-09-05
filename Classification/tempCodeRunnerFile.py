import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0.5, 0.9]])
y = np.array([[1]])

np.random.seed(1)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

w_input_hidden = np.random.rand(input_neurons, hidden_neurons)
w_hidden_output = np.random.rand(hidden_neurons, output_neurons)

b_hidden = np.random.rand(1, hidden_neurons)
b_output = np.random.rand(1, output_neurons)

learning_rate = 0.5

for epoch in range(10000):
    hidden_input = np.dot(X, w_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, w_hidden_output) + b_output
    final_output = sigmoid(final_input)
    
    error = y - final_output
    
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(w_hidden_output.T) * sigmoid_derivative(hidden_output)
    
    w_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += d_output * learning_rate
    
    w_input_hidden += X.T.dot(d_hidden) * learning_rate
    b_hidden += d_hidden * learning_rate


print("Targeted output")
print(y)

print("Output after training:")
print(final_output)

print("\n Name: Dipesh Shrestha \n Roll no:08 \n")
