from numpy import *

# y = mx + b
# m is slope, b is y-intercept

def compute_error_for_line_given_points(b, m , data):
    total_error = 0
    for i in range(len(data)):
        x = data[i,0]
        y = data[i,1]
        total_error += (y - (m * x + b)) ** 2
    return total_error/float(len(data))


def step_gradient(b_current, m_current, data, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(data))
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        b_gradient += (1/N) * ((m_current * x + b_current) - y)
        m_gradient += (1/N) * ((m_current * x + b_current) - y) * x
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m



def gradient_descent_runner(starting_b, starting_m, data, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(data), learning_rate)
    return b, m


def run():
    data = genfromtxt('data.csv', delimiter=",")
    initial_b = 0
    initial_m = 0
    learning_rate = 0.0001
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1} and error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, data)))
    print("Running............")
    [b, m] = gradient_descent_runner(initial_b, initial_m, data, learning_rate, num_iterations)
    print("Starting gradient descent at b = {0}, m = {1} and error = {2}".format(b, m, compute_error_for_line_given_points(b, m, data)))


if __name__ == '__main__':
    run()
