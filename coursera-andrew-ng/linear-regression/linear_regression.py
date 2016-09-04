
# Hypothesis function
# h(x) = theta_0 + theta_1 * x
# theta_0 = y intercept
# theta_1 = slope
learning_rate = 0.5 # alpha

error_forgivance_rate = 0.00

def calculate_cost_function(theta_0, theta_1, X, Y):
  """
  Cost function chosen for linear regression is 
  RMS method.
  J(theta_0, theta_1) = 1/2m*(Sum over 1 to m (h(x) - y))
  """
  iterations = min(len(X), len(Y))
  result = 0

  for iteration in range(iterations):
    result += (theta_0 + theta_1 * X[iteration] - Y[iteration]) ** 2

  result / (2 * iterations)
  return round(result, 2)

def gradient_descent(X, Y, theta_0 = 0, theta_1 = 0):
  iterations = min(len(X), len(Y))

  while calculate_cost_function(theta_0, theta_1, X, Y) > error_forgivance_rate:
    accum_result_0 = 0
    accum_result_1 = 0

    for iteration in range(iterations):
      accum_result_0 += ((theta_0 + (theta_1 * X[iteration])) - Y[iteration])
      accum_result_1 += ((theta_0 + (theta_1 * X[iteration])) - Y[iteration]) * X[iteration]

    tmp_theta_0 = theta_0 - (learning_rate/iterations) * accum_result_0
    tmp_theta_1 = theta_1 - (learning_rate/iterations) * accum_result_1

    theta_0 = tmp_theta_0
    theta_1 = tmp_theta_1

  return (theta_0, theta_1)

if __name__ == "__main__":
  X = [1, 2, 3, 4, 5]
  Y = [1, 2, 3, 4, 5]

  (theta_0, theta_1) = gradient_descent(X, Y)
  print "Slope = ", theta_1

