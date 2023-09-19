# Calling Scipy for optimization instead of our home-grown gradient descent

from optimization import forward_gradient, Variable
from typing import List
from scipy import optimize

# Re-use the same code.
from objectives import distance_to_point, linear_regression

# Lift the function F to a new function that takes the same arguments and returns the gradient.
# Here we are now "hiding" the symbolic variable infrastructure inside the `grad()` function. 
def grad(f): 
  def g(inputs):
    variables = [ Variable(f"x_{i}", p) for (i, p) in enumerate(inputs) ] 
    result_node = f(variables)
    return forward_gradient(result_node, variables)
  return g

def call_scipy(f, initial_parameters: List[float]): 
  result = optimize.minimize(f, initial_parameters, jac = grad(f), method = "SLSQP")
  return [float(x) for x in result.x]

def main(): 
  print("\n -- Distance to Point -- ")
  initial_parameters = [-10, 10]
  result = call_scipy(distance_to_point, initial_parameters)
  print("final parameters:", result)

  print("\n -- Linear Regression -- ")
  initial_parameters = [1, 1, 0]
  result = call_scipy(linear_regression, initial_parameters)
  print("final parameters:", result)
  a,b,c = result 
  print(f"y = {-a/b:.2f}x + {-c/b:.2f}")

if __name__ == "__main__":
  main()