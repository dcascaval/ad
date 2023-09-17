# Now we will do all of that, but with JAX to do the heavy lifting.

from jax import jit, grad
from scipy import optimize

# Still the same objective functions!
from objectives import distance_to_point, linear_regression

def call_scipy(f, initial_parameters): 
  forward = jit(f)
  derivative = jit(grad(f))
  result = optimize.minimize(forward, initial_parameters, jac = derivative, method = "SLSQP")
  print(result)
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

main()