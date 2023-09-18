# Demonstrate a constrained optimization problem

from jax import jit, grad
from scipy import optimize

# Reuse some library code; written in python.
from objectives import distance

def constrain_max(value, threshold):
  return (threshold * threshold) - value # Holds as long as value < (threshold^2)

class Model:

  def __init__(self):
    # Setup some goals and some constraints when the model is initialized. 
    # This model models two endpoints reaching two targets
    self.targetA = [100, 0]
    self.targetB = [0, 100]
    self.MAX_DISTANCE = 80
    self.pointR = [30, 30]
    self.MAX_DISTANCE_B_FROM_POINT = 50
    
  # We want the two line endpoints to be close to their respective targets
  def objective(self, args):
    ax, ay, bx, by = args
    d1 = distance([ax, ay], self.targetA)
    d2 = distance([bx, by], self.targetB)
    return d1 + d2

  # But constrain that 
  #   (a) the points cannot be too far apart and
  #   (b) one point cannot be too far from a different point R.
  def constraints(self, args): 
    ax, ay, bx, by = args
    d_apart = constrain_max(distance([ax, ay], [bx, by]), self.MAX_DISTANCE) 
    d_from_point = constrain_max(distance([bx, by], self.pointR), self.MAX_DISTANCE_B_FROM_POINT)
    return [d_apart, d_from_point]


def call_scipy(model, initial_parameters, num_constraints): 
  objective = model.objective

  # Our actual constraint function has to return an MxN matrix, where 
  # - M is the number of constraints we have
  # - N is the number of parameters we are optimizing over the function. 
  # We accomplish this by calling "grad" several times. Note that because of
  # `jit` this function is only called once, which we can verify via print.
  def c(x):
    # print("Calling C", x)
    return [grad(lambda y: model.constraints(y)[i])(x) for i in range(num_constraints)]

  constraints = {
    "type": "ineq",
    "fun": jit(model.constraints), 
    "jac": jit(c) # Comment out this line to see what happens without constraint gradients...
  }

  result = optimize.minimize(jit(objective), 
                            initial_parameters, 
                            jac = jit(grad(objective)), 
                            constraints = constraints,
                            method = "SLSQP")
  print(result)
  return [float(x) for x in result.x]

def main(): 
  print("\n -- CONSTRAINED OPTIMIZATION -- ")
  initial_parameters = [10, 10, 20, 20]
  result = call_scipy(Model(), initial_parameters, 2)
  print("final parameters:", result)



main()