from typing import List
from forward import partial, Variable
from objectives import distance_to_point, linear_regression

# Compute the gradient [dResult / dInput_0 , ... dResult / dInput_n ]
def forward_gradient(result_node, inputs): 
  gradient = []
  for input in inputs: 
    for other in inputs: 
      if other == input: 
        other.set_partial(1.0)
      else:
        other.set_partial(0.0)
    gradient.append(partial(result_node))
  return gradient

# Lift the function F to a new function that takes the same arguments and returns the gradient.
def grad(f): 
  def g(inputs): 
    result_node = f(inputs)
    return forward_gradient(result_node, inputs)
  return g

def optimize(f, initial_parameters: List[float]): 
  step_size = 0.1
  max_steps = 10000
  num_steps = 0 

  g = grad(f)
  parameters = list(initial_parameters)
  variables = [ Variable(f"x_{i}", p) for (i, p) in enumerate(initial_parameters) ] 
  loss = f(parameters)

  # Iterate in the direction of the negative gradient until we converge
  while loss >= 1e-3 and num_steps < max_steps: 
    gradient = g(variables)
    for i in range(len(gradient)): 
      parameters[i] = parameters[i] - (gradient[i] * step_size)
      variables[i].value = parameters[i]

    # Check if we're still doing anything
    next_loss = f(parameters)
    if abs(next_loss - loss) <= 1e-3:
      break

    loss = next_loss
    num_steps += 1
    if num_steps % 10 == 0: 
      print(f"[step {num_steps}] loss = {loss}, gradient = {gradient}")
  
  # Lots of additional things we could do here: 
  # - Try several step sizes at each gradient iteration (line-search)
  # - Track "best" found point-so-far
  # - Quadratic approximations

  return parameters

def main(): 
  print("\n -- Distance to Point -- ")
  initial_parameters = [-10, 10]
  result = optimize(distance_to_point, initial_parameters)
  print("final parameters:", result)

  print("\n -- Linear Regression -- ")
  initial_parameters = [1, 1, 0]
  result = optimize(linear_regression, initial_parameters)
  print("final parameters:", result)
  a,b,c = result 
  print(f"y = {-a/b:.2f}x + {-c/b:.2f}")

if __name__ == "__main__":
  main()