# Reverse mode Autodiff

def lift(value): 
  if isinstance(value, Diff):
    return value # Already a node 
  else:
    return Constant(value) # Make constant
  
# Ancilliary: node ordering requires a topological-sort,
# which we accomplish using a depth-first search.
def order_nodes(node): 
  order = []
  permanent = set()
  temporary = set()

  # Internal DFS visitor function
  stack = []

  def visit(rootNode):
    stack.append(rootNode)

    while len(stack) != 0:
      node = stack.pop()
      if node in permanent:
        continue
      if node in temporary:
        temporary.remove(node)
        permanent.add(node)
        order.append(node)
      else:
        stack.append(node)
        temporary.add(node)
        
        # Visit A before B, purely for legibility
        for parent in reversed(node.parents): 
          stack.append(parent)

  visit(node)
  # In reverse mode, we want to compute in exactly the reverse order, 
  # to traverse the graph from the root node up towards the variables.
  return list(reversed(order))

class Diff(): 
  def __add__(self, b):
    return Add(self, lift(b))
  
  def __radd__(self, b): 
    return Add(lift(b), self)

  def __sub__(self, b):
    return Sub(self, lift(b))
  
  def __mul__(self, b):
    return Mul(self, lift(b))
  
  def __truediv__(self, b):
    return Div(self, lift(b))
  

class Constant(Diff):
  def __init__(self, value: float): 
    self.value = value
    self.adjoint = 0.0
    self.parents = []
  
  def d(self): # New: Reverse mode accumulation
    pass 

class Variable(Diff):
  def __init__(self, name: str, value: float):
    self.name = name
    self.value = value
    self.adjoint = 0.0
    self.parents = []
  
  def get_adjoint(self):
    return self.adjoint

  def d(self):
    pass
    
class Add(Diff): 
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value + b.value
    self.adjoint = 0.0
    self.parents = [a, b]

  def d(self): 
    a,b = self.parents
    a.adjoint += self.adjoint 
    b.adjoint += self.adjoint

class Sub(Diff): 
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value - b.value   
    self.adjoint = 0.0
    self.parents = [a, b]

  def d(self): 
    a,b = self.parents
    a.adjoint += self.adjoint
    b.adjoint -= self.adjoint

class Mul(Diff):
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value * b.value
    self.adjoint = 0.0
    self.parents = [a, b]

  def d(self): 
    a,b = self.parents 
    a.adjoint += self.adjoint * b.value
    b.adjoint += a.value * self.adjoint

class Div(Diff):
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value / b.value
    self.adjoint = 0.0
    self.parents = [a, b]

  def d(self): 
    a,b = self.parents
    denominator = b.value * b.value
    a.adjoint += b.value * self.adjoint / denominator
    b.adjoint += -(a.value * self.adjoint / denominator)

# Compute the gradient [dResult / dInput_0 , ... dResult / dInput_n ]
def reverse_gradient(result_node, *inputs): 
  order = order_nodes(result_node)
  for node in order:
    node.adjoint = 0.0
  result_node.adjoint = 1.0
  for node in order: 
    node.d()
  gradient = [ v.adjoint for v in inputs ]
  return gradient

# "Lift" the function F to a new function that takes the same arguments and returns the gradient.
def grad(f): 
  def g(*inputs): 
    result_node = f(*inputs)
    return reverse_gradient(result_node, *inputs)
  return g

# Our function
def foo(x, y): 
  # print(f"Calling Foo ({x},{y})")
  return (x * x) + (y - x) + 2

def main(): 
  x = Variable("x", 5) 
  y = Variable("y", 3)

  # z = x^2 + (y-x) + 2
  z = foo(x, y)
  print("result.value:", z.value)

  print("reverse(z):", reverse_gradient(z, x, y))

  gradF = grad(foo)
  print("∇F:", gradF(x, y))

if __name__ == "__main__":
  main()