# Forward mode Autodiff

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
  # print(order)
  return order

class Diff(): 

  def use(self, *values):
    for value in values: 
      value.uses.append(self)
    return values

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
    self.parents = []
    self.uses = []
  
  def forward(self): # New: Forward partial derivative
    self.partial = 0.0 

class Variable(Diff):
  def __init__(self, name: str, value: float):
    self.name = name
    self.value = value
    self.parents = []
    self.uses = []
  
  def set_partial(self, partial):
    self.partial = partial

  def forward(self):
    pass
    
class Add(Diff): 
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value + b.value
    self.parents = self.use(a, b)
    self.uses = []

  def forward(self): 
    a,b = self.parents
    self.partial = a.partial + b.partial

class Sub(Diff): 
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value - b.value
    self.parents = self.use(a, b)
    self.uses = []

  def forward(self): 
    a,b = self.parents
    self.partial = a.partial - b.partial

class Mul(Diff):
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value * b.value
    self.parents = self.use(a, b)
    self.uses = []

  def forward(self): 
    a,b = self.parents
    # Product rule!
    self.partial = (a.value * b.partial) + (a.partial * b.value)

class Div(Diff):
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value / b.value
    self.parents = self.use(a, b) 
    self.uses = []

  def forward(self): 
    a,b = self.parents
    # Quotient Rule!
    numerator = (b.value * a.partial) - (a.value * b.partial)
    denominator = b.value * b.value
    self.partial = numerator / denominator

# Compute the the partial (dResult / dInput), assuming the inputs have already been set.
def partial(result): 
  order = order_nodes(result)
  for node in order:
    node.forward() 
  return result.partial

# Compute the gradient [dResult / dInput_0 , ... dResult / dInput_n ]
def forward_gradient(result_node, *inputs): 
  gradient = []
  for input in inputs: 
    for other in inputs: 
      if other == input: 
        other.set_partial(1.0)
      else:
        other.set_partial(0.0)
    gradient.append(partial(result_node))
  return gradient

# "Lift" the function F to a new function that takes the same arguments and returns the gradient.
def grad(f): 
  def g(*inputs): 
    result_node = f(*inputs)
    return forward_gradient(result_node, *inputs)
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

  x.set_partial(1.0)
  y.set_partial(0.0)
  print("partial(result):", partial(z)) # dz/dx
  # dz/dx = 2x - 1 (with x = 5, this is 9)

  x.set_partial(0.0)
  y.set_partial(1.0)   
  print("partial(result):", partial(z)) # dz/dy
  # dz/dy = 1

  g = forward_gradient(z, x, y)
  print("g:", g)

  gradF = grad(foo)
  print("∇F:", gradF(x, y))

if __name__ == "__main__":
  main()