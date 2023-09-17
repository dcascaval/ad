# Infrastructure: Operator overloading

def lift(value): 
  if isinstance(value, Diff):
    return value # Already a node 
  else:
    return Constant(value) # Make constant

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
    self.uses = []

class Variable(Diff):
  def __init__(self, name: str, value: float):
    self.name = name
    self.value = value
    self.uses = []
    
class Add(Diff): 
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value + b.value
    self.parents = self.use(a, b)
    self.uses = []

class Sub(Diff): 
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value - b.value
    self.parents = self.use(a, b)
    self.uses = []

class Mul(Diff):
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value * b.value
    self.parents = self.use(a, b)
    self.uses = []

class Div(Diff):
  def __init__(self, a: Diff, b: Diff):
    self.value = a.value / b.value
    self.parents = self.use(a, b) 
    self.uses = []


def foo(x, y): 
  return (x * x) + (y - x) + 2


def main(): 
  x = Variable("x", 10) 
  y = Variable("y", 20)
  # result = x^2 + (y-x) + 2
  result = foo(x, y)

  print(result)
  print(result.value)
  print(foo(10, 20))

  # same as before
  print(result.uses) # None 
  print(result.parents) # Add, Constant
  print(result.parents[0].uses) # Result

main()