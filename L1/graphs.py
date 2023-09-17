
# Infrastructure: Building computation graphs 
class Diff(): 

  def use(self, *values):
    for value in values: 
      value.uses.append(self)
    return values
  
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
    
    
def main(): 
  x = Variable("x", 10) 
  y = Variable("y", 20)
  # result = x^2 + (y-x) + 2
  result = Add(Add(Mul(x, x), Sub(y, x)), Constant(2))

  print(result)
  print(result.value)
  print(result.uses) # None 
  print(result.parents) # Add, Constant

main()