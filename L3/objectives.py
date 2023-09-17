
def distance(a, b): 
  dx = a[0] - b[0]
  dy = a[1] - b[1] 
  return dx * dx + dy * dy

# Optimization objective
def distance_to_point(args): 
  x,y = args
  target_point = (25, -25)
  return distance((x,y), target_point)

def distance_to_line(line, point): 
  a,b,c = line 
  x,y = point 
  numerator = (a*x) + (b*y) + c
  denominator = (a*a) + (b*b)
  return (numerator*numerator) / denominator

# Optimization objective
def linear_regression(args): 
  target_points = [(10, 2), (51, 13), (6, 6), (31, 9)]
  total = 0 
  for point in target_points: 
    total += distance_to_line(args, point)
  return total