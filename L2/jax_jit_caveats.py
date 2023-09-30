from jax import jit 
import jax.numpy as jnp
from timeit import default_timer as timer

def make_matrix(n):
  return [[float(i) / n for i in range(n)] for j in range(n)]

def jax_matrix(n):
  return jnp.array([jnp.array([float(i) / n for i in range(n)]) for j in range(n)])

def time(msg, f, *x): 
  start = timer()
  result = f(*x)
  end = timer()
  print(f"{msg}:", f"{(end-start) * 1000.0:.2f}ms")
  return result

# JIT is a really convenient mechanism for "speeding up" python, but it can be slwo itself: 
# The tradeoff is:
# - The first time you call the function, it will take longer, because the JAX runtime is 
#   converting it to C for you. Subsequent calls are pretty zippy.
# - Loops are OK (can be compiled) BUT big loops take a long time (why?)
#   => Instead we can use JAX's built-in numpy operations, or use the `lax` package
#      to access their control-flow primitives to use instead of python's.
# 
# grad() and friends still work without jit(), but it is usually worth calling jit()
#   because the function and gradients are evaluated many times.

def matmul(a, b): 
    n = len(a)
    total = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total += a[i][k] * b[k][j]
    return total

def matmul_n_slow(n):
  def mul(a,b):
    total = 0.0
    for i in range(n):
      for j in range(n):
        for k in range(n):
          total += a[i][k] * b[k][j]
    return total
  return mul

def matmul_fast(a, b):
  return jnp.matmul(a, b)   

def main():
  print("--- PYTHON MATMULS ---")
  time("5x5", matmul, make_matrix(5) , make_matrix(5))
  time("20x20", matmul, make_matrix(20), make_matrix(20))
  time("128x128", matmul, make_matrix(128), make_matrix(128))
  time("256x256", matmul, make_matrix(256), make_matrix(256))

  print("--- JIT (NAIVE) MATMULS ---")
  m5 = jit(matmul_n_slow(5))
  m10 = jit(matmul_n_slow(10))
  time("5x5 (1)", m5, make_matrix(5) , make_matrix(5))
  time("5x5 (2)", m5, make_matrix(5) , make_matrix(5))
  time("10x10 (1)", m10, make_matrix(10), make_matrix(10)) # Takes a long time!
  time("10x10 (2)", m10, make_matrix(10), make_matrix(10)) # ...but OK to call later.

  print("--- JIT (FAST) MATMULS ---")
  m = jit(matmul_fast)
  time("5x5", m, jax_matrix(5) , jax_matrix(5)) # Pay once (not too bad)! Call many times! 
  time("20x20", m, jax_matrix(20), jax_matrix(20))
  time("128x128", m, jax_matrix(128), jax_matrix(128))
  time("256x256", m, jax_matrix(256), jax_matrix(256))
  time("1024x1024", m, jax_matrix(1024), jax_matrix(1024))
main()