# Standard imports
import numpy as np

from pathos.pools import ProcessPool
from scipy.integrate import solve_ivp
from typing import Any, Callable, Tuple

##############################
# Class: ODE system
##############################
class ODESystem:
  """ODE System."""
  def __init__(self, g: Callable, s0: Tuple, T: float) -> None:
    self.g = g
    self.s0 = s0
    self.T = T

  def gen_operator_data(self, space: Any, m: int, num: int, verbose: bool=True) -> Tuple[np.ndarray]:
    if verbose:
      print("Generating operator data...", flush=True)
    features = space.random(num)
    sensors = np.linspace(0, self.T, num=m)[:, None]
    sensor_values = space.eval_batch(features, sensors)
    y = self.T * np.random.rand(num)[:, None]
    s = self.eval_s_space(space, features, y)
    return sensor_values, y, s

  def eval_s_space(self, space: Any, features: np.ndarray, y: np.ndarray) -> np.ndarray:
    def f(feature, yi):
      return self.eval_s(lambda t: space.eval_one(feature, t), yi[0])

    p = ProcessPool(nodes=4)
    res = p.map(f, features, y)
    return np.array(list(res))

  def eval_s_fn(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    res = map(lambda yi: self.eval_s(u, yi[0]), y)
    return np.array(list(res))

  def eval_s(self, u: np.ndarray, tf: float) -> Any:
    def f(t, y):
      return self.g(y, u(t), t)

    sol = solve_ivp(f, [0, tf], self.s0, method="RK45")
    return sol.y[0, -1:]



##############################
# Function: Creates a gravity pendulum ODE system
##############################
def ode_system(T: float) -> ODESystem:
	"""ODE."""
	def g(s, u, x):
		# gravity pendulum
		k = 1
		return [s[1], - k * np.sin(s[0]) + u]
	s0 = [0, 0]
	return ODESystem(g, s0, T)


