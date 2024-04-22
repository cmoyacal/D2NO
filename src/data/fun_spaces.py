# Standard imports
import numpy as np

from pathos.pools import ProcessPool

# Deep learning imports
from scipy import interpolate
from sklearn import gaussian_process as gp

##############################
# Class: Function space base class
##############################
class FunctionSpace:
  """Function space base class."""
  def random(self, size: int) -> np.ndarray:
    pass

  def eval_one(self, feature: float, x: float) -> float:
    pass

  def eval_batch(self, features: np.ndarray, xs: np.ndarray) -> np.ndarray:
    pass


##############################
# Class: Gaussian Random Field (GRF) function space
##############################
class GRF(FunctionSpace):
  """Gaussian Random Field (GRF) function space in 1D."""
  def __init__(
    self, 
    T: float=1.0, 
    kernel: str="RBF", 
    length_scale: float=1.0,
    N: int=1000, 
    interp: str="cubic"
    ) -> None:
    
    self.N = N
    self.interp = interp
    self.x = np.linspace(0, T, num=N)[:, None]

    if kernel == "RBF":
      K = gp.kernels.RBF(length_scale=length_scale)
    elif kernel == "AE":
      K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
    elif kernel == "ExpSineSquared":
      K = gp.kernels.ExpSineSquared(length_scale=length_scale, periodicity=T)
    
    self.K = K(self.x)
    self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

  def random(self, size: int) -> np.ndarray:
    u = np.random.randn(self.N, size)
    return np.dot(self.L, u).T

  def eval_one(self, feature: float, x: float) -> float:
    if self.interp == "linear":
      return np.interp(x, np.ravel(self.x), feature)
    
    f = interpolate.interp1d(
            np.ravel(self.x), feature, kind=self.interp, copy=False, assume_sorted=True
        )
    return f(x)

  def eval_batch(self, features: np.ndarray, xs: np.ndarray) -> np.ndarray:
    if self.interp == "linear":
      return np.vstack([np.interp(xs, np.ravel(self.x), y).T for y in features])

    p = ProcessPool(nodes=4)
    res = p.map(
      lambda y: interpolate.interp1d(
        np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )(xs).T,
        features,
      )
    return np.vstack(list(res))
