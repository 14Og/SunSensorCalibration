import matplotlib.pyplot as plt
from scipy.interpolate import *
import numpy as np







def test_regular_grid_interpolation():
    rng = np.random.default_rng()
    x = np.linspace(1,10, 10)
    y = np.linspace(3,12, 10)
    z = np.hypot(x, y)
    print(x,y,z,sep="\n\n")
    X = np.linspace(min(x), max(x))
    Y = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)
    # plt.pcolormesh(X, Y, Z, shading='auto')
    # plt.plot(x, y, "ok", label="input point")
    # plt.legend()
    # plt.colorbar()
    # plt.axis("equal")
    # plt.show()


    
    
    
    
    
if __name__ == "__main__":
    test_regular_grid_interpolation()