import matplotlib.pyplot as plt

import numpy as np
import math
import math


ax = plt.gca()
ax.set_aspect('equal', adjustable = 'box')


class Calibration:
    def __init__(self, azimuth_scope:list = [-45, 45], elevation_scope:list = [0, 360]):
        az_num = (azimuth_scope[1] - azimuth_scope[0]) // 5 + 1
        el_num = (elevation_scope[1] - elevation_scope[0]) // 5
        
        self.az_range = np.linspace(azimuth_scope[0], azimuth_scope[1], az_num)
        self.el_range = np.linspace(elevation_scope[0], elevation_scope[1], el_num, endpoint=False)
        self.spherical_sample_space, self.az_grid, self.el_grid = \
        Calibration.get_sample_space(self.az_range, self.el_range)
                
        self.x_range, _ = Calibration.get_x_range(self.spherical_sample_space)
        self.y_range, _ = Calibration.get_y_range(self.spherical_sample_space)
        self.cartesian_sample_space, self.x_grid, self.y_grid = \
        Calibration.get_sample_space(self.x_range, self.y_range)

        self.create_stand_calibration_data(self.cartesian_sample_space)
    
    def create_stand_calibration_data(self, cartesian_sample_space:list[tuple[float,float]], output:str = "calibration.yaml") -> list[tuple[float, float]]:
        radius = math.cos(math.pi/4)
        new_space = []
        for pair in cartesian_sample_space:
            x = pair[0]
            y = pair[1]
            if math.sqrt(x**2 + y**2) <= radius:
                z = math.sqrt(1 - x**2 - y**2)
                azimuth =  math.pi / 2 - math.asin(z)
                elevation = math.atan2(y, x)
                elevation = 2 * math.pi + elevation if elevation < 0 else elevation
                new_space.append(tuple(map(math.degrees,(azimuth, elevation))))
                
        new_space.sort(key=lambda h: (h[0], h[1]))
        
        
        
            
        
        
    
    @staticmethod 
    def get_sample_space(range1:np.ndarray, range2:np.ndarray) -> tuple: # sample_space, grid1, grid2
        grid1, grid2 = np.meshgrid(range1, range2)
        spherical_sample_space = list(map(tuple, np.column_stack((grid1.ravel(), grid2.ravel()))))
        return spherical_sample_space, grid1, grid2
    

    @staticmethod
    def get_x_range(spherical_space:list[tuple[float,float]]) -> np.ndarray:
        x_all = set([math.sin(math.radians(sample[0]))*math.cos(math.radians(sample[1])) for sample in spherical_space])
        x_max = max(x_all)
        x_min = min(x_all)
        x_range = np.linspace(x_min, x_max, 20)
        return x_range, x_all
    
    
    @staticmethod
    def get_y_range(spherical_space:list[tuple[float,float]]) -> np.ndarray:
        y_all = set([math.sin(math.radians(sample[0]))*math.sin(math.radians(sample[1])) for sample in spherical_space])
        y_max = max(y_all)
        y_min = min(y_all)
        y_range = np.linspace(y_min, y_max, 20)
        return y_range, y_all
        

    def plot(self, xv, yv):
            plt.plot(xv, yv, marker="o", color="k", markersize=3, linewidth=0)
            plt.show()
    
    
        



if __name__ == "__main__":
    c = Calibration()
