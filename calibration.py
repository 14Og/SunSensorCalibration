import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
import math
from typing import List, Tuple
from itertools import cycle
from skimage.measure import CircleModel
from scipy.interpolate import griddata, interp2d

from log_processor import LogProcessor


#TODO: creation of calibration yaml file for stand

class Calibration:
    
    colors =  [(i/700, 1 - (i+300)/700, i/500) for i in range(400)]


    def __init__(self, azimuth_scope:list = [-45, 45], elevation_scope:list = [0, 360]):
        self.log_processor = LogProcessor("calibration_data", filename="merged_data.yaml")
        
        az_num:int = (azimuth_scope[1] - azimuth_scope[0]) // 5 + 1
        el_num:int = (elevation_scope[1] - elevation_scope[0]) // 5
        
        self.az_range:np.ndarray = np.linspace(azimuth_scope[0], azimuth_scope[1], az_num)
        self.el_range:np.ndarray = np.linspace(elevation_scope[0], elevation_scope[1], el_num, endpoint=False)
        self.spherical_sample_space, self.az_grid, self.el_grid = \
        Calibration.get_sample_space(self.az_range, self.el_range)
                
        self.x_range, _ = Calibration.get_x_range(self.spherical_sample_space)
        self.y_range, _ = Calibration.get_y_range(self.spherical_sample_space)
        self.cartesian_sample_space, self.x_grid, self.y_grid = \
        Calibration.get_sample_space(self.x_range, self.y_range)

        self.create_stand_calibration_data(self.cartesian_sample_space)
    
    def create_stand_calibration_data(self, cartesian_sample_space:list[tuple[float,float]], output:str = "calibration.yaml") \
        -> list[tuple[float, float]]:
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
        

    def plot_stand_data(self, data:pd.DataFrame = None) -> None:
        
        angles = data.loc[:, "elevation":"azimuth"].apply(np.radians)
        X = np.multiply(np.sin(angles.loc[:, "elevation"]), np.cos(angles.loc[:, "azimuth"]))
        Y = np.multiply(np.sin(angles.loc[:, "elevation"]), np.sin(angles.loc[:, "azimuth"]))
        
        new_frame = pd.DataFrame({"X":X, "Y":Y, "x_light": data.loc[:, "x"], "y_light": data.loc[:, "y"]})

        fig, (stand, light) = plt.subplots(1,2)
        stand.set_aspect('equal', adjustable = 'box')
        light.set_aspect('equal', adjustable = 'box')
        color_gen = Calibration.color()
        for i in range (data.shape[0]):
            c = next(color_gen)
            stand.scatter(new_frame.loc[i, "X"], new_frame.loc[i, "Y"], color = c)
            light.scatter(new_frame.loc[i, "x_light"], new_frame.loc[i, "y_light"], color = c)
        
        xc, yc, r = self.calculate_light_shape()
        light.add_patch(Circle((xc, yc), r, fill=False, color="red"))
        self.calibrate(new_frame, (xc,yc,r))
        plt.show()
    
      
      
    def calculate_light_shape(self, last_points_included=50) -> Tuple[float, float, float]:  # xc, yc, radius of estimated circle
        light_shape = CircleModel()
        success = light_shape.estimate(self.log_processor.df.loc[self.log_processor.measurements_count-last_points_included:, "x":"y"].to_numpy())
        if not success:
            raise Exception("Could not fit circle to light sensor data")
        
        return light_shape.params
        
          
    def calibrate(self, new_dataframe:pd.DataFrame, light_circle_shape:Tuple[float, float, float]) -> bool:
        xc, yc, r = light_circle_shape
        
        x_lin = np.linspace(xc-r, xc+r, 30)
        y_lin = np.linspace(yc-r, yc+r, 30)
        
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)
        calibration_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        calibration_grid = tuple(filter(lambda pair: (pair[0] - xc)**2 + (pair[1] - yc)**2 < r**2, calibration_grid))
        print(calibration_grid)
        
        
        x_points = new_dataframe.loc[:, "x_light"].to_numpy()
        y_points = new_dataframe.loc[:, "y_light"].to_numpy()
        x_values = new_dataframe.loc[: , "X"]
        y_values = new_dataframe.loc[: , "X"]
        X = interp2d(x_points, y_points, x_values)
        Y = interp2d(x_points, y_points, y_values)
        # print(X)


    @classmethod 
    def color(cls):
        for color in cls.colors:
            yield color
            
        
    
    
        



if __name__ == "__main__":
    c = Calibration()
    logs = LogProcessor("calibration_data", filename="merged_data.yaml")
    c.plot_stand_data(c.log_processor.df)
