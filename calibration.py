import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
import math
from typing import List, Tuple
from itertools import cycle
from skimage.measure import CircleModel
from scipy.interpolate import LinearNDInterpolator
from collections import namedtuple
from log_processor import LogProcessor
from mpl_toolkits.mplot3d import Axes3D

#TODO: creation of calibration yaml file for stand
#TODO: interpolation/calibration error check
#TODO: organize plots more understandable

class Preparator:
    """Class-preparator for stand input data preparation before calibration process."""
    
    def __init__(self, azimuth_scope:list=[-45, 45], elevation_scope:list=[0, 360], log_path:str="stand_data") -> None:
        az_num:int = (azimuth_scope[1] - azimuth_scope[0]) // 5 + 1
        el_num:int = (elevation_scope[1] - elevation_scope[0]) // 5
        
        self.az_range:np.ndarray = np.linspace(azimuth_scope[0], azimuth_scope[1], az_num)
        self.el_range:np.ndarray = np.linspace(elevation_scope[0], elevation_scope[1], el_num, endpoint=False)
        
        self.spherical_sample_space, self.az_grid, self.el_grid = \
        self._get_sample_space(self.az_range, self.el_range)
                
        self.x_range, _ = self._get_x_range(self.spherical_sample_space)
        self.y_range, _ = self._get_y_range(self.spherical_sample_space)
        self.cartesian_sample_space, self.x_grid, self.y_grid = \
        self._get_sample_space(self.x_range, self.y_range)

        self.create_stand_calibration_data(self.cartesian_sample_space, log_path)
        
        
    @staticmethod 
    def _get_sample_space(range1:np.ndarray, range2:np.ndarray) -> tuple: # sample_space, grid1, grid2
        grid1, grid2 = np.meshgrid(range1, range2)
        spherical_sample_space = list(map(tuple, np.column_stack((grid1.ravel(), grid2.ravel()))))
        return spherical_sample_space, grid1, grid2
    
    
    @staticmethod
    def _get_x_range(spherical_space:list[tuple[float,float]]) -> np.ndarray:
        x_all = set([math.sin(math.radians(sample[0]))*math.cos(math.radians(sample[1])) for sample in spherical_space])
        x_max = max(x_all)
        x_min = min(x_all)
        x_range = np.linspace(x_min, x_max, 20)
        return x_range, x_all
    
    
    @staticmethod
    def _get_y_range(spherical_space:list[tuple[float,float]]) -> np.ndarray:
        y_all = set([math.sin(math.radians(sample[0]))*math.sin(math.radians(sample[1])) for sample in spherical_space])
        y_max = max(y_all)
        y_min = min(y_all)
        y_range = np.linspace(y_min, y_max, 20)
        return y_range, y_all
    
    
    def create_stand_calibration_data(self, cartesian_sample_space:list[tuple[float,float]], output_path:str) \
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
        return new_space


class Postprocessor:
    """Class that operates with stand data to reformat it for following calibration."""
    
    def __init__(self,stand_log_path:str, stand_log_filename:str) -> None:
        self.log_processor = LogProcessor(stand_log_path, stand_log_filename)
        self.calibration_data = self.process_stand_data(self.log_processor.df)
        self.light_shape_params = self._calculate_light_shape(last_points_included=50)
        
    
    def process_stand_data(self, raw_data:pd.DataFrame) -> pd.DataFrame:
        angles = raw_data.loc[:, "elevation":"azimuth"].apply(np.radians)
        X = np.multiply(np.sin(angles.loc[:, "elevation"]), np.cos(angles.loc[:, "azimuth"]))
        Y = np.multiply(np.sin(angles.loc[:, "elevation"]), np.sin(angles.loc[:, "azimuth"]))
        
        new_frame = pd.DataFrame({"X":X, "Y":Y, "x_light": raw_data.loc[:, "x"], "y_light": raw_data.loc[:, "y"]})
        return new_frame
    
    
    def _calculate_light_shape(self, last_points_included=50) -> Tuple[float, float, float]:  # xc, yc, r
        light_shape = CircleModel()
        success = light_shape.estimate(self.log_processor.df.loc[self.log_processor.measurements_count-last_points_included:, "x":"y"].to_numpy())
        if not success:
            raise Exception("Could not fit circle to light sensor data")
        
        LightParams = namedtuple("LightParams", ['xc', 'yc', "r"])
        
        return LightParams(*light_shape.params)
        

class Calibration:
    """ Class that contains all used methods for calibration process."""
    
    colors =  lambda r: [(1 - i/(1.1*r), 0.5*i/r, 0.3*i/r) for i in range(r)]
    
    def __init__(self, preparator:Preparator, postprocessor:Postprocessor) -> None:
        self.preparator = preparator
        self.postprocessor = postprocessor
        self.calibrated_sun_vector:np.ndarray = None
        self.calibrated_sensor_data:np.ndarray = None
    
    
    def emit(self):
        self.calibrate(self.postprocessor.calibration_data, self.postprocessor.light_shape_params)
        self.plot_calibration_data()
        self.plot_calibrated_data()
        
        

    def plot_calibration_data(self) -> None:
        fig, subplots = plt.subplots(2,2)
        [subplot.set_aspect('equal', adjustable = 'box') for subplot in subplots.ravel()]
        stand, light, _, _ = subplots.ravel()
        stand.title.set_text("Stand coordinates data")
        light.title.set_text("Light coordinates data")
      
        color_gen = Calibration.color(400)
        for i in range (self.postprocessor.calibration_data.shape[0]):
            c = next(color_gen)
            stand.scatter(self.postprocessor.calibration_data.loc[i, "X"], self.postprocessor.calibration_data.loc[i, "Y"], color = c)
            light.scatter(self.postprocessor.calibration_data.loc[i, "x_light"], self.postprocessor.calibration_data.loc[i, "y_light"], color = c)
        
        xc, yc, r = self.postprocessor.light_shape_params
        light.add_patch(Circle((xc, yc), r, fill=False, color="red"))

        X = fig.add_subplot(223, projection="3d")
        X.title.set_text("X coordinate of sunvector")
        Y = fig.add_subplot(224, projection="3d")
        Y.title.set_text("Y coordinate of sunvector")
        X.set_xlabel("x_light")
        Y.set_xlabel("x_light")
        X.set_ylabel("y_light")
        Y.set_ylabel("y_light")
          
        X.scatter(self.postprocessor.calibration_data.loc[:, "x_light"], self.postprocessor.calibration_data.loc[:, "y_light"], self.postprocessor.calibration_data.loc[:, "X"], c="green")
        Y.scatter(self.postprocessor.calibration_data.loc[:, "x_light"], self.postprocessor.calibration_data.loc[:, "y_light"], self.postprocessor.calibration_data.loc[:, "Y"], c="black")
        
        plt.show()
    
    
    def plot_calibrated_data(self) -> None:
        fig, subplots = plt.subplots(2,2)
        [subplot.set_aspect('equal', adjustable = 'box') for subplot in subplots.ravel()]
        sunvector, light, _, _ = subplots.ravel()
        sunvector.set_aspect('equal', adjustable = 'box')
        sunvector.title.set_text("Calibrated sun vector")
        light.set_aspect('equal', adjustable = 'box')
        light.title.set_text("Calibrated sensor data")
        color_gen = Calibration.color(700)
        for i in range (self.calibrated_sun_vector.shape[0]):
            c = next(color_gen)
            sunvector.scatter(self.calibrated_sun_vector[i, 0], self.calibrated_sun_vector[i, 1], color=c)
            light.scatter(self.calibrated_sensor_data[i, 0], self.calibrated_sensor_data[i, 1], color=c)
        
        X = fig.add_subplot(223, projection="3d")
        X.title.set_text("X coordinate of sunvector")
        Y = fig.add_subplot(224, projection="3d")
        Y.title.set_text("Y coordinate of sunvector")
        X.set_xlabel("x_light")
        Y.set_xlabel("x_light")
        X.set_ylabel("y_light")
        Y.set_ylabel("y_light")

        X.scatter(self.calibrated_sensor_data[:, 0], self.calibrated_sensor_data[:, 1], self.calibrated_sun_vector[:, 0], c="green")    
        Y.scatter(self.calibrated_sensor_data[:, 0], self.calibrated_sensor_data[:, 1], self.calibrated_sun_vector[:, 1], c="black")    
        
        plt.show()


    def calibrate(self, new_dataframe:pd.DataFrame, light_circle_shape:Tuple[float, float, float]) -> bool:
        xc, yc, r = light_circle_shape
        
        x_lin = np.linspace(xc-r, xc+r, 30)
        y_lin = np.linspace(yc-r, yc+r, 30)
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)
        
        calibration_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        calibration_grid = np.array(list(filter(lambda pair: (pair[0] - xc)**2 + (pair[1] - yc)**2 < r**2, calibration_grid)))

        self.calibrated_sensor_data = calibration_grid

        x_l_points = new_dataframe.loc[:, "x_light"].to_numpy()
        y_l_points = new_dataframe.loc[:, "y_light"].to_numpy()
        x_values = new_dataframe.loc[: , "X"]
        y_values = new_dataframe.loc[: , "Y"]
        
        X_interpolator = LinearNDInterpolator(list(zip(x_l_points, y_l_points)), x_values)
        Y_interpolator = LinearNDInterpolator(list(zip(x_l_points, y_l_points)), y_values)
        
        self.calibrated_sun_vector = np.stack((X_interpolator(calibration_grid[:,0], \
        calibration_grid[:,1]), Y_interpolator(calibration_grid[:,0], calibration_grid[:,1])), axis=1)
  

    @classmethod 
    def color(cls, i):
        colors = cls.colors(i)
        for color in colors:
            yield color
            

if __name__ == "__main__":
    prep = Preparator(log_path="stand_data")
    post = Postprocessor("calibration_data", "merged_data.yaml")
    
    c  = Calibration(preparator=prep, postprocessor=post)
    c.emit()
