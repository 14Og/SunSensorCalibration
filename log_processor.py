import numpy as np
import pandas as pd
import os
import yaml

from typing import Tuple


class LogProcessor:
    def __init__(self, path, filename) -> None:
        self.file_path = self.get_file_path(path = path)
        self.df, self.arr = self.load(self.file_path, filename)
        self.measurements_count = np.shape(self.arr)[0]
        # self.sensor_data, self.stand_data = self.preprocess(self.arr) # necessary action bc YAML reads dummy
    
    def get_file_path(self, path = None):
        current_directory = os.getcwd()
        # parent_directory = os.path.dirname(current_directory)
        file_path = os.path.join(current_directory, path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Directory {file_path} doesn't exist.") 
            
        return file_path


    def load(self, path: str, filename: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Brief:
            Read YAML file and represent it in the numpy array

        Args:
            path (str): place of the folder with logs 
            filename (str): name of YAML file to read

        Raises:
            Exception: incorrect arguments
            Exception: file not exist

        Returns:
            np.array: fucking dict of the YAML reading
        """
        if not filename or not path:
            raise Exception(f"Bad arguments in function load()")
        
        yaml_file_path = os.path.join(path, filename)

        if not os.path.isfile(yaml_file_path):
            raise FileNotFoundError(f"File {yaml_file_path} not found.")

        with open(yaml_file_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

        calibration_df = pd.DataFrame(data=data)
        calibration_array = calibration_df.to_numpy()

        return calibration_df, calibration_array

    
    def save(self, filename: str = "calibrated", file_path: str = "calibrated", 
             data: np.ndarray = None, field: str = None) -> bool:
        """
        Brief:
            Save np.array into the named file located in directory (make if not exist).

        Args:
            filename (str, optional): name of file. Defaults to "calibrated".
            file_path (str, optional): directory which contains file. Defaults to "calibrated".
            data (np.ndarray, optional): array with data to save. Defaults to None.
            field (str, optional): name of the field in file. Defaults to None.

        Raises:
            Exception: if arguments not passed

        Returns:
            bool: if file is saved
        """
        if not filename or not file_path:
            raise Exception("Bad arguments in function check_path()")

        os.makedirs(file_path, exist_ok = True)

        file = os.path.join(file_path, filename)
        
        # preprocess before writing to yaml
        data_list = data.tolist()
        data_dict = {
            field: {
                id + 1: {'x': item[0], 'y': item[1]} for id, item in enumerate(data_list)
            }
        }
        yaml_data = yaml.dump(data_dict, default_flow_style = False)
        
        output_file = f'{file}.yaml'
        with open(output_file, 'w') as new_yaml:
            new_yaml.write(yaml_data)
        
        return True


    def preprocess(self, arr : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Brief:
            Preprocess log data from dict type to the two arrays - with sensor coordinates 
            and with stand angles in the form of Decart coordinates 

        Args:
            arr (np.array): dict from YAML reading method

        Raises:
            Exception: keys in sensor data don't match
            Exception: keys in stand data don't match

        Returns:
            Tuple[np.array, np.array]: sensor and stand coordinates
        """
        sensor_arr = np.empty((self.sensor_measurement_count, 2))
        sensor_key = list(arr[0][0].keys())[0]

        stand_arr = np.empty((self.stand_measurement_count, 2))
        stand_key = list(arr[1][0].keys())[0]

        for i in range(0, self.sensor_measurement_count):
            if list(arr[0][i].keys())[0] == sensor_key:
                sensor_arr[i, 0] = arr[0][i][sensor_key]["x"]
                sensor_arr[i, 1] = arr[0][i][sensor_key]["y"]
            else:
                raise Exception("Sensor datalog is not correct!")

        for i in range(0, self.stand_measurement_count):
            if list(arr[1][i].keys())[0] == stand_key:
                stand_arr[i, 0] = arr[1][i][stand_key]["x"]
                stand_arr[i, 1] = arr[1][i][stand_key]["y"]
            else:
                raise Exception("Stand datalog is not correct!")
        return sensor_arr, stand_arr


    def change_file_path(self, path : str):
        self.file_path = self.get_file_path(path=path)
    

    def change_yaml(self, filename : str):
        self.df, self.arr = self.read_yaml(self.file_path, filename)
        self.sensor_measurement_count = np.size(self.arr[0])
        self.stand_measurement_count = np.size(self.arr[1])
        
        
if __name__ == "__main__":
    data = LogProcessor(path="calibration_data",filename="merged_data.yaml")
    