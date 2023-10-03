import sys
import re
from math import pi, atan, degrees, acos, asin, cos, sin
import numpy as np

np.set_printoptions(formatter={'float_kind':'{:f}'.format}, precision=5)

class Datamodel:
    def __init__(self, path_to_logs = ["\\".join(__file__.split("\\")[:-1]) + "\\input_data\\coordinateXLog_29_09.bin", \
                                        "\\".join(__file__.split("\\")[:-1]) +   "\\input_data\\coordinateYLog_29_09.bin"]) -> None:
        self.x_log = path_to_logs[0]
        self.y_log = path_to_logs[1]
        with open(self.x_log, "r") as x_log, open(self.y_log, "r") as y_log:
            x_lines = x_log.readlines()
            y_lines = y_log.readlines()
 
            
            x_lines = [re.findall("[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d*\.\d+|\d+", line) for line in x_lines]
            y_lines = [re.findall("[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d*\.\d+|\d+", line) for line in y_lines]


            azimuth_lines = [[float(x_lines[i][0]), float(y_lines[i][0]), int(x_lines[i][2])] for i in range(len(x_lines))]
            azimuth_lines.sort(key= lambda x: float(x[:][2]))
            azimuth_lines = np.array(azimuth_lines, dtype=np.float64)

            elevation_lines = [[float(x_lines[i][0]), float(y_lines[i][0]), int(x_lines[i][1])] for i in range(len(x_lines))]
            elevation_lines.sort(key = lambda x: float(x[:][2]))
            elevation_lines = np.array(elevation_lines, dtype=np.float64)
            # print(azimuth_lines)
            self.azimuth_lines = azimuth_lines
            self.elevation_lines = elevation_lines
            x_log.close()
            y_log.close()



    def cartesian(self, lines):
        lines = np.array([[i[0], str(round((180/pi)*acos(cos(float(i[1])*180/pi)*sin(float(i[2])*180/pi)),3)), \
                    str(round((180/pi)*acos(sin(float(i[1])*180/pi)*sin(float(i[2])*180/pi)),3))] for i in lines])
        return lines
    


    def __radius_calculate(self, dataframe: list) -> float:
        x_coord = dataframe[0]
        y_coord = dataframe[1]
        r = (x_coord**2 + y_coord**2)**0.5
        return r

    def create_segmented_azimuth_data(self, tresh = 0) -> dict:
        segmented_azimuth_data  = dict()
        az_keys = ["0-20", "20-40", "40-60", "60-80",
                  "80-100", "100-120", "120-140", "140-160",
                  "160-180", "180-200", "200-320", "220-240",
                  "240-260", "26-280", "280-300", "300-320",
                  "320-340", "340-360"]
        line = 0
        segmented_azimuth_data = {key:np.empty((0,3)) for key in az_keys}

        for angle in range(1,19):
                while self.azimuth_lines[line][2] <= angle * 20:
                    radius = self.__radius_calculate(list(self.azimuth_lines[line]))
                    if radius > tresh:
                        segmented_azimuth_data[az_keys[angle-1]] = \
                        np.append(segmented_azimuth_data[az_keys[angle-1]], [self.azimuth_lines[line]], axis=0)
                        if self.azimuth_lines[line][2] == angle * 20 and angle != 12:
                            segmented_azimuth_data[az_keys[angle]] = \
                            np.append(segmented_azimuth_data[az_keys[angle]], [self.azimuth_lines[line]], axis=0)
                    line += 1
                    if line == len(self.azimuth_lines):
                         break
                lst = list(map(list,segmented_azimuth_data[az_keys[angle-1]]))
                lst.sort(key=lambda x: self.__radius_calculate(x[:]))
                segmented_azimuth_data[az_keys[angle-1]] = np.array(lst)
                
                    
        # print(segmented_azimuth_data)       
        return segmented_azimuth_data


    def calculate_azimuth(self, dataframe: list) -> list:
        x = dataframe[0]
        y = dataframe[1]
        th_az = dataframe[2]

        azimuth =  -atan(y/x) if x > 0 and y < 0 else \
              -atan(y/x) + pi if x < 0 and y < 0 else \
                pi - atan(y/x) if x < 0 and y > 0 else \
                2 * pi - atan(y/x) if x > 0 and y > 0 else None
        azimuth = degrees(azimuth) 

        dataframe = np.append(dataframe, azimuth)
        dataframe = np.append(dataframe, (abs(azimuth-th_az)))

        return dataframe
    
    def save_azimuth_error(self, data : dict,  tresh = 0, path_to_save = "matlab/data") -> None:
        with open(path_to_save + "/azimuth_error.txt", "w") as er_file:
            for key in data:
                for line in data[key]:
                    x = line[0]
                    y = line[1]
                    if (x**2 + y**2)**0.5 >= tresh:
                        er_file.write(str(self.calculate_azimuth(line)[-3]) + " ")
                        er_file.write(str(self.calculate_azimuth(line)[-1]) + "\n")
            er_file.close()

                    
                    
    
    def save_matlab_data(self, az_source, el_source, angles = "spherical", path_to_save = "matlab/data"):
        with open(path_to_save + "/azimuth.txt", "w") as az_file, open(path_to_save + "/elevation.txt", "w") as el_file:
            for az_line in az_source if angles == "spherical" else self.cartesian(self.azimuth_lines):
                for val in az_line:
                    az_file.write(str(val) + " ")
                az_file.write("\n")
            for el_line in el_source if angles == "spherical" else self.cartesian(self.elevation_lines):
                for val in el_line:
                    el_file.write(str(val) + " ")
                el_file.write("\n")


            az_file.close()
            el_file.close()
            

                        


if __name__ == "__main__":
    test = Datamodel()
    segments = test.create_segmented_azimuth_data(tresh=0.1)
    test.save_matlab_data(az_source=list(test.azimuth_lines), el_source=test.elevation_lines)
    print(test.azimuth_lines)
    print(list(map(list,segments.values())))
    test.save_azimuth_error(segments, tresh=0.05)



        