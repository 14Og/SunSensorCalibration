import sys
import re
from math import pi, atan, degrees, acos, asin, cos, sin


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
            elevation_lines = [[float(x_lines[i][0]), float(y_lines[i][0]), int(x_lines[i][1])] for i in range(len(x_lines))]
            elevation_lines.sort(key = lambda x: float(x[:][2]))
            # print(azimuth_lines)
            self.azimuth_lines = azimuth_lines
            self.elevation_lines = elevation_lines
            x_log.close()
            y_log.close()


    def cartesian(self, lines):
        lines = [[i[0], str(round((180/pi)*acos(cos(float(i[1])*180/pi)*sin(float(i[2])*180/pi)),3)), \
                    str(round((180/pi)*acos(sin(float(i[1])*180/pi)*sin(float(i[2])*180/pi)),3))] for i in lines]
        return lines
    

    def create_segmented_azimuth_data(self) -> dict:
        segmented_azimuth_data  = dict()
        az_keys = ["0-20", "30-50", "60-80", "90-110",
                  "120-140", "150-170", "180-200", "210-230",
                  "240-260", "270-290", "300-320", "330-350"]
        line = 0
        segmented_azimuth_data = {key:[] for key in az_keys}

        for angle in range(1,13):
                while self.azimuth_lines[line][2] < angle * 30:
                    segmented_azimuth_data[az_keys[angle-1]].append(self.azimuth_lines[line])
                    line += 1
                    if line == len(self.azimuth_lines):
                         break
                    
        return segmented_azimuth_data


    def calculate_azimuth(self, dataframe: list) -> list:
        x = dataframe[0]
        y = dataframe[1]

        azimuth =  -atan(y/x) if x > 0 and y < 0 else \
              -atan(y/x) + pi if x < 0 and y < 0 else \
                pi - atan(y/x) if x < 0 and y > 0 else \
                2 * pi - atan(y/x) if x > 0 and y > 0 else None
        azimuth = degrees(azimuth) 

        dataframe.append(azimuth)
        dataframe.append(abs(azimuth-dataframe[2]))

        return dataframe
    
    def save_matlab_data(self, angles = "spherical", path_to_save = "matlab/data"):

        with open(path_to_save + "/azimuth.txt", "w") as az_file, open(path_to_save + "/elevation.txt", "w") as el_file:
            for az_line in self.azimuth_lines if angles == "spherical" else self.cartesian(self.azimuth_lines):
                for val in az_line:
                    az_file.write(str(val) + " ")
                az_file.write("\n")
            for el_line in self.elevation_lines if angles == "spherical" else self.cartesian(self.elevation_lines):
                for val in el_line:
                    el_file.write(str(val) + " ")
                el_file.write("\n")


            az_file.close()
            el_file.close()
            

                      




test = Datamodel()
segments = test.create_segmented_azimuth_data()
test.save_matlab_data()
# for frame in segments:
#      for data in segments[frame]:
#         modified_frame = test.calculate_azimuth(data)
#         print(f"{modified_frame[-1]} {modified_frame[-2]};")



        