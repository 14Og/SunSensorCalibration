import math
import sys

with  open("coordinateXLog.bin", "r+") as x_file, \
      open("coordinateYLog.bin", "r+") as y_file, \
      open("splitting_logs_output/AzEllog.txt", "w+") as new_file, \
      open("splitting_logs_output/az_inc_0_180.txt", "w") as file1, \
      open("splitting_logs_output/az_inc_180_360.txt", "w") as file2, \
      open("splitting_logs_output/el_inc_min_0.txt", "w") as file3, \
      open("splitting_logs_output/el_inc_0_max.txt", "w") as file4:


        x_lines = [line[:-2:] for line in x_file.readlines()]
        x_lines[0] = x_lines[0][10::]
        x_lines = [s.split(" ") for s in x_lines]
        x_lines = x_lines[:-1]
        if len(sys.argv) > 1 and sys.argv[1]  == "cartesian":
            x_lines = [[i[0], str(round((180/math.pi)*math.acos(math.cos(float(i[1])*180/math.pi)*math.sin(float(i[2])*180/math.pi)),3)), \
                    str(round((180/math.pi)*math.acos(math.sin(float(i[1])*180/math.pi)*math.sin(float(i[2])*180/math.pi)),3))] for i in x_lines]
    


        y_lines = [line[:-2:] for line in y_file.readlines()]
        y_lines[0] = y_lines[0][10::]
        y_lines = [s.split(" ") for s in y_lines]
        y_lines = y_lines[:-1]


        az_inc = [[x_lines[i][0], y_lines[i][0], x_lines[i][2]] for i in range(len(x_lines))]
        el_inc = [[x_lines[i][0], y_lines[i][0], x_lines[i][1]] for i in range(len(x_lines))]

        az_inc.sort(key = lambda x: float(x[:][2]))
        el_inc.sort(key = lambda x: float(x[:][2]))
        for lst_az in range(len(az_inc)):
              if float(az_inc[lst_az][2]) > 180:
                    break
              

        for lst_el in range(len(el_inc)):
              if float(el_inc[lst_el][2]) > 0:
                    break
              
              
        

        az_inc_0_180 = az_inc[::]
        az_inc_180_360 = az_inc[lst_az:]
        el_inc_min_0 = el_inc[:lst_el]
        el_inc_0_max = el_inc[::]

        if __name__ == "__main__":
             
            for i in range(2):
                new_file.write("az_inc_0_180 = [" if  i == 0 else "az_inc_180_360 = [")
                for vector in az_inc_0_180 if i == 0 else az_inc_180_360:
                    new_file.write(vector[0] + " ")
                    new_file.write(vector[1] + " ")
                    new_file.write(vector[2] + ";\n")
                if i == 0:
                     for vector in az_inc_0_180:
                        for val in vector:
                            file1.write(val + " ")
                        file1.write("\n")
                else:
                    for vector in az_inc_180_360:
                        for val in vector:
                            file2.write(val + " ")
                        file2.write("\n")
                     
                        
                new_file.write("]\n\n")
            
            for i in range(2):
                new_file.write("el_inc_min_0 = [" if  i == 0 else "el_inc_0_max = [" )
                for vector in el_inc_min_0 if i == 0 else el_inc_0_max:
                    new_file.write(vector[0] + " ")
                    new_file.write(vector[1] + " ")
                    new_file.write(vector[2] + ";\n")
                if i == 0:
                    for vector in el_inc_min_0:
                        for val in vector:
                            file3.write(val + " ")
                        file3.write("\n")
                    else:
                        for vector in el_inc_0_max:
                            for val in vector:
                                file4.write(val + " ")
                            file4.write("\n")


                    
                    

                new_file.write("]\n\n")


        x_file.close()
        y_file.close()
        new_file.close()
