import re

with  open("/home/nikolai/projects/pypon/standLogParser/coordinateXLog.bin", "r+") as x_file, \
      open("/home/nikolai/projects/pypon/standLogParser/coordinateYLog.bin", "r+") as y_file, \
          open("output/AzEllog.txt", "w+") as new_file:
    x_lines = [line[:-2:] for line in x_file.readlines()]
    x_lines[0] = x_lines[0][11::]
    x_lines = [s.split(" ") for s in x_lines]
    x_lines = x_lines[:-1]
    print(x_lines)

    y_lines = [line[:-2:] for line in y_file.readlines()]
    y_lines[0] = y_lines[0][11::]
    y_lines = [s.split(" ") for s in y_lines]
    y_lines = y_lines[:-1]


    x_list = [[str(round(float(x_lines[i][0]),5)), str(round(float(y_lines[i][0]),5)) \
               , y_lines[i][1]] for i in range(len(y_lines))]
    x_list.sort(key = lambda x: float(x[:][0]))


    y_list = [[str(round(float(x_lines[i][0]),5)), str(round(float(y_lines[i][0]),5)) \
               , y_lines[i][2]] for i in range(len(y_lines))]
    y_list.sort(key = lambda x: float(x[:][1]))


    
    new_file.write("x_inc = [")
    for vector in x_list:
        new_file.write(vector[0] + " ")
        new_file.write(vector[1] + " ")
        new_file.write(vector[2] + ";\n")

    new_file.write("]\n\n")

    new_file.write("y_inc = [")
    for vector in y_list:
        new_file.write(vector[0] + " ")
        new_file.write(vector[1] + " ")
        new_file.write(vector[2] + ";\n")

    new_file.write("]\n")

    
    x_file.close()
    y_file.close()
    new_file.close()


   
    
    



