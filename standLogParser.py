import re
import subprocess
import sys

'''
Simple script that parses log file from stand and creates
reformatted file for matlab with 4 arrays from 4 adc channels.
This script also copying all the reformatted data to your buffer
(if you run it on linux actually...)
'''

def ParseAndCopy(input_file_path, output_file_path = "standlog.txt") -> None:

    with open(input_file_path, "r") \
    as input_file, open(output_file_path, "w+") as new_file:
        all_lines = input_file.readlines()
        angles = all_lines[4: -1: 5]
        # adc0 = all_lines[5: -1: 5]
        # adc1 = all_lines[6: -1: 5]
        # adc2 = all_lines[7: -1: 5]
        # adc3 = all_lines[8: -1: 5]


        angles = [re.findall(r"[+-]?\d+(?:\.\d+)?", string) for string in angles]
        for channel in range(4):

            new_file.write(f"Data_{channel} = [")

            for line, adc  in zip(angles, all_lines[5+channel: -1: 5]):

                new_file.write(line[0] + " ")
                new_file.write(line[1] + " ")
                new_file.write(re.findall(r'\d+',adc)[1] + ";")
                new_file.write("\n")

            new_file.write("]\n\n") 


        subprocess.run(f"xclip -selection clipboard {output_file_path}".split(" "))
        new_file.close()
        input_file.close()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SyntaxError("not enough arguments passed to the script")
    elif len(sys.argv) == 2:
        ParseAndCopy(sys.argv[1])
    else:
        ParseAndCopy(sys.argv[1], sys.argv[2])







