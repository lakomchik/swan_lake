import os

# specify path to the folder containing your text files
folder_path = "test/labels"

# loop through all files in the folder
for filename in os.listdir(folder_path):
    print(filename)
    if filename.endswith(".txt"):
        # open the file
        with open(os.path.join(folder_path, filename), "r") as f:
            # read the contents of the file
            lines = f.readlines()
            
        # create a new list to store the modified lines
        new_lines = []
        
        # loop through each line in the file
        for line in lines:
            # split the line into individual floats
            team, x, y, w, h = map(float, line.split(" "))
            
            # calculate the new values
            new_x = x + w/2
            new_y = y + h/2
            
            # create the new line and add it to the list
            new_line = "{} {} {} {} {}".format(int(team), new_x, new_y, w, h)
            new_lines.append(new_line)
        
        # overwrite the old file with the modified lines
        with open(os.path.join(folder_path, filename), "w") as f:
            f.writelines("\n".join(new_lines))