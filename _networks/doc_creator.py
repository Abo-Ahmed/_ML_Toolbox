import os

results = ""
lineCounter = 0
fileCounter = 0
functionCounter = 0
fileTree = ""
functionTree = ""
root = "D:\master\_master_network"
removedFolders = [".git" , "_networks" , "0000" , "_temp" , "test" , "train" , "random"]

for root, dirs, files in os.walk(root):

    for r in removedFolders:
        if(r in dirs):
            dirs.remove(r)

    for d in dirs:
        fileTree = fileTree + "\n>>> " + d 
        newRoot = root+"\\"+d   
        for newRoot, ds, fs in os.walk(newRoot):
            for f in fs:
                if f == "rating.py":
                    continue
                if f.endswith(".py"):
                    fileTree = fileTree + "\n\t" + f
                    functionTree = functionTree + "\n>>> " + f
                    fileCounter = fileCounter + 1
                    fileName = "D:\\master\\_master_network\\"+ d + "\\" + f
                    with open(fileName) as file:
                        for line in file:
                            if(len(line.strip()) != 0):
                                lineCounter = lineCounter + 1
                            if("def " in line and not("#" in line)):
                                functionCounter = functionCounter + 1
                                functionTree = functionTree + "\n\t" + line.split("def")[1].split("(")[0]

        functionTree = functionTree + "\n-----------------------------\n"
        fileTree = fileTree + "\n-----------------------------\n"

# adding main
fileCounter = fileCounter + 1 
lineCounter = lineCounter + 12

# showing results
results = results + "\nNumber of files: " + str(fileCounter)
results = results + "\nNumber of functions: " + str(functionCounter)
results = results + "\nNumber of line: " + str(lineCounter)
results = results + "\n==================================\nFile tree:\n" + fileTree
results = results + "\n==================================\nFunction tree:\n" + functionTree

print(results)

with open("D:\\master\\_master_network\\_networks\\documentation.txt", "w") as text_file:
    text_file.write(results)