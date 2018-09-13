import numpy as np

with open("/home/dellxenial/Programming/Workspaces/Python3 Workspace/BBFRVR/polynomialFits.txt", "r") as inFile:
    line = inFile.readline()

    strList = []
    averageTime = 0
    averageHeight = 0
    while line:
        line = line[:-2]
        print(line)
        strList.append([int(x.strip()) for x in line.split(",")])
        line = inFile.readline()

    npList = np.asarray(strList)
    # Compute average peak height and time ball flies
    for p in npList:
        averageTime += (-p[1] / (2 * p[0]))
        currentTime = (-p[1] / (2 * p[0]))
        averageHeight += ((p[0] * (currentTime ** 2)) + (p[1] * currentTime) + p[2])
        print ((p[0] * (currentTime ** 2)) + (p[1] * currentTime) + p[2])

    sumAndAveraged = np.true_divide(np.sum(npList, axis=0), 11)
    averageTime /= 11
    averageHeight /= 11

    # 0th index needs to be multiplied by 2 to get accel
    print("Polynomial: {}".format(sumAndAveraged))
    print("Time to Peak: {}".format(averageTime))
    print("Peak Height: {}".format(averageHeight))
