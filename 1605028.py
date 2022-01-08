import numpy as np
import pandas as pd
from scipy.stats import norm
import copy


def initialEstimator(mat):
    mat1 = copy.deepcopy(mat)
    for i in range(20):
        mat1 = mat1 @ mat1
    # print(np.isclose(mat1, mat1[0]).all())
    # print(mat1)
    return mat1[0]


def forwardCalc(emission, transition, initialMat):
    forward = np.zeros([dataSize, n])
    for i in range(n):
        forward[0, i] = initialMat[i]*emission[0, i]

    forward[0] = forward[0]/np.sum(forward[0])

    for i in range(1, dataSize):
        for j in range(n):
            forward[i, j] = (forward[i-1] @ transition[:, j]) * emission[i, j]
        forward[i] = forward[i]/np.sum(forward[i])
    # np.savetxt("forward.txt", forward, delimiter=" ")

    return forward


def backwardCalc(emission, transition):
    backward = np.zeros([dataSize, n])
    backward[dataSize-1] = np.ones(n)

    for i in range(dataSize-2, -1, -1):
        for j in range(n):
            temp = copy.deepcopy(backward[i+1])
            for k in range(n):
                temp[k] = temp[k]*emission[i+1, k]
            # backward[i,j]= (backward[i+1] * emission[i+1,:])@transition[j,:]
            backward[i, j] = temp@transition[j, :]
        backward[i] = backward[i]/np.sum(backward[i])
    np.savetxt("backwardTemp.txt", backward, delimiter=" ")
    return backward


def viterbiAlgorithm(emission, transition, learningParam):
    dpTable = np.zeros([dataSize, n])  # dpTable size 2x1000
    backTrack = np.zeros([dataSize, n])
    for i in range(n):
        dpTable[0, i] = np.log(initial[i]) + np.log(emission[0, i])
    for i in range(1, dataSize):
        for s in range(n):
            max = -np.Infinity
            prev = -1
            for k in range(n):
                temp = dpTable[i-1, k] + \
                    np.log(emission[i, s]) + np.log(transition[k, s])
                if (temp > max):
                    max = temp
                    prev = k
            dpTable[i][s] = max
            backTrack[i][s] = int(prev)

    ans = []
    ansIndex = int(np.argmax(np.array(dpTable[dataSize-1])))
    ans.append(ansIndex)
    for i in range(dataSize-1, 0, -1):
        # print(i ,ansIndex)
        ansIndex = backTrack[i, int(ansIndex)]
        ans.append(ansIndex)
    # print("ans size is ", len(ans))
    ans = np.array(ans)
    ans = np.flip(ans)
    # print("after flip" ,ans)
    ans = ans.tolist()

    # print(ans)
    for i in range(len(ans)):
        if ans[i] == 0:
            ans[i] = "El Nino"
        else:
            ans[i] = "La Nina"
    np.savetxt("dpTable.txt", dpTable, delimiter=' ')
    # np.savetxt("vitebriOutput.txt",ans,delimiter="\n")
    if(learningParam == 0):
        with open('states_Viterbi_wo_learning.txt', 'w') as f:
            for item in ans:
                f.write("\"%s\"\n" % item)
    else:
        with open('states_Viterbi_after_learning.txt', 'w') as f:
            for item in ans:
                f.write("\"%s\"\n" % item)


def BaumWelchAlgorithm(emission, transition, initialMat):
    for r in range(6):

        forward = forwardCalc(emission, transition, initialMat)
        backward = backwardCalc(emission, transition)

        piStar = np.zeros([dataSize, n])
        for i in range(dataSize):
            for j in range(n):
                piStar[i, j] = forward[i, j] * backward[i, j]
            piStar[i] = piStar[i]/np.sum(piStar[i])
        piDoubleStar = np.zeros([dataSize-1, n, n])
        for i in range(dataSize-1):
            for k in range(n):
                for l in range(n):
                    piDoubleStar[i, k, l] = forward[i, k] * backward[i+1,
                                                                     l] * emission[i+1, l] * transition[k, l]
            piDoubleStar[i] = piDoubleStar[i]/np.sum(piDoubleStar[i])

        meanList = np.zeros(n)
        SDlist = np.zeros(n)
        for i in range(n):        # initialMat = initialEstimator(transition)

            mul = piStar[:, i] * data
            sum = np.sum(piStar[:, i])
            meanList[i] = np.sum(mul)/sum
        # print("Mean list is", meanList)

        for i in range(n):
            mul = piStar[:, i] * (data-meanList[i])**2
            sum = np.sum(piStar[:, i])
            SDlist[i] = np.sqrt(np.sum(mul)/sum)
        # print("SD list is ", SDlist)

        ans = np.zeros([n, n])
        for i in range(dataSize-1):
            for k in range(n):
                for l in range(n):
                    ans[k, l] = ans[k, l]+piDoubleStar[i, k, l]

        for i in range(n):
            ans[i] = ans[i]/np.sum(ans[i])
        # print("Transition matrix is \n", ans)

        transition = ans
        emission = np.empty((n, dataSize))
        for i in range(n):
            emission[i] = norm(meanList[i], SDlist[i]).pdf(data)
        emission = np.asmatrix(emission).T

    viterbiAlgorithm(emission, transition, 1)

    # Output file for parameter
    file = open("parameters_learned.txt", "w")
    for i in range(n):
        for j in range(n):
            t = '{0:.4f}'.format(transition[i, j])
            file.write(f"{t}\t")
        file.write("\n")
    for i in range(n):
        t = '{0:.4f}'.format(meanList[i])
        file.write(f"{t}\t")
    file.write("\n")
    for i in range(n):
        t = '{0:.4f}'.format(SDlist[i]**2)
        file.write(f"{t}\t")

    file.write("\n")
    for i in range(n):
        t = '{0:.4f}'.format(initialMat[i])
        file.write(f"{t}\t")


##################################### Starting point ##################################
# reading parameterfile
f = open("parameters.txt", "r")
n = int(f.readline())
transitionMatrix = np.empty([n, n])
for i in range(n):
    str = f.readline()
    l = [np.double(p) for p in str.split()]
    transitionMatrix[i] = l

Mean = f.readline()
Mean = [np.double(p) for p in Mean.split()]
Variance = f.readline()
Variance = [np.double(p) for p in Variance.split()]

# initial probability
initial = initialEstimator(transitionMatrix)


# Reading data file
dataFile = open("data.txt", "r")
data = np.double(dataFile.readlines())

dataSize = len(data)

# Emission Matrix build
emissionMatrix = np.empty((n, dataSize))
for i in range(n):
    emissionMatrix[i] = norm(Mean[i], np.sqrt(Variance[i])).pdf(data)
emissionMatrix = np.asmatrix(emissionMatrix).T


# running  Viterbi and  Baum Welch algorithm
viterbiAlgorithm(emissionMatrix, transitionMatrix, 0)
BaumWelchAlgorithm(emissionMatrix, transitionMatrix, initial)
