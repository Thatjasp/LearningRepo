import math


def getShannonInformationContent(prob: float):
    return -math.log2(prob)


def getShannonEntropy(probs: list[float]):
    sum = 0
    for x in probs:
        sum += x * -getShannonInformationContent(x)
    return -sum


probabilityList = [(1 / 12), (1 / 12), (10 / 12)]
print(getShannonEntropy(probabilityList))
