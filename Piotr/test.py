import numpy as np

a = [2,2,2,2,2,3,2,5,4,2]
b = [2,2,2,5,12,12,12]
c = [2,2,2]

def evaluation(ratio):
    return max(min(ratio),np.var(ratio))

print(evaluation(a))