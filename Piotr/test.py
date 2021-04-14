import numpy as np
from math import sqrt

a = [2,3,4,4,5,6]
b = [2,2,2,5,12,12,12]
c = [2,2,2]

def evaluation(ratio):
    # sqrt(sum(r^2))
    se = sqrt(sum( [(it - 1)**2 for it in ratio]))
    return se


print(evaluation(a))
print(evaluation(b))
print(evaluation(c))