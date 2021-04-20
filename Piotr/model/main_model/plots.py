from matplotlib import pyplot as plt
import numpy as np

def barplot(y, x_names, bar_names, title='title', y_name='y', width=0.1, legend='upper center'):
    x = np.arange(len(x_names))
    fig, ax = plt.subplots()
    q = (len(y)-1)/(-2)
    for i in range(len(y)):
        ax.bar(x+q*width, y[i], width, label=bar_names[i])
        q+=1
    ax.set_ylabel(y_name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_names)

    if not legend == False:
        ax.legend(loc=legend)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    y = [[1,2,3,4],[1.1,2.1,3.1,4.1],[1,1,1,1]]
    x_names = ['a','b','c','d']
    bar_names = ['b1','b1.1','c1']
    barplot(y,x_names,bar_names)