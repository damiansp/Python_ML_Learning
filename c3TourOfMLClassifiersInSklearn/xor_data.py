#from sklearn import datasets
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import StandardScaler
#from matplotlib.colors import ListedColormap
#from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np 
#import sys, os
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                             '../'))
#from plot_decision_regions import plot_decision_regions



np.random.seed(30)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c = 'b',
            marker = 'x',
            label = '1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c = 'r',
            marker = 's',
            label = '-1')
plt.ylim(-3.)
plt.legend()
plt.show()
