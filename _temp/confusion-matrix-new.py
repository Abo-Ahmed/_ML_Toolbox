# trying

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import random

values = [ 0 , 1 , 2 ]

actual = []
predicted = []
for i in range(60000):
  if i < 20000:
    actual.append(0)
    predicted.append(0)
  elif i < 40000:
    actual.append(1)
    predicted.append(1)
  else:
    actual.append(2)
    predicted.append(2)

  if i % 8 == 0 :
    predicted[i] = random.choice(values)

data = {'y_Actual':    actual,
        'y_Predicted': predicted
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()


from sklearn.metrics import accuracy_score
accuracy_score(actual, predicted)