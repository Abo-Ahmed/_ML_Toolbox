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



# anther good method

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2])
labels = ["Cats", "Dogs", "Horses"]

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()



## the combination of the two methods
# trying

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import random

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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

  if i % 24 == 0 :
    predicted[i] = random.choice(values)

data = {'y_Actual':    actual,
        'y_Predicted': predicted
        }


from sklearn.metrics import accuracy_score
print(accuracy_score(actual, predicted))


cm = confusion_matrix(actual, predicted)

cm = cm / np.sum(cm)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()