# trying

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score


def fill_data( limit , vertical , horizontal):
  actual = []
  predicted = []
  for i in range(400):
    if ( i < 40):
      continue

    if( i > limit):
      actual.append(i / vertical)
      predicted.append(65 + limit * limit / horizontal ) 
      continue

    actual.append(i / vertical)
    predicted.append(65 + i * i / horizontal ) 

  return actual , predicted

actual_1 , predicted_1 = fill_data(180 , 1.5 , 1000)
# actual_1 = []
# predicted_1 = []
# for i in range(180):
#   if ( i < 40):
#     continue

#   actual_1.append(i / 1.5)
#   predicted_1.append(65 + i * i / 1000 ) 

actual_2 , predicted_2 = fill_data(175 , 2 , 900)
# actual_2 = []
# predicted_2 = []
# for i in range(175):
#   if ( i < 40):
#     continue

#   actual_2.append(i / 2)
#   predicted_2.append(65 + i * i / 900 ) 

actual_3 , predicted_3 = fill_data(155 , 2 , 700)
# actual_3 = []
# predicted_3 = []
# for i in range(155):
#   if ( i < 40):
#     continue

#   actual_3.append(i / 2)
#   predicted_3.append(65 + i * i / 700 ) 

plt.xlabel('training period (hours)')

plt.ylabel('model accuracy (%)')


plt.plot(actual_1, predicted_1 , actual_2, predicted_2 , actual_3, predicted_3)
plt.show()
