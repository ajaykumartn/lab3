#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\abhis\Documents\Machine learning\lungcancer.csv")
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=45)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
score


# In[35]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\abhis\Documents\Machine learning\lungcancer.csv")
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=45)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
score


# In[36]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\abhis\Documents\Machine learning\lungcancer.csv")
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=45)
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
score


# In[37]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\abhis\Documents\Machine learning\lungcancer.csv")
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=45)
model=KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
score


# In[38]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\abhis\Documents\Machine learning\lungcancer.csv")
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=45)
model=SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
score


# In[39]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\abhis\Documents\Machine learning\lungcancer.csv")
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=45)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
score


# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv(r"C:\Users\abhis\Documents\Machine learning\lungcancer.csv")
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=45)

accuracies = {}

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
accuracies["Random Forest"] = accuracy_score(y_test, y_pred_rf)

# SVM Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
accuracies["SVM"] = accuracy_score(y_test, y_pred_svm)

# KNN Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
accuracies["KNN"] = accuracy_score(y_test, y_pred_knn)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
accuracies["Decision Tree"] = accuracy_score(y_test, y_pred_dt)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
accuracies["Logistic Regression"] = accuracy_score(y_test, y_pred_lr)

# Sort models by accuracy in decreasing order
sorted_accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))

# Find the model with the highest accuracy
max_accuracy_model = next(iter(sorted_accuracies))

# Assign colors: highest accuracy → green, rest → different colors
colors = ["blue", "red", "purple", "orange", "brown"]
color_map = ["green" if model == max_accuracy_model else colors[i] for i, model in enumerate(sorted_accuracies.keys())]

# Plot Bar Chart
plt.figure(figsize=(12, 5))
bars = plt.bar(sorted_accuracies.keys(), sorted_accuracies.values(), color=color_map)

# Display accuracy values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel("Classification Models")
plt.ylabel("Accuracy Score")
plt.title("Comparison of Classification Model Accuracies (Sorted)")
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.xticks(rotation=15)  # Rotate labels for better readability
plt.show()

# Plot Line Chart
plt.figure(figsize=(12, 5))
plt.plot(list(sorted_accuracies.keys()), list(sorted_accuracies.values()), marker='o', linestyle='-', color='black', label="Accuracy")

# Highlight the highest accuracy model in green
for i, (model, acc) in enumerate(sorted_accuracies.items()):
    color = "green" if model == max_accuracy_model else "black"
    plt.scatter(i, acc, color=color, s=100, label=model if color == "green" else "")

# Display accuracy values on the line
for i, acc in enumerate(sorted_accuracies.values()):
    plt.text(i, acc, f"{acc:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel("Classification Models")
plt.ylabel("Accuracy Score")
plt.title("Line Plot of Classification Model Accuracies")
plt.xticks(rotation=15)  # Rotate labels for better readability
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




