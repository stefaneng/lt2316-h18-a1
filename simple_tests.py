import pandas as pd
import numpy as np
from io import StringIO
from myid3 import DecisionTree

df = pd.read_csv(StringIO("""cheese sauce spicy vegetables like
mozza hllnds yes no no
gouda tomato no no yes
mozza tomato yes no yes
jarls bbq no no no
mozza bbq yes yes no
gouda tomato yes yes yes
jarls hllnds yes yes yes
mozza tomato no yes yes
mozza bbq yes no maybe"""), sep=" ")

X = df.drop('like', axis=1)
y = df['like']
cols = X.columns.values
dt = DecisionTree()
dt.train(X, y, cols)
print(dt.model)

pred = pd.read_csv(StringIO("""cheese sauce spicy vegetables
mozza hllnds yes no
jarls hllnds yes no
mozzla tomato no yes
jarls tomato no no
jarls bbq no maybe"""), sep=" ")

#print(pred)
print(pd.concat([pred, dt.predict(pred)], axis=1))
dt.test(pred, np.array(["no", "no", "no", "maybe", "yes"]), display=True)

cont_df = pd.read_csv(StringIO("""Outlook Temperature Playball
Sunny 15.5 Yes
Sunny 0.5 No
Overcast 10.0 No
Rain 35.5 No
Rain 17.7 No
Rain 15.5 Yes
Rain 10.0 No
Sunny 25.0 No
Sunny 19.9 Yes
Rain 22.0 Yes
Sunny 15.0 No
Overcast 10.5 No
Overcast 32 Yes
Rain 5 No
"""), sep=" ")

X = cont_df.drop('Playball', axis=1)
y = cont_df['Playball']
cols = X.columns.values
dt = DecisionTree()
dt.train(X, y, cols)
print(dt.model)
