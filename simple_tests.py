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
