import pandas as pd
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
mozza tomato no yes yes"""), sep=" ")

X = df.drop('like', axis=1)
y = df['like']
cols = X.columns.values
dt = DecisionTree()
dt.train(X, y, cols)
print(dt.model)
