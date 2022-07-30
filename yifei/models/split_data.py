from sklearn.model_selection import train_test_split
import pandas as pd

all_train = pd.read_csv("../preprocess/train.csv")

features = all_train.columns.tolist()
features.remove("card_id")
features.remove("target")

x_data = all_train[features]
y_data = all_train['target']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

