import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn import preprocessing

target = "pimin"

# Read CSV File
df = pd.read_csv("Cigar.csv")

df_target = pd.DataFrame(df[target])
df_predictors = df.drop(target, 1)
min_max_scaler = preprocessing.MinMaxScaler()
df_target_scaled = pd.DataFrame(min_max_scaler.fit_transform(df_target), columns=[target])
# pprint(df_target_scaled)
mean = df_target_scaled[target].mean()
binarizer = preprocessing.Binarizer(threshold=mean)
df_target_binarized = pd.DataFrame(binarizer.transform(df_target_scaled), columns=[target])

df_predictors = sm.add_constant(df_predictors)
model = sm.Logit(df_target_binarized, df_predictors).fit()
predictions = model.predict(df_predictors)
predictions2 = (predictions > 0.5).astype(int)
print(model.summary())
print(model.summary2())
pprint(predictions)

df_predictors['predictions'] = predictions
pprint(df_predictors)


for clm in df_predictors.columns.values:
    plt.scatter(df_predictors[clm] , predictions2)
    plt.savefig("scatter/part2/{}_{}_part2.png".format(clm,"predicted"), bbox_inches='tight')
    plt.show()
    plt.scatter(df_predictors[clm] , predictions)
    plt.savefig("scatter/part2/{}_{}_part2.png".format(clm,"predicted_prob"), bbox_inches='tight')
    plt.show()

