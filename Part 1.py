import time

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from pprint import pprint

from scipy.stats import pearsonr
from sklearn import preprocessing


def sig_level(corr_cof_abs: dict):
    ret = {}
    for key, value in corr_cof_abs.items():
        if (value > 0.5):
            ret.update({key: "High"})
        elif value > 0.3:
            ret.update({key: "Medium"})
        else:
            ret.update({key: "Low"})
    return ret


target = "price"

# Read CSV File
df = pd.read_csv("Cigar.csv")
predictor_corr_cof = {}
for clm in df.columns.values:
    if clm != target:
        plot = sns.scatterplot(y=target, x=clm, data=df)
        fig = plot.get_figure()
        fig.show()
        fig.savefig("scatter/part1/{}_scatter.png".format(clm), bbox_inches='tight', dpi=fig.dpi)
        R, _ = pearsonr(df[clm], df[target])
        predictor_corr_cof.update({clm: R})

predictor_corr_cof_abs = {key: abs(value) for key, value in predictor_corr_cof.items()}
predictor_corr_sig_level = sig_level(predictor_corr_cof_abs)

print("{:^15} {:^15} {:^15} {:^15}".format("Predictor", "Correlation cof", "Abs Correlation cof", "Significance Level"))
print("{:^15} {:^15} {:^15} {:^15}".format("---------", "---------------", "-------------------", "------------------"))
for key, value in predictor_corr_cof_abs.items():
    print("{:^15} {:^15} {:^15} {:^15}".format(key, round(predictor_corr_cof[key], 5),
                                               round(predictor_corr_cof_abs[key], 5),
                                               predictor_corr_sig_level[key]))

df_target = pd.DataFrame(df[target])
df_predictors = df.drop(target, 1)

df_predictors = sm.add_constant(df_predictors)
model = sm.OLS(df_target, df_predictors).fit()
predictions = pd.DataFrame(model.predict(df_predictors))
print(model.summary())
R, _ = pearsonr(df_target, predictions)
print("Corr of prediction with ob :> {}".format(R))

print("\nDropped predictors:>")
for key, pred in model.pvalues.items():
    if abs(pred) > 0.05:
        print(key)
