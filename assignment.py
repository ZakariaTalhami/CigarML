import csv
from pprint import pprint

from sklearn import linear_model

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, sem
import statsmodels.api as sm
from sklearn.metrics import matthews_corrcoef, mean_squared_error, r2_score
from sklearn import linear_model

target = "Price"


def remove_key(d, key):
    r = dict(d)
    del r[key]
    return r


def get_pvalues(X, y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
    return est2.pvalues


# Read the CSV file
with open("Cigar.csv", "r") as readfile:
    fieldnames = ["State", "Year", "Price", "Pop", "Pop 16", "Cpi", "Ndi", "Sales", "Pimin"]
    reader = csv.DictReader(readfile, fieldnames=fieldnames)
    header = next(reader)
    # data = [row for row in reader]
    data = []
    for row in reader:
        row["State"] = int(row["State"])
        row["Year"] = int(row["Year"])
        row["Price"] = float(row["Price"])
        row["Pop"] = float(row["Pop"])
        row["Pop 16"] = float(row["Pop 16"])
        row["Cpi"] = float(row["Cpi"])
        row["Ndi"] = float(row["Ndi"])
        row["Sales"] = float(row["Sales"])
        row["Pimin"] = float(row["Pimin"])
        data.append(row)

    # Extract the predictors and target variables

target_data = [x[target] for x in data]
predictor_data = [remove_key(x, target) for x in data]

v = 1
corr = []
for key in header:
    if key != target:
        # plt.subplot(len(header)/2, 1 , v)
        fig = plt.figure()
        variables = [x[key] for x in predictor_data]
        plt.scatter(variables, target_data, color="red", s=10)
        plt.title("{} scatter".format(key))
        # plt.show()
        plt.savefig("scatter/{}_scatter.png".format(key), bbox_inches='tight', dpi=fig.dpi)
        R, _ = pearsonr(variables, target_data)
        corr.append({key: R})
        # v += 1

# plt.show()

pprint(corr)
# Calculate the Correlation

reg = linear_model.LinearRegression()

reg.fit([list(x.values()) for x in predictor_data], target_data)

# mean_squared_error()
predicted_data = reg.predict([list(x.values()) for x in predictor_data])

pprint(reg.coef_)
pprint(reg.intercept_)
pprint(sem(predicted_data))
r2 = r2_score(target_data, predicted_data)
pprint(r2)

n = len(predicted_data)
p = len(header) - 1

ar2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
pprint(ar2)

# corr of observed and predicted
R, _ = pearsonr(target_data, predicted_data)
pprint(R)
# p-values
pprint(get_pvalues([list(x.values()) for x in predictor_data] , target_data))



print("ss")
