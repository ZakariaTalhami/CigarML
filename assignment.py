import csv
from pprint import pprint
from sklearn import linear_model

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, mean_squared_error
from sklearn import linear_model
target = "Price"


def remove_key(d, key):
    r = dict(d)
    del r[key]
    return r


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
        R, _ = pearsonr(variables , target_data)
        corr.append({key:R})
        # v += 1

# plt.show()

pprint(corr)
# Calculate the Correlation

reg = linear_model.LinearRegression()

reg.fit([list(x.values()) for x in predictor_data] , target_data)
# mean_squared_error()

pprint(reg.coef_)
pprint(reg.intercept_)

print("ss")
