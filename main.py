import inline as inline
import pandas as pd
import matplotlib
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from DecisionTree import DecisionTree
from KNN import KNN
from SVM import SVM
from adaboost import adaboost
from logistic import logistic

matplotlib.rcParams['figure.figsize'] = (12, 8)

df = pd.read_csv("movies.csv")

# Create Classification version of target variable
dff = df.drop(['writer', 'star', 'released', 'name', 'director', 'company', 'country'], axis=1)

dff['grossquality'] = ['good' if x >= 40000000 else 'bad' for x in df['gross']]
dff['scorequality'] = ['good' if (x >= 6.7 and y >= 100) else 'bad' for x, y in zip(dff['score'], dff['runtime'])]

dff['genre'] = [1 if x == ("Adventure")
                else 2 if x == "Comedy"
else 3 if x == "Action"
else 4 if x == "Drama"
else 5 if x == "Crime"
else 6 if x == "Thriller"
else 7 if x == "Horror"
else 8 if x == "Animation"
else 9 if x == "Biography"
else 10 if x == "Sci-Fi"
else 11 if x == "Musical"
else 12 if x == "Family"
else 13 if x == "Fantasy"
else 14 if x == "Mystery"
else 15 if x == "War"
else 16 if x == "Romance"
else 17 for x in df['genre']]

dff['rating'] = [1 if x == "R"
                 else 2 if x == "PG-13"
else 3 if x == "PG"
else 4 if x == "UNRATED"
else 5 if x == "Not specified"
else 6 if x == "G"
else 7 if x == "NC-17"
else 8 if x == "NOT RATED"
else 9 if x == "TV-PG"
else 10 if x == "TV-MA"
else 11 if x == "B"
else 12 if x == "B15"
else 13 for x in df['rating']]

# Separate feature variables and target variable
data_gross = dff.drop(['gross', 'grossquality', 'scorequality'], axis=1)
data_score = dff.drop(['runtime', 'score', 'grossquality', 'scorequality'], axis=1)

cl_gross = dff['grossquality']
cl_score = dff['scorequality']

# gross
print("\nResult for gross only :")
KNN(data_gross, cl_gross)
DecisionTree(data_gross, cl_gross)
logistic(data_gross, cl_gross)
SVM(data_gross, cl_gross)
adaboost(data_gross, cl_gross)

# score & runtime
print("\nResult for score & runtime :")
KNN(data_score, cl_score)
DecisionTree(data_score, cl_score)
logistic(data_score, cl_score)
SVM(data_score, cl_score)
adaboost(data_score, cl_score)

###################################################### PLOTS ################################################################

# figure = plt.figure(figsize=(15, 5))
# plt.scatter(x=df['score'], y=df['runtime'], color='red')
# plt.title("Score vs Runtime", size=30)
# plt.xlabel("Score", size=20)
# plt.ylabel("Runtime", size=20)
# plt.show()


# Countries = pd.DataFrame(df['genre'].value_counts())
# Ten_countries = pd.DataFrame(df['genre'].value_counts())
#
# sns.barplot(x=Ten_countries.index, y=Ten_countries['genre'])

# labels = Ten_countries.index.tolist()
# plt.gcf().set_size_inches(15, 7)

# plt.title('Genre vs movies released', fontsize=20)
# plt.xlabel('genre', fontsize=15)
# plt.ylabel('Movies released', fontsize=15)
#
# plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10 ,11,12,13,14,15,16], labels=labels, rotation='45')
# plt.show()

##histograma

# fig = px.histogram(dff, x='score')
# fig.show()
# fig = px.histogram(dff, x='runtime')
# fig.show()
# print(dff['scorevote'])
