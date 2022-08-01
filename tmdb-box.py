#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import re
from scipy.stats import pearsonr
import math
from statistics import median
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xbg
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.tree import DecisionTreeRegressor
from numpy import mean
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None
#%%
# STEP [1]================================== Look At The Big Picture ================================================
# %%
# STEP [2]================================== Get The Data ================================================
df = pd.read_csv('../datasets/films_revenue_dataset.csv')

# %%
# STEP [3]================================== Discover and Visualize The Data to Gain Insights ================================================
df.head(5)
# %%
df.info()
# %%
print(f"dataset data has {df.shape[0]} rows and {df.shape[1]} columns")

# %%
print(f"Missing values of dataset is:\n {df.isna().sum()}")
# %%
# Phân tích dữ liệu
train_features = []
log_features = []
drop_cols = []

# %%
# Phân tích cột Revenue - doanh thu
print(f"Missing values of revenue in train sets is {df['revenue'].isna().sum()}")
# %%
print(f"Mean of revenue in dataset set is {df['revenue'].mean()}")
print(f"Median of revenue in dataset set is {df['revenue'].median()}")
# %%
fig, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(x=df['revenue'], ax = ax[0], color='pink')
ax[0].set_title("Box Plot of revenue variable")
sns.distplot(a=df['revenue'], kde = False, ax = ax[1], color='pink', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of revenue variable")
fig.tight_layout()
# %%
df['revenue'].nsmallest(20)
# %%
df['revenue'].nlargest(10)
# %%
log_features.append('revenue')



# %% NEW
# Phân tích cột Belong to collections
# Cột này biểu diễn xem một bộ phim có nằm trong tập hay không.
# Tạm thời thì những bộ phim nào có thuộc bộ sưu tập sẽ cho là 1 ngược lại là 0
df['belongs_to_collection'] = df['belongs_to_collection'].fillna('')
df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: 1 if len(x) != 0 else 0)
# %%
df['belongs_to_collection'].value_counts()
# %%
labels = "Belongs to Collection", "Not Belongs to Collection"
btc_1 = [df['belongs_to_collection'].value_counts()[1], df['belongs_to_collection'].value_counts()[0]]
explode = (0.1, 0)
colors = ['red', 'pink']
# plot
fig, ax = plt.subplots()
ax.pie(btc_1, explode=explode, autopct="%1.1f%%", shadow=True, startangle=90, colors=colors, textprops={'color': 'w', 'fontsize': 22, 'weight': 'bold'})
ax.axis('equal')
ax.legend(labels, title='Legend', loc='center left', fontsize=14, bbox_to_anchor=(0.8, 0.25, 0.5, 1))

# Nhận xét: Có thể thuộc tính belongs_to_collection có thể ảnh hưởng đến doanh thu của bộ phim đó

# %%
print(df['belongs_to_collection'].value_counts())
labels = "Belongs to Collection", "Not Belongs to Collection"
pal = ['red', 'pink']
#plot
fig, ax = plt.subplots(figsize=(12,6))
sns.boxplot(x="belongs_to_collection", y = 'revenue', data=df, palette=pal)
plt.title("Revenue of films based on attribute Belongs_to_collection")
plt.xlabel("Collection's status")
plt.ylabel("Revenue")

ax.set_xticklabels(labels)

# Chắc chắn việc doanh thu của một bộ phim có phụ thuộc vào việc bộ phim đó có thuộc vào một bộ sưu tập hay không.
# %%
train_features.append("belongs_to_collection")
# %%
corr, _ = pearsonr(df['belongs_to_collection'], df['revenue'])
print('Pearsons correlation between belongs_to_collection and revenue: %.3f' % corr)




# %% NEW
# Phân tích cột budget - kinh phí
print(f"Missing values of budget  is {df['budget'].isna().sum()}")
# %%
#plotting the data
fig, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(x=df['budget'], ax = ax[0], color='navy')
ax[0].set_title("Box Plot of budget")
sns.distplot(a=df['budget'], kde = False, ax = ax[1], color='navy', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of budget")
fig.tight_layout()
# %%
print(f"Number of films have budget is 0: {sum(df['budget'] == 0)}")

# ==> Có vấn đề khi kinh phí làm phim không thể nào bằng 0.
# %%
median_budget_higer_0 = df.loc[df['budget']> 0, 'budget'].median()
median_budget_higer_0
# %%
df['budget_processed'] = df['budget'].mask(df['budget'] == 0, median_budget_higer_0)
# %%
df['budget_processed'].nsmallest(10)

# Vẫn còn vấn đề khi kinh phí thấp nhất để hoàn thành một bộ phim là 7000 đô (theo google)
# %%
median_budget_higher_10000 = df.loc[df['budget']>= 10000, 'budget'].median()
df['budget_processed'] = df['budget'].mask(df['budget'] < 10000, median_budget_higher_10000)
# %%
# Trực quan hóa
fig, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(x=df['budget_processed'], ax = ax[0], color='mediumaquamarine')
ax[0].set_title("Box Plot of budget_processed variable")
sns.distplot(a=df['budget_processed'], kde = False, ax = ax[1], color='mediumaquamarine', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of budget_processed variable")
fig.tight_layout()
# %%
log_features.append('budget_processed')
# %%
corr, _ = pearsonr(df['budget_processed'], df['revenue'])
print('Pearsons correlation between budget_processed and revenue: %.3f' % corr)



# %% NEW
# Phân tích cột genres
df['genres'].head(10)

# %%
print(f"Missing values of genres is {df['genres'].isna().sum()}")
# %%
df['genres'] = df['genres'].fillna('')
df['genres_reform'] = df['genres'].apply(lambda x:re.findall("'name': \'(.+?)\'", x))

#%%
df['genres_reform']

#%%
def get_genres(list, df):
    for i in range(len(df)):
        for name in list[i]:
            if name not in df.columns:
                df[name] = 0
            df[name][i] = 1
    return df
# %%
df = get_genres(df['genres_reform'], df)
print(df.columns)
#%%
df['num_genres'] = df['genres_reform'].apply(lambda x : len(x))
#%%
df['num_genres'].value_counts()
# %%
# Lấy số lượng phim theo thể loại phim
genresDict = dict()

for genre in df["genres_reform"]:
    for elem in genre:
        if elem not in genresDict:
            genresDict[elem] = 1
        else:
            genresDict[elem] += 1

#%%
sns.set(rc={'figure.figsize':(12,6)})
genres_df = pd.DataFrame.from_dict(genresDict, orient='index')
genres_df.columns = ["number_of_movies"]
genres_df = genres_df.sort_values(by="number_of_movies", ascending=False)
genres_df.plot.bar(color='royalblue')
plt.title("Number of films per genre")
plt.ylabel("Number of Films")
plt.xlabel("Genre")

# %%
# Giống one-hot-coder
for genre in genres_df.index.values:
    df[genre] = df['genres_reform'].apply(lambda x: 1 if genre in x else 0)

#%%
# median and mean revenue per genre type
for index, genre in enumerate(genres_df.index.values):
    genres_df.loc[genre, "median_rev"] = df[df[genre]==1].revenue.median()
    genres_df.loc[genre, "mean_rev"] = df[df[genre]==1].revenue.mean()
    

# %%
genres_df.sort_values(by=["median_rev"], ascending=False).median_rev.plot.bar(color='royalblue')
plt.title("Film Genre by Median Revenue")
plt.ylabel("Revenue")
plt.xlabel("Genre")
# %%
genres_df.sort_values(by=["mean_rev"], ascending=False).mean_rev.plot.bar(color='royalblue')
plt.title("Film Genre by Mean Revenue")
plt.ylabel("Revenue")
plt.xlabel("Genre")
# %%
topGenreDict = {}
for element in df[["revenue", "genres_reform"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in topGenreDict:
        topGenreDict[element[1][0]] = [element[0], 1]
    else:
        topGenreDict[element[1][0]][0] += element[0]
        topGenreDict[element[1][0]][1] += 1  
topGenreDict  
#%%
for genre in topGenreDict:
    topGenreDict[genre][0] = topGenreDict[genre][0]/topGenreDict[genre][1]
    topGenreDict[genre] = topGenreDict[genre][0]

#%%
topGenreDict   
#%%    
genres_df = pd.DataFrame.from_dict(topGenreDict, orient='index', columns=["mean_movies_revenue"])
genres_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='royalblue')
# %%
topGenreDict = {k: v for k, v in sorted(topGenreDict.items(), key=lambda item: item[1], reverse = False)}
#%%
topGenreDict
# %%
def getGenreRank(genres):
    sum = 0
    for g in genres:
        sum += list(topGenreDict.keys()).index(g)
    return (sum / len(genres))
# %%
df['genre_rank'] = df['genres_reform'].apply(lambda x: getGenreRank(x) if len(x) > 0 else 0)

df['genre_rank'].value_counts()
#%%
df['genre_rank'].head(5)
# %%
log_features.append('genre_rank')
log_features.append('num_genres')
# %%
corr, _ = pearsonr(df['genre_rank'], df['revenue'])
print('Pearsons correlation between genre_rank and revenue: %.3f' % corr)
corr, _ = pearsonr(df['num_genres'], df['revenue'])
print('Pearsons correlation between num_genres and revenue: %.3f' % corr)


# %% NEW
# Phân tích Homepage
print(f"Mising values of homepage is {df['homepage'].isna().sum()}")

#%%
df['homepage'] = df['homepage'].fillna('')
df['has_homepage'] = df['homepage'].apply(lambda x: 1 if len(x) != 0 else 0)
# %%
df['has_homepage'].value_counts()
# %%
labels = 'Has a homepage', 'Does not have a homepage'
btc_1 = [sum(df.has_homepage), (df.has_homepage == 0).sum(axis=0)]
explode = (0.1, 0)

colors = ['pink', 'red']

fig, ax = plt.subplots()
ax.pie(btc_1, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'color':"w", 'fontsize': 22, 'weight':"bold"})
ax.axis('equal')

ax.legend(labels,
          title="Legend",
          loc="center left",
          fontsize=14,
          bbox_to_anchor=(0.8, 0.25, 0.5, 1))

plt.show()
# %%
print("Mean revenue for movies with a homepage: %.2f" % df.loc[df['has_homepage'] == 1, "revenue"].mean())
print("Median revenue for movies with a homepage: %.2f" % df.loc[df['has_homepage'] == 1, "revenue"].median())
print("\n")
print("Mean revenue for movies without a homepage: %.2f" % df.loc[df['has_homepage'] == 0, "revenue"].mean())
print("Median revenue for movies without a homepage: %.2f" % df.loc[df['has_homepage'] == 0, "revenue"].median())
# %%
pal1 = ['red', 'pink']

ax = sns.boxplot(x='has_homepage', y='revenue', data=df, palette=pal1);
plt.title('Revenue for films with and without a homepage');

labels = 'Has a homepage', 'Does not have a homepage'
ax.set_xticklabels(labels)
# %%
train_features.append('has_homepage')
# %%
corr, _ = pearsonr(df['has_homepage'], df['revenue'])
print('Pearsons correlation between has_homepage and revenue: %.3f' % corr)


# %% NEW
# Phân tích cột IMDB_id
drop_cols.append("imdb_id")


# %% NEW
# Phân tích cột original_language
print("Counts of each original language:")
print(df['original_language'].value_counts()[0:5])
# %%
sns.boxplot(x='original_language', y='revenue', color='deepskyblue', data=df.loc[df['original_language'].isin(df['original_language'].value_counts().head(10).index)])
plt.title('Revenue for a movie and its and original_language')
# %%
df['originally_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
df['originally_english'].value_counts()
# %%
labels = 'English', 'Not English'
btc_1 = [(df['originally_english'] == 1).sum(axis=0), (df['originally_english'] == 0).sum(axis=0)]
explode = (0.1, 0)

colors = ['red', 'pink']

fig, ax = plt.subplots()
ax.pie(btc_1, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'color':"w", 'fontsize': 22, 'weight':"bold"})
ax.axis('equal')

ax.legend(labels,
          title="Legend",
          loc="center left",
          fontsize=14,
          bbox_to_anchor=(0.8, 0.25, 0.5, 1))

plt.show()

#%%
pal2 = ['red', 'pink']

ax = sns.boxplot(x='originally_english', y='revenue', data=df, palette=pal2);
plt.title('Revenue for films orginally in english vs other languages');

labels = 'Not in English', 'In English'
ax.set_xticklabels(labels)

#%%
train_features.append('originally_english')
# %%
corr, _ = pearsonr(df['originally_english'], df['revenue'])
print('Pearsons correlation between in_english and revenue: %.3f' % corr)



# %% NEW
# Phân tích cột Orinal_title
drop_cols.append('original_title')


# %% NEW
# Phân tích cột overview
print(f"Missing values of Overview: {df['overview'].isna().sum()}")
# %%
drop_cols.append('overview')


# %% NEW
# Phân tích cột popularity
print(f"Missing values of popularity is {df['popularity'].isna().sum()}")
# %%
sns.distplot(df['popularity'], kde=False, color='pink', hist_kws=dict(alpha=1))
sns.set(rc={'figure.figsize':(12, 2)})
plt.title('Histogram of Film Popularity')
plt.show();
# %%
print(df['popularity'].describe())
# %%
sns.set(rc={'figure.figsize':(12, 6)})
plt.plot(df['revenue'], df['popularity'], 'o', color='pink')
plt.ylabel("Popularity")
plt.xlabel("Revenue")
plt.title("Popularity of Films by Revenue")

#%%
drop_cols.append('popularity')


# %% NEW
# Phân tích cột Poster path
drop_cols.append('poster_path')


# %% NEW
# Phân tích cột production_companies
print(f"Missing values of production companies is {df['production_companies'].isna().sum()}")

# %%
df['production_companies'].head(5)
# %%
df['production_companies'] = df['production_companies'].fillna('')
df['production_companies_reform'] = df['production_companies'].apply(lambda x:re.findall("'name': \'(.+?)\'", x))


df['production_companies'] = df['production_companies'].fillna('')
df['production_companies_reform'] = df['production_companies'].apply(lambda x:re.findall("'name': \'(.+?)\'", x))
# %%
df['production_companies_reform']
# %%
df['num_studios'] = df['production_companies_reform'].apply(lambda x: len(x))
df['num_studios'] = df['production_companies_reform'].apply(lambda x: len(x))

df['num_studios'].value_counts()
# %%
sns.distplot(df['num_studios'], kde=False, color='pink', hist_kws=dict(alpha=1))
sns.set(rc={'figure.figsize':(15,6)})
plt.title('Histogram of Number of Production Studios')
plt.show();

# %%
companiesDict = {}
for element in df["production_companies_reform"].values:
    for company in element:
        if company not in companiesDict:
            companiesDict[company] = 1
        else:
            companiesDict[company] += 1

#%%
companies_df = pd.DataFrame.from_dict(companiesDict, orient='index', columns=["number_of_studios"])
#%%
companies_df.index.values
for company in companies_df.index.values:
    df[company] = df['production_companies_reform'].apply(lambda x: 1 if company in x else 0)
#%%
# median revenue per production studio
for index, company in enumerate(companies_df.index.values):
    companies_df.loc[company, "median_rev"] = df[df[company]==1].revenue.median()
    companies_df.loc[company, "mean_rev"] = df[df[company]==1].revenue.mean()
    companies_df.loc[company, "sum_rev"] = df[df[company]==1].revenue.sum()
    
#%%
companies_df.head(10)
# %%
companies_df.sort_values(by=["median_rev"], ascending=False).median_rev.head(20).plot.bar(color='mediumseagreen')
plt.title("Production Studios by Median Revenue")
plt.ylabel("Revenue")
plt.xlabel("Production Studio")
# %%
companies_df.sort_values(by=["mean_rev"], ascending=False).mean_rev.head(20).plot.bar(color='mediumseagreen')
plt.title("Production Studios by Mean Revenue")
plt.ylabel("Revenue")
plt.xlabel("Production Studio")
# %%
companies_df.sort_values(by=["sum_rev"], ascending=False).sum_rev.head(20).plot.bar(color='mediumseagreen')
plt.title("Production Studios by Total Revenue")
plt.ylabel("Revenue")
plt.xlabel("Production Studio")
# %%
studiosDict = {}
for element in df[["revenue", "production_companies_reform"]].values:
    temp = 3
    if len(element[1]) < temp:
        temp = len(element[1])
    for i in range(temp):
        if element[1][i] not in studiosDict:
            studiosDict[element[1][i]] = [element[0], 1]
        else:
            studiosDict[element[1][i]][0] += element[0]
            studiosDict[element[1][i]][1] += 1    

studiosDict = {k: v for k, v in studiosDict.items() if v[1] >= 3}
#%%
studiosDict
#%%
for company in studiosDict:
    studiosDict[company][0] = studiosDict[company][0]/studiosDict[company][1]
    studiosDict[company] = studiosDict[company][0]


    
    
studios_df = pd.DataFrame.from_dict(studiosDict, orient='index', columns=["mean_movies_revenue"])
studios_df.sort_values(by="mean_movies_revenue", ascending=False).head(20).plot.bar(color='mediumseagreen')
plt.title("Primary (Top 3) Production Studios by Mean Revenue")
plt.ylabel("Revenue")
plt.xlabel("Production Studio")
# %%
topStudiosDict = {}
for element in df[["revenue", "production_companies_reform"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in topStudiosDict:
        topStudiosDict[element[1][0]] = [element[0], 1]
    else:
        topStudiosDict[element[1][0]][0] += element[0]
        topStudiosDict[element[1][0]][1] += 1      

#%%
topStudiosDict
#%%
topStudiosDict = {k: v for k, v in topStudiosDict.items() if v[1] >= 5}

#%%
for company in topStudiosDict:
    topStudiosDict[company][0] = topStudiosDict[company][0]/topStudiosDict[company][1]
    topStudiosDict[company] = topStudiosDict[company][0]


    
    
studios_df = pd.DataFrame.from_dict(topStudiosDict, orient='index', columns=["mean_movies_revenue"])
studios_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='mediumseagreen')
# %%
topStudiosDict = {k: v for k, v in sorted(topStudiosDict.items(), key=lambda item: item[1], reverse = True)}
studiosDict = {k: v for k, v in sorted(studiosDict.items(), key=lambda item: item[1], reverse = True)}
# %%
def checkTopStudios(studio):
    if len(studio) < 1 or studio[0] not in list(topStudiosDict)[:50]:
        return 0
    else:
        return 1
# %%
def checkStudios(studio):
    if len(studio) < 1:
        return 0
    count = 0
    for company in studio[:10]:
        if company in list(studiosDict)[:100]:
            count += 1
    return count
# %%
def getStudioRanks(studios):
    if len(studios) < 1:
        return 400
    rank = 0
    for s in studios[:5]:
        if s in list(studiosDict):
            rank += list(studiosDict.keys()).index(s)
    if rank == 0:
        rank = 400
    return rank / len(studios)

#%%
df.columns
# %%
df['topStudio'] = df['production_companies_reform'].apply(lambda x: checkTopStudios(x))
df['numTopStudios'] = df['production_companies_reform'].apply(lambda x: checkStudios(x))

# %%
df['studioRank'] = df['production_companies_reform'].apply(lambda x: getStudioRanks(x))
# %%
print(df['topStudio'].value_counts())
print()
print(df['numTopStudios'].value_counts())
print()
print(df['studioRank'].value_counts())
# %%
corr, _ = pearsonr(df['topStudio'], df['revenue'])
print('Pearsons correlation between topStudio and revenue: %.3f' % corr)

corr, _ = pearsonr(df['numTopStudios'], df['revenue'])
print('Pearsons correlation between numTopStudios and revenue: %.3f' % corr)

corr, _ = pearsonr(df['studioRank'], df['revenue'])
print('Pearsons correlation between studioRank and revenue: %.3f' % corr)

corr, _ = pearsonr(df['num_studios'], df['revenue'])
print('Pearsons correlation between num_studios and revenue: %.3f' % corr)
# %%
train_features.append('topStudio')
log_features.append('numTopStudios')
log_features.append('num_studios')
log_features.append('studioRank')



# %% NEW
# Phân tích cột Production Countries
print(f"Missing values of production_countries is {df['production_countries'].isna().sum()}")
# %%
df['production_countries'].head(10)
# %%
df['production_countries'] = df['production_countries'].fillna("")
df['production_countries_processed'] = df['production_countries'].apply(lambda x: re.findall("'name': \'(.+?)\'", x))
df['num_production_countries'] = df['production_countries_processed'].apply(lambda x: len(x))
# %%
sns.set(rc={'figure.figsize':(12,8)})
sns.countplot(x=df['num_production_countries'], color='cornflowerblue')
# %%
countriesDict = {}
for element in df["production_countries_processed"].values:
    for country in element:
        if country not in countriesDict:
            countriesDict[country] = 1
        else:
            countriesDict[country] += 1

countries_df = pd.DataFrame.from_dict(countriesDict, orient='index', columns=["movies per country"])
countries_df.sort_values(by="movies per country", ascending=False).head(10).plot.bar(color='cornflowerblue')
# %%
df['usa_produced'] = df['production_countries_processed'].apply(lambda x: 1 if 'United States of America' in x else 0)
df['usa_produced'].value_counts()
# %%
labels = 'Produced in USA', 'Produced outside of USA'
btc = [df['usa_produced'].value_counts()[1], df['usa_produced'].value_counts()[0]]
explode = (0.1, 0)

colors = ['cornflowerblue', 'lightsteelblue']

fig, ax = plt.subplots()
ax.pie(btc, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'color':"w", 'fontsize': 22, 'weight':"bold"})
ax.axis('equal') # ensures chart is a circle

ax.legend(labels,
          title="Legend",
          loc="center left",
          fontsize=14,
          bbox_to_anchor=(0.8, 0.25, 0.5, 1))

plt.show() 
# %%
sns.boxplot(x='num_production_countries', y='revenue', data=df, color='cornflowerblue')
plt.title('Revenue based on number of production countries')
# %%
corr, _ = pearsonr(df['num_production_countries'], df['revenue'])
print('Pearsons correlation between num_production_countries and revenue: %.3f' % corr)
# %%
pal = ['red', 'pink']

ax = sns.boxplot(x='usa_produced', y='revenue', data=df, palette=pal);
plt.title('Revenue for films produced in the USA vs produced in other countries');

labels = 'Produced outside of USA', 'Produced in USA'
ax.set_xticklabels(labels)
# %%
corr, _ = pearsonr(df['usa_produced'], df['revenue'])
print('Pearsons correlation between usa_produced and revenue: %.3f' % corr)
# %%
log_features.append('num_production_countries')
train_features.append('usa_produced')




# %% NEW
# Phân tích cột release_date
df['release_date'].head(10)

# %%
df_date = df['release_date']

# converting to datetime format, with .dt used for accessing quantities
df_date = pd.to_datetime(df_date)
df_date.dt
print(df_date)
# %%
fig, dx = plt.subplots()
sns.distplot(df_date.dt.year, bins=99, kde=False, color='gold', hist_kws=dict(alpha=1))
sns.set(rc={'figure.figsize':(12, 6)})

dx.set(xlim=(1965, 2020),ylim=(0,200))
dx.set_xlabel("Year")
dx.set_ylabel("Number of Films")
dx.set_title("Year of Release Date by Number of Films")
# %%
fig, ex = plt.subplots()
sns.distplot(df_date.dt.dayofweek, kde=False, color='gold', hist_kws=dict(alpha=1))
sns.set(rc={'figure.figsize':(15,6)})

ex.set_xlabel("Day of Week")
ex.set_ylabel("Number of Films")
ex.set_title("Release Date Day of the Week")

labels = [item.get_text() for item in ex.get_xticklabels()]
labels[1] = 'Monday'
labels[2] = 'Tuesday'
labels[3] = 'Wednesday'
labels[4] = 'Thursday'
labels[5] = 'Friday'
labels[6] = 'Saturday'
labels[7] = 'Sunday'

ex.set_xticklabels(labels)
# %%
fig, fx = plt.subplots()
sns.distplot(df_date.dt.dayofyear, bins=365, kde=False, color='gold', hist_kws=dict(alpha=1))
sns.set(rc={'figure.figsize':(15,6)})

fx.set_xlabel("Day of Year")
fx.set_ylabel("Number of Films")
fx.set_title("Release Date Day of the Year")
# %%
fig, gx = plt.subplots()
sns.distplot(df_date.dt.weekofyear, bins=52, kde=False, color='gold', hist_kws=dict(alpha=1))
sns.set(rc={'figure.figsize':(12,6)})

gx.set_xlabel("Week of Year")
gx.set_ylabel("Number of Films")
gx.set_title("Release Date Week of the Year")

print()

# plotting the films release month of the year
fig, hx = plt.subplots()
sns.distplot(df_date.dt.month, bins=12, kde=False, color='gold', hist_kws=dict(alpha=1))
sns.set(rc={'figure.figsize':(12,6)})

hx.set_xlabel("Month of Year")
hx.set_ylabel("Number of Films")
hx.set_title("Release Date Month of the Year")
# %%
fig, dx = plt.subplots()
sns.boxplot(x=df_date.dt.year, y=df['revenue'], color='pink')
sns.set(rc={'figure.figsize':(12, 6)})

dx.set(xlim=(-5, 50))
dx.set_xlabel("Year of Film")
dx.set_ylabel("Revenue")
dx.set_title("Year of Release Date by Revenue")

dx.set_xticklabels(dx.get_xticklabels(), rotation=90)
# %%
fig, ex = plt.subplots()
sns.boxplot(x=df_date.dt.dayofweek, y=df['revenue'], color='gold')
sns.set(rc={'figure.figsize':(12, 6)})

ex.set_xlabel("Day of Week")
ex.set_ylabel("Revenue")
ex.set_title("Release Date Day of the Week by Revenue")

labels = [item.get_text() for item in ex.get_xticklabels()]
labels[0] = 'Monday'
labels[1] = 'Tuesday'
labels[2] = 'Wednesday'
labels[3] = 'Thursday'
labels[4] = 'Friday'
labels[5] = 'Saturday'
labels[6] = 'Sunday'

ex.set_xticklabels(labels)
# %%
fig, fx = plt.subplots()
sns.boxplot(x=df_date.dt.dayofyear, y=df['revenue'], color='gold')
sns.set(rc={'figure.figsize':(12, 6)})

fx.set_xlabel("Day of Year")
fx.set_ylabel("Revenue")
fx.set_title("Release Date Day of the Year by Revenue")
# %%
fig, gx = plt.subplots()
sns.boxplot(x=df_date.dt.weekofyear, y=df['revenue'], color='gold')
sns.set(rc={'figure.figsize':(12, 6)})

gx.set_xlabel("Week of Year")
gx.set_ylabel("Revenue")
gx.set_title("Release Date Week of the Year by Revenue")
# %%
fig, hx = plt.subplots()
sns.boxplot(x=df_date.dt.month, y=df['revenue'], color='gold')
sns.set(rc={'figure.figsize':(12, 6)})

hx.set_xlabel("Month of Year")
hx.set_ylabel("Revenue")
hx.set_title("Release Date Month by Revenue")

#%%
df["release_date"].mode()

# %%
df["release_date"] = df["release_date"].fillna(df["release_date"].mode()[0])
df['temp'] = pd.to_datetime(df['release_date'])

df["month"] = df["temp"].apply(lambda x: x.month)
df["year"] = df["temp"].apply(lambda x: x.year)

df["day_of_week"] = df["temp"].apply(lambda x: x.weekday()+1)

df["week_of_year"] = df["temp"].apply(lambda x: x.isocalendar()[1])

df = df.drop(['temp'], axis=1)


df["day_of_week"] = df["day_of_week"].fillna(df["day_of_week"].mode()[0])

df["year"] = df["year"].fillna(df["year"].mode()[0])

df["month"] = df["month"].fillna(df["month"].mode()[0])

df["week_of_year"] = df["week_of_year"].fillna(df["week_of_year"].mode()[0])


df[["release_date", "month", "year", "day_of_week", "week_of_year"]].head()
# %%
corr, _ = pearsonr(df['year'], df['revenue'])
print('Pearsons correlation between year and revenue: %.3f' % corr)

corr, _ = pearsonr(df['month'], df['revenue'])
print('Pearsons correlation between month and revenue: %.3f' % corr)

corr, _ = pearsonr(df['week_of_year'], df['revenue'])
print('Pearsons correlation between week_of_year and revenue: %.3f' % corr)

# %%
df['1960s'] = df['year'].map(lambda x: 1 if (x >= 1960 and x <= 1969) else 0)

df['1970s'] = df['year'].map(lambda x: 1 if (x >= 1970 and x <= 1979) else 0)

df['1980s'] = df['year'].map(lambda x: 1 if (x >= 1980 and x <= 1989) else 0)

df['1990s'] = df['year'].map(lambda x: 1 if (x >= 1990 and x <= 1999) else 0)

df['2000s'] = df['year'].map(lambda x: 1 if (x >= 2000 and x <= 2009) else 0)

df['2010s'] = df['year'].map(lambda x: 1 if (x >= 2010 and x <= 2019) else 0)
# %%
df['day_of_week'].value_counts()
# %%
df['mondayRelease'] = df['day_of_week'].map(lambda x: 1 if (x == 1) else 0)

df['tuesdayRelease'] = df['day_of_week'].map(lambda x: 1 if (x == 2) else 0)

df['wednesdayRelease'] = df['day_of_week'].map(lambda x: 1 if (x == 3) else 0)

df['thursdayRelease'] = df['day_of_week'].map(lambda x: 1 if (x == 4) else 0)

df['fridayRelease'] = df['day_of_week'].map(lambda x: 1 if (x == 5) else 0)

df['saturdayRelease'] = df['day_of_week'].map(lambda x: 1 if (x == 6) else 0)

df['sundayRelease'] = df['day_of_week'].map(lambda x: 1 if (x == 7) else 0)
# %%
df['Winter'] = df.month.map(lambda x: 1 if (x == 12 or x <= 2) else 0)

df['Fall'] = df.month.map(lambda x: 1 if (x >= 9 and x <= 11) else 0)

df['Spring'] = df.month.map(lambda x: 1 if (x >= 3 and x <= 5) else 0)

df['Summer'] = df.month.map(lambda x: 1 if (x >= 6 and x <= 8) else 0)
# %%
train_features.append('1960s')
train_features.append('1970s')
train_features.append('1980s')
train_features.append('1990s')
train_features.append('2000s')
train_features.append('2010s')
train_features.append('mondayRelease')
train_features.append('tuesdayRelease')
train_features.append('wednesdayRelease')
train_features.append('thursdayRelease')
train_features.append('fridayRelease')
train_features.append('saturdayRelease')
train_features.append('sundayRelease')
train_features.append('Winter')
train_features.append('Fall')
train_features.append('Spring')
train_features.append('Summer')


# %% NEW
# Phân tích cột runtime
print(f"Missing values of runtime is {df['runtime'].isna().sum()}")

# %%
df['runtime'] = df['runtime'].fillna(df['runtime'].median())
# %%
f, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(x=df['runtime'], ax = ax[0], color='pink')
ax[0].set_title("Box Plot of runtime variable")
sns.distplot(a=df['runtime'], kde = False, ax = ax[1], color='pink', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of runtime variable")
f.tight_layout()
# %%
df.loc[df['runtime'].argmax(), ['title', 'runtime', 'revenue']]
#%%
df['runtime'].nlargest(5)
# %%
df = df.drop(df['runtime'].argmax())
# %%
print(f"Number of films have 0 runtime: {sum(df['runtime'] == 0)}")
# %%
median_runtime_higer_0 = df.loc[df['runtime'] > 0, 'runtime'].median()
df["runtime_processed"] = df["runtime"].mask(df["runtime"] == 0, median_runtime_higer_0)
# %%
f, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(x=df.runtime_processed, ax = ax[0], color='slateblue')
ax[0].set_title("Box Plot of runtime_processed variable")
sns.distplot(a=df.runtime_processed, kde = False, ax = ax[1], color='slateblue', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of runtime_processed variable")
f.tight_layout()
# %%
# calculate Pearson's correlation
corr, _ = pearsonr(df['runtime_processed'], df['revenue'])
print('Pearsons correlation: %.3f' % corr)
# %%

log_features.append('runtime_processed')



# %% NEW
# Phân tích cột spoken_language
df['spoken_languages'].head(10)

# %%
# new column for a count of the number of spoken languages
df['spoken_languages'] = df['spoken_languages'].fillna("")
df['spoken_languages_reform'] = df['spoken_languages'].apply(lambda x: re.findall("'name': \'(.+?)\'", x))
df['num_languages'] = df['spoken_languages_reform'].apply(lambda x: len(x))
print(df['num_languages'])

# %%
print(df['num_languages'].describe())
# %%
languagesDict = {}
for element in df["spoken_languages_reform"].values:
    for name in element:
        if name not in languagesDict:
            languagesDict[name] = 1
        else:
            languagesDict[name] += 1
            
sns.set(rc={'figure.figsize':(12,6)})
            
languages_df = pd.DataFrame.from_dict(languagesDict, orient='index', columns=["movies per spoken language"])
languages_df.sort_values(by="movies per spoken language", ascending=False).head(6).plot.bar(color='darkcyan')

languages_df.columns = ["number_of_languages"]
# %%
df['released_in_english'] = df['spoken_languages_reform'].apply(lambda x: 1 if 'English' in x else 0)

# %%
df['released_in_english'].value_counts()
#%%
lang = [df['released_in_english'].value_counts()[1], df['released_in_english'].value_counts()[0]]

labels = 'Spoken Language English', 'Non-English Spoken Language'
explode = (0.1, 0)

# plot
colors = ["red", "pink"]

fig, ix = plt.subplots(figsize=(10, 8))
ix.pie(lang, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'color':"w", 'fontsize':22, 'weight':"bold"})
ix.axis('equal')

ix.legend(labels,
          title="Legend",
          loc="center left",
          fontsize=14,
          bbox_to_anchor=(0.8, 0.25, 0.5, 1))
# %%
print(df['released_in_english'].value_counts())

labels = 'Non-English Spoken Language', 'Spoken Language English'

pal2 = ["darkcyan", "turquoise"]

fig, ix = plt.subplots(figsize=(12, 6))
sns.boxplot(x='released_in_english', y='revenue', data=df, palette=pal2)
plt.title('Films by Revenue Based on Spoken Language')
plt.ylabel("Revenue")
plt.xlabel("Language Status")

ix.set_xticklabels(labels)
# %%
corr, _ = pearsonr(df['released_in_english'], df['revenue'])
print('Pearsons correlation between released_in_english and revenue: %.3f' % corr)
# %%
corr, _ = pearsonr(df['num_languages'], df['revenue'])
print('Pearsons correlation between num_languages and revenue: %.3f' % corr)
# %%
train_features.append('released_in_english')
log_features.append('num_languages')




# %% NEW
# Phân tích cột status

print(df['status'].value_counts())

# %%
drop_cols.append('status')



# %% NEW
# Phân tích cột tagline
df['tagline'].head(5)
# %%
print(f"Missing values of tagline is {df['tagline'].isna().sum()}")
# %%
df['tagline'] = df['tagline'].astype('string')
df['tagline'] = df['tagline'].fillna('')
df['has_tagline'] = df['tagline'].apply(lambda x: 1 if x != "" else 0)
#%%
df['has_tagline'].value_counts()
# %%
labels = 'Has Tagline', 'Does not have Tagline'
btc = [df['has_tagline'].value_counts()[1], df['has_tagline'].value_counts()[0]]
explode = (0.1, 0)

colors = ['palevioletred', 'lightpink']

# plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.pie(btc, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'color':"w", 'fontsize': 22, 'weight':"bold"})
ax.axis('equal')

ax.legend(labels,
          title="Legend",
          loc="center left",
          fontsize=14,
          bbox_to_anchor=(0.8, 0.25, 0.5, 1))
# %%
pal = ['lightpink', 'palevioletred']

sns.set(rc={'figure.figsize':(8, 8)})
ax = sns.boxplot(x='has_tagline', y='revenue', data=df, palette=pal)
plt.title('Films by Revenue Based on Prescence of Tagline')
plt.ylabel("Revenue")
plt.xlabel("Tagline Exists")
labels = 'Does not have tagline', 'Has tagline'
ax.set_xticklabels(labels)
# %%
corr, _ = pearsonr(df['has_tagline'], df['revenue'])
print('Pearsons correlation between has_tagline and revenue: %.3f' % corr)
# %%
train_features.append('has_tagline')



# %% NEW
# Phân tích cột title
print(f"Missing values of title is {df['title'].isna().sum()}")
# %%
df['title'].fillna('')
df['title_len'] = df['title'].apply(lambda x: len(str(x)))
# %%
f, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(x=df.title_len, ax = ax[0], color='burlywood')
ax[0].set_title("Box Plot of title_len variable")
sns.distplot(a=df.title_len, kde = False, ax = ax[1], color='burlywood', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of title_len variable")
f.tight_layout()
# %%
corr, _ = pearsonr(df['title_len'], df['revenue'])
print('Pearsons correlation between title_len and revenue: %.3f' % corr)
# %%
log_features.append('title_len')



# %% NEW 
# Phân tích cột keywords
print(f"Missing values keywords is {df['Keywords'].isna().sum()}")
# %%
df['Keywords'] = df['Keywords'].astype('string')
df['Keywords'] = df['Keywords'].fillna('')
# %%
df['has_keywords'] = df['Keywords'].apply(lambda x: 1 if x != "" else 0)
# %%
labels = 'Has Keywords', 'Does not have Keywords'
btc = [sum(df.has_keywords), (df.has_keywords == 0).sum(axis=0)]
explode = (0.1, 0)

colors = ['red', 'pink']

# plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.pie(btc, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'color':"w", 'fontsize': 22, 'weight':"bold"})
ax.axis('equal') # ensures chart is a circle

ax.legend(labels,
          title="Legend",
          loc="center left",
          fontsize=14,
          bbox_to_anchor=(0.8, 0.25, 0.5, 1))
# %%
pal = ['red', 'pink']

sns.set(rc={'figure.figsize':(12, 6)})
ax = sns.boxplot(x='has_keywords', y='revenue', data=df, palette=pal)
plt.title('Films by Revenue Based on Keywords')
plt.ylabel("Revenue")
plt.xlabel("Keywords Exist")

labels = 'Does not have keywords', 'Has keywords'
ax.set_xticklabels(labels)
# %%
corr, _ = pearsonr(df['has_keywords'], df['revenue'])
print('Pearsons correlation between has_keywords and revenue: %.3f' % corr)
# %%
train_features.append('has_keywords')


# %% NEW
# Phân tích cột cast
df['cast'].head(5)

# %%
print(f"Missing values of cast is {df['cast'].isna().sum()}")
# %%
#pre-processing
df['cast'] = df['cast'].fillna("")
df['cast_processed'] = df['cast'].apply(lambda x: re.findall("'name': \'(.+?)\'", x))
#%%
df['cast_processed']

# %%
df['num_cast'] = df['cast_processed'].apply(lambda x: len(x))
df['num_cast'].value_counts()
#%%
df['num_cast'].nsmallest(5)
# %%
median_cast_higher_0 = df.loc[df['num_cast']> 0, 'num_cast'].median()
df["num_cast"] = df["num_cast"].mask(df["num_cast"] == 0, median_cast_higher_0)
# %%
f, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(x=df.num_cast, ax = ax[0], color='darkslateblue')
ax[0].set_title("Box Plot of num_cast variable")
sns.distplot(a=df.num_cast, kde = False, ax = ax[1], color='darkslateblue', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of num_cast variable")
f.tight_layout()
# %%
actorsDict = {}
for element in df[["revenue", "cast_processed"]].values:
    
    for actor in element[1]:
        if actor not in actorsDict:
            actorsDict[actor] = [element[0], 1]
        else:
            actorsDict[actor][0] += element[0]
            actorsDict[actor][1] += 1    

actorsDict = {k: v for k, v in actorsDict.items() if v[1] >= 5}

for actor in actorsDict:
    actorsDict[actor][0] = actorsDict[actor][0]/actorsDict[actor][1]
    actorsDict[actor] = actorsDict[actor][0]


    
    
actors_df = pd.DataFrame.from_dict(actorsDict, orient='index', columns=["mean_movies_revenue"])
actors_df.sort_values(by="mean_movies_revenue", ascending=False).head(20).plot.bar(color='pink')
# %%
#cast

actorsDict = {}
for element in df[["revenue", "cast_processed"]].values:
    temp = 5
    if len(element[1]) < temp:
        temp = len(element[1])
    for i in range(temp):
        if element[1][i] not in actorsDict:
            actorsDict[element[1][i]] = [element[0], 1]
        else:
            actorsDict[element[1][i]][0] += element[0]
            actorsDict[element[1][i]][1] += 1    

actorsDict = {k: v for k, v in actorsDict.items() if v[1] >= 5}

for actor in actorsDict:
    actorsDict[actor][0] = actorsDict[actor][0]/actorsDict[actor][1]
    actorsDict[actor] = actorsDict[actor][0]


    
    
actors_df = pd.DataFrame.from_dict(actorsDict, orient='index', columns=["mean_movies_revenue"])
actors_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='pink')
# %%
leadActorsDict = {}
for element in df[["revenue", "cast_processed"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in leadActorsDict:
        leadActorsDict[element[1][0]] = [element[0], 1]
    else:
        leadActorsDict[element[1][0]][0] += element[0]
        leadActorsDict[element[1][0]][1] += 1 

leadActorsDict = {k: v for k, v in leadActorsDict.items() if v[1] >= 5}

for actor in leadActorsDict:
    leadActorsDict[actor][0] = leadActorsDict[actor][0]/leadActorsDict[actor][1]
    leadActorsDict[actor] = leadActorsDict[actor][0]


    
    
actors_df = pd.DataFrame.from_dict(leadActorsDict, orient='index', columns=["mean_movies_revenue"])
actors_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='pink')
# %%
leadActorsDict = {k: v for k, v in sorted(leadActorsDict.items(), key=lambda item: item[1], reverse = True)}
actorsDict = {k: v for k, v in sorted(actorsDict.items(), key=lambda item: item[1], reverse = True)}

#%%
def checkLeadActor(cast):
    if len(cast) < 1 or cast[0] not in list(leadActorsDict)[:50]:
        return 0
    else:
        return 1
#%%
def checkTopActors(cast):
    if len(cast) < 1:
        return 0
    count = 0
    for actor in cast[:10]:
        if actor in list(actorsDict)[:100]:
            count += 1
    return count
#%%
def getActorRanks(cast):
    if len(cast) < 1:
        return len(actorsDict)
    rank = 0
    for a in cast[:5]:
        if a in list(actorsDict):
            rank += list(actorsDict.keys()).index(a)
    if rank == 0:
        rank = len(actorsDict)
    return rank / len(cast)
#%%
def getTopActorRank(cast):
    if len(cast) < 1:
        return len(leadActorsDict)
    if cast[0] in list(leadActorsDict):
        rank = list(leadActorsDict.keys()).index(cast[0])
    else:
        rank = len(leadActorsDict)
    return rank

# %%
df['topLeadActor'] = df['cast_processed'].apply(lambda x: checkLeadActor(x))
df['numTopActors'] = df['cast_processed'].apply(lambda x: checkTopActors(x))
df['actorRanks'] = df['cast_processed'].apply(lambda x: getActorRanks(x))
df['topActorRank'] = df['cast_processed'].apply(lambda x: getTopActorRank(x))
# %%
labels = 'Top Actor in Lead Role', 'Not in a Top Actor in Lead Role'
btc_1 = [sum(df.topLeadActor), (df.topLeadActor == 0).sum(axis=0)]
explode = (0.1, 0)

colors = ['darkslateblue', 'thistle']

fig, ax = plt.subplots()
plt.pie(btc_1, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'color':"w", 'fontsize': 22, 'weight':"bold"})
plt.axis('equal') # ensures chart is a circle

ax.legend(labels,
          title="Legend",
          loc="center left",
          fontsize=14,
          bbox_to_anchor=(0.8, 0.25, 0.5, 1))


plt.show()
# %%
sns.set(rc={'figure.figsize':(12,8)})
sns.countplot(x=df['numTopActors'], color='darkslateblue')
# %%
pal = ['red', 'pink']

ax = sns.boxplot(x='topLeadActor', y='revenue', data=df, palette=pal);
plt.title('Revenue for films with Keywords vs without Keywords');

labels = 'Has a Top Leading Actor', 'Does not have a Top Leading Actor'
ax.set_xticklabels(labels)
# %%
sns.boxplot(x='numTopActors', y='revenue', data=df, color='darkslateblue');
plt.title('Revenue for films based on the number of top actors');
# %%
corr, _ = pearsonr(df['num_cast'], df['revenue'])
print('Pearsons correlation between num_cast and revenue: %.3f' % corr)

corr, _ = pearsonr(df['topLeadActor'], df['revenue'])
print('Pearsons correlation between topLeadActor and revenue: %.3f' % corr)

corr, _ = pearsonr(df['numTopActors'], df['revenue'])
print('Pearsons correlation between numTopActors and revenue: %.3f' % corr)

corr, _ = pearsonr(df['actorRanks'], df['revenue'])
print('Pearsons correlation between actorRanks and revenue: %.3f' % corr)

corr, _ = pearsonr(df['topActorRank'], df['revenue'])
print('Pearsons correlation between topActorRank and revenue: %.3f' % corr)

# %%
log_features.append('num_cast')
train_features.append('topLeadActor')
log_features.append('numTopActors')
log_features.append('actorRanks')
log_features.append('topActorRank')



# %% NEW
# Phân tích cột crew

df['crew'].head(5)

# %%
# pre-processing
df['crew'] = df['crew'].fillna("")
df['crew_processed'] = df['crew'].apply(lambda x: re.findall("'name': \'(.+?)\'", x))
# %%
df['crew_processed'].head(10)
# %%
# new feature for a count of the number of crew
df['crew']= df['crew'].fillna("")
df["num_crew"] = df["crew"].str.count("'job")


f, ax = plt.subplots(2, figsize=(12,7))

# plot
sns.boxplot(x=df['num_crew'], ax = ax[0], color='pink')
ax[0].set_title("num_crew Boxplot")

sns.distplot(a=df['num_crew'], kde = False, ax = ax[1], color='pink', hist_kws=dict(alpha=1))
ax[1].set_title("num_crew Histogram")


f.tight_layout()
# %%
corr, _ = pearsonr(df['num_crew'], df['revenue'])
print('Pearsons correlation between num_crew and revenue: %.3f' % corr)
# %%
log_features.append('num_crew')
# %%
df["num_male_crew"] = df["crew"].str.count("'gender': 2")

f, ax = plt.subplots(2, figsize=(12,7))

df["num_male_crew"] = df["num_male_crew"].fillna(0)

# plot
sns.boxplot(x=df["num_male_crew"], ax = ax[0], color='red')
ax[0].set_title("num_male_crew Boxplot")

sns.distplot(a=df["num_male_crew"], kde = False, ax = ax[1], color='red', hist_kws=dict(alpha=1))
ax[1].set_title("num_male_crew Histogram")

f.tight_layout()
# %%
#analysis of correlation and create log feature
corr, _ = pearsonr(df['num_male_crew'], df['revenue'])
print('Pearsons correlation between num_male_crew and revenue: %.3f' % corr)

# %%
log_features.append('num_male_crew')
# %%
# female crew
df["num_female_crew"] = df["crew"].str.count("'gender': 1")

f, ax = plt.subplots(2, figsize=(12,7))

df["num_female_crew"] = df["num_female_crew"].fillna(0)

# plot
sns.boxplot(x=df["num_female_crew"], ax = ax[0], color='salmon')
ax[0].set_title("num_female_crew Boxplot")

sns.distplot(a=df["num_female_crew"], kde = False, ax = ax[1], color='salmon', hist_kws=dict(alpha=1))
ax[1].set_title("num_female_crew Histogram")


f.tight_layout()
# %%
corr, _ = pearsonr(df['num_female_crew'], df['revenue'])
print('Pearsons correlation between num_female_crew and revenue: %.3f' % corr)
# %%
log_features.append('num_female_crew')
# %%
# Directors
df['directors'] = df['crew'].apply(lambda x: re.findall("Director', 'name': '(.+?)'", x))

directorsDict = {}
for element in df[["revenue", "directors"]].values:

    for director in element[1]:
        if director not in directorsDict:
            directorsDict[director] = [element[0], 1]
        else:
            directorsDict[director][0] += element[0]
            directorsDict[director][1] += 1

directorsDict = {k: v for k, v in directorsDict.items() if v[1] >= 5}

for director in directorsDict:
    directorsDict[director][0] = directorsDict[director][0]/directorsDict[director][1]
    directorsDict[director] = directorsDict[director][0]

    
directors_df = pd.DataFrame.from_dict(directorsDict, orient='index', columns=["mean_movies_revenue"])
directors_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='pink')

directors_df.columns = ["number_of_directors"]
# %%
leadDirectorsDict = {}
for element in df[["revenue", "directors"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in leadDirectorsDict:
        leadDirectorsDict[element[1][0]] = [element[0], 1]
    else:
        leadDirectorsDict[element[1][0]][0] += element[0]
        leadDirectorsDict[element[1][0]][1] += 1    

leadDirectorsDict = {k: v for k, v in leadDirectorsDict.items() if v[1] >= 5}

for director in leadDirectorsDict:
    leadDirectorsDict[director][0] = leadDirectorsDict[director][0]/leadDirectorsDict[director][1]
    leadDirectorsDict[director] = leadDirectorsDict[director][0]


    
    
directors_df = pd.DataFrame.from_dict(leadDirectorsDict, orient='index', columns=["mean_movies_revenue"])
directors_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadDirectorsDict = {k: v for k, v in sorted(leadDirectorsDict.items(), key=lambda item: item[1], reverse = True)}
directorsDict = {k: v for k, v in sorted(directorsDict.items(), key=lambda item: item[1], reverse = True)}

#%%
def checkLeadDirector(crew):
    if len(crew) < 1 or crew[0] not in list(leadDirectorsDict)[:25]:
        return 0
    else:
        return 1
#%%
def checkTopDirectors(crew):
    if len(crew) < 1:
        return 0
    count = 0
    for director in crew[:5]:
        if director in list(directorsDict)[:100]:
            count += 1
    return count
#%%
def getDirectorRank(crew):
    if len(crew) < 1:
        return len(directorsDict)
    rank = 0
    for c in crew[:5]:
        if c in list(directorsDict):
            rank += list(directorsDict.keys()).index(c)
    if rank == 0:
        rank = len(directorsDict)
    return rank / len(crew)
# %%
df['topLeadDirector'] = df['directors'].apply(lambda x: checkLeadDirector(x))
df['numTopDirectors'] = df['directors'].apply(lambda x: checkTopDirectors(x))
df['directorsRank'] = df['directors'].apply(lambda x: getDirectorRank(x))

# %%
# Executive Producers
df['exec_producers'] = df['crew'].apply(lambda x: re.findall("Executive Producer', 'name': '(.+?)'", x))

exec_producersDict = {}
for element in df[["revenue", "exec_producers"]].values:

    for exec_producer in element[1]:
        if exec_producer not in exec_producersDict:
            exec_producersDict[exec_producer] = [element[0], 1]
        else:
            exec_producersDict[exec_producer][0] += element[0]
            exec_producersDict[exec_producer][1] += 1

exec_producersDict = {k: v for k, v in exec_producersDict.items() if v[1] >= 5}

for exec_producer in exec_producersDict:
    exec_producersDict[exec_producer][0] = exec_producersDict[exec_producer][0]/exec_producersDict[exec_producer][1]
    exec_producersDict[exec_producer] = exec_producersDict[exec_producer][0]




exec_producers_df = pd.DataFrame.from_dict(exec_producersDict, orient='index', columns=["mean_movies_revenue"])
exec_producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadExecProdDict = {}
for element in df[["revenue", "exec_producers"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in leadExecProdDict:
        leadExecProdDict[element[1][0]] = [element[0], 1]
    else:
        leadExecProdDict[element[1][0]][0] += element[0]
        leadExecProdDict[element[1][0]][1] += 1    

leadExecProdDict = {k: v for k, v in leadExecProdDict.items() if v[1] >= 5}

for exec_producer in leadExecProdDict:
    leadExecProdDict[exec_producer][0] = leadExecProdDict[exec_producer][0]/leadExecProdDict[exec_producer][1]
    leadExecProdDict[exec_producer] = leadExecProdDict[exec_producer][0]


    
    
exec_producers_df = pd.DataFrame.from_dict(leadExecProdDict, orient='index', columns=["mean_movies_revenue"])
exec_producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadExecProdDict = {k: v for k, v in sorted(leadExecProdDict.items(), key=lambda item: item[1], reverse = True)}
exec_producersDict = {k: v for k, v in sorted(exec_producersDict.items(), key=lambda item: item[1], reverse = True)}

#%%
def checkLeadExecProd(crew):
    if len(crew) < 1 or crew[0] not in list(leadExecProdDict)[:25]:
        return 0
    else:
        return 1
#%%
def checkTopExecProd(crew):
    if len(crew) < 1:
        return 0
    count = 0
    for director in crew[:5]:
        if director in list(exec_producersDict)[:100]:
            count += 1
    return count
#%%
def getExecProdRank(crew):
    if len(crew) < 1:
        return len(exec_producersDict)
    rank = 0
    for c in crew[:5]:
        if c in list(exec_producersDict):
            rank += list(exec_producersDict.keys()).index(c)
    if rank == 0:
        rank = len(exec_producersDict)
    return rank / len(crew)
# %%
df['topLeadExecProd'] = df['exec_producers'].apply(lambda x: checkLeadExecProd(x))
df['numTopExecProd'] = df['exec_producers'].apply(lambda x: checkTopExecProd(x))
df['execProdRank'] = df['exec_producers'].apply(lambda x: getExecProdRank(x))
# %%
# producers
df['producers'] = df['crew'].apply(lambda x: re.findall("Producer', 'name': '(.+?)'", x))

producersDict = {}
for element in df[["revenue", "producers"]].values:

    for producer in element[1]:
        if producer not in producersDict:
            producersDict[producer] = [element[0], 1]
        else:
            producersDict[producer][0] += element[0]
            producersDict[producer][1] += 1

producersDict = {k: v for k, v in producersDict.items() if v[1] >= 5}

for producer in producersDict:
    producersDict[producer][0] = producersDict[producer][0]/producersDict[producer][1]
    producersDict[producer] = producersDict[producer][0]




producers_df = pd.DataFrame.from_dict(producersDict, orient='index', columns=["mean_movies_revenue"])
producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadProducerDict = {}
for element in df[["revenue", "producers"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in leadProducerDict:
        leadProducerDict[element[1][0]] = [element[0], 1]
    else:
        leadProducerDict[element[1][0]][0] += element[0]
        leadProducerDict[element[1][0]][1] += 1    

leadProducerDict = {k: v for k, v in leadProducerDict.items() if v[1] >= 5}

for producer in leadProducerDict:
    leadProducerDict[producer][0] = leadProducerDict[producer][0]/leadProducerDict[producer][1]
    leadProducerDict[producer] = leadProducerDict[producer][0]


    
    
producers_df = pd.DataFrame.from_dict(leadProducerDict, orient='index', columns=["mean_movies_revenue"])
producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadProducerDict = {k: v for k, v in sorted(leadProducerDict.items(), key=lambda item: item[1], reverse = True)}
producersDict = {k: v for k, v in sorted(producersDict.items(), key=lambda item: item[1], reverse = True)}


def checkLeadProducer(crew):
    if len(crew) < 1 or crew[0] not in list(leadProducerDict)[:25]:
        return 0
    else:
        return 1

def checkTopProducers(crew):
    if len(crew) < 1:
        return 0
    count = 0
    for producer in crew[:5]:
        if producer in list(producersDict)[:100]:
            count += 1
    return count


def getProducerRank(crew):
    if len(crew) < 1:
        return len(producersDict)
    rank = 0
    for c in crew[:5]:
        if c in list(producersDict):
            rank += list(producersDict.keys()).index(c)
    if rank == 0:
        rank = len(producersDict)
    return rank / len(crew)
# %%
df['topLeadProducer'] = df['producers'].apply(lambda x: checkLeadProducer(x))

df['numTopProducers'] = df['producers'].apply(lambda x: checkTopProducers(x))

df['producersRank'] = df['producers'].apply(lambda x: getProducerRank(x))

# %%
# Composers
df['composers'] = df['crew'].apply(lambda x: re.findall("Composer', 'name': '(.+?)'", x))

composersDict = {}
for element in df[["revenue", "composers"]].values:

    for composer in element[1]:
        if composer not in composersDict:
            composersDict[composer] = [element[0], 1]
        else:
            composersDict[composer][0] += element[0]
            composersDict[composer][1] += 1

composersDict = {k: v for k, v in composersDict.items() if v[1] >= 5}

for composer in composersDict:
    composersDict[composer][0] = composersDict[composer][0]/composersDict[composer][1]
    composersDict[composer] = composersDict[composer][0]




composers_df = pd.DataFrame.from_dict(composersDict, orient='index', columns=["mean_movies_revenue"])
composers_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadComposerDict = {}
for element in df[["revenue", "composers"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in leadComposerDict:
        leadComposerDict[element[1][0]] = [element[0], 1]
    else:
        leadComposerDict[element[1][0]][0] += element[0]
        leadComposerDict[element[1][0]][1] += 1

leadComposerDict = {k: v for k, v in leadComposerDict.items() if v[1] >= 5}

for composer in leadComposerDict:
    leadComposerDict[composer][0] = leadComposerDict[composer][0]/leadComposerDict[composer][1]
    leadComposerDict[composer] = leadComposerDict[composer][0]


    
    
composers_df = pd.DataFrame.from_dict(leadComposerDict, orient='index', columns=["mean_movies_revenue"])
composers_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
# Sort both of our dicts
leadComposerDict = {k: v for k, v in sorted(leadComposerDict.items(), key=lambda item: item[1], reverse = True)}
composersDict = {k: v for k, v in sorted(composersDict.items(), key=lambda item: item[1], reverse = True)}


def checkLeadComposer(crew):
    if len(crew) < 1 or crew[0] not in list(leadComposerDict)[:25]:
        return 0
    else:
        return 1

def checkTopComposers(crew):
    if len(crew) < 1:
        return 0
    count = 0
    for composer in crew[:5]:
        if composer in list(composersDict)[:100]:
            count += 1
    return count

def getComposerRank(crew):
    if len(crew) < 1:
        return len(composersDict)
    rank = 0
    for c in crew[:5]:
        if c in list(composersDict):
            rank += list(composersDict.keys()).index(c)
    if rank == 0:
        rank = len(composersDict)
    return rank / len(crew)
# %%
df['topLeadComposer'] = df['composers'].apply(lambda x: checkLeadComposer(x))

df['numTopComposers'] = df['composers'].apply(lambda x: checkTopComposers(x))

df['composersRank'] = df['composers'].apply(lambda x: getComposerRank(x))
# %%
# Director of Photography
df['director_photos'] = df['crew'].apply(lambda x: re.findall("Director of Photography', 'name': '(.+?)'", x))

director_photosDict = {}
for element in df[["revenue", "director_photos"]].values:

    for director_photo in element[1]:
        if director_photo not in director_photosDict:
            director_photosDict[director_photo] = [element[0], 1]
        else:
            director_photosDict[director_photo][0] += element[0]
            director_photosDict[director_photo][1] += 1

director_photosDict = {k: v for k, v in director_photosDict.items() if v[1] >= 5}

for director_photo in director_photosDict:
    director_photosDict[director_photo][0] = director_photosDict[director_photo][0]/director_photosDict[director_photo][1]
    director_photosDict[director_photo] = director_photosDict[director_photo][0]




director_photos_df = pd.DataFrame.from_dict(director_photosDict, orient='index', columns=["mean_movies_revenue"])
director_photos_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadDirectorPhotoDict = {}
for element in df[["revenue", "director_photos"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in leadDirectorPhotoDict:
        leadDirectorPhotoDict[element[1][0]] = [element[0], 1]
    else:
        leadDirectorPhotoDict[element[1][0]][0] += element[0]
        leadDirectorPhotoDict[element[1][0]][1] += 1   

leadDirectorPhotoDict = {k: v for k, v in leadDirectorPhotoDict.items() if v[1] >= 5}

for director_photo in leadDirectorPhotoDict:
    leadDirectorPhotoDict[director_photo][0] = leadDirectorPhotoDict[director_photo][0]/leadDirectorPhotoDict[director_photo][1]
    leadDirectorPhotoDict[director_photo] = leadDirectorPhotoDict[director_photo][0]


    
    
director_photos_df = pd.DataFrame.from_dict(leadDirectorPhotoDict, orient='index', columns=["mean_movies_revenue"])
director_photos_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadDirectorPhotoDict = {k: v for k, v in sorted(leadDirectorPhotoDict.items(), key=lambda item: item[1], reverse = True)}
director_photosDict = {k: v for k, v in sorted(director_photosDict.items(), key=lambda item: item[1], reverse = True)}


def checkLeadDirectorPhoto(crew):
    if len(crew) < 1 or crew[0] not in list(leadDirectorPhotoDict)[:25]:
        return 0
    else:
        return 1

def checkTopDirectorsPhotos(crew):
    if len(crew) < 1:
        return 0
    count = 0
    for director in crew[:5]:
        if director in list(director_photosDict)[:100]:
            count += 1
    return count

def getDirectorsPhotosRank(crew):
    if len(crew) < 1:
        return len(director_photosDict)
    rank = 0
    for c in crew[:5]:
        if c in list(director_photosDict):
            rank += list(director_photosDict.keys()).index(c)
    if rank == 0:
        rank = len(director_photosDict)
    return rank / len(crew)
# %%

df['topLeadDirectorPhoto'] = df['director_photos'].apply(lambda x: checkLeadDirectorPhoto(x))
df['numTopDirectorsPhoto'] = df['director_photos'].apply(lambda x: checkTopDirectorsPhotos(x))

df['directorsPhotoRank'] = df['director_photos'].apply(lambda x: getDirectorsPhotosRank(x))
# %%
# editors
df['editors'] = df['crew'].apply(lambda x: re.findall("Editor', 'name': '(.+?)'", x))

editorsDict = {}
for element in df[["revenue", "editors"]].values:

    for editor in element[1]:
        if editor not in editorsDict:
            editorsDict[editor] = [element[0], 1]
        else:
            editorsDict[editor][0] += element[0]
            editorsDict[editor][1] += 1

editorsDict = {k: v for k, v in editorsDict.items() if v[1] >= 5}

for editor in editorsDict:
    editorsDict[editor][0] = editorsDict[editor][0]/editorsDict[editor][1]
    editorsDict[editor] = editorsDict[editor][0]




editors_df = pd.DataFrame.from_dict(editorsDict, orient='index', columns=["mean_movies_revenue"])
editors_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadEditorDict = {}
for element in df[["revenue", "editors"]].values:
    if len(element[1]) < 1:
        continue
    if element[1][0] not in leadEditorDict:
        leadEditorDict[element[1][0]] = [element[0], 1]
    else:
        leadEditorDict[element[1][0]][0] += element[0]
        leadEditorDict[element[1][0]][1] += 1       

leadEditorDict = {k: v for k, v in leadEditorDict.items() if v[1] >= 5}

for editor in leadEditorDict:
    leadEditorDict[editor][0] = leadEditorDict[editor][0]/leadEditorDict[editor][1]
    leadEditorDict[editor] = leadEditorDict[editor][0]


    
    
editors_df = pd.DataFrame.from_dict(leadEditorDict, orient='index', columns=["mean_movies_revenue"])
editors_df.sort_values(by="mean_movies_revenue", ascending=False).head(25).plot.bar(color='salmon')
# %%
leadEditorDict = {k: v for k, v in sorted(leadEditorDict.items(), key=lambda item: item[1], reverse = True)}
editorsDict = {k: v for k, v in sorted(editorsDict.items(), key=lambda item: item[1], reverse = True)}

def checkLeadEditor(crew):
    if len(crew) < 1 or crew[0] not in list(leadEditorDict)[:25]:
        return 0
    else:
        return 1

def checkTopEditors(crew):
    if len(crew) < 1:
        return 0
    count = 0
    for editor in crew[:5]:
        if editor in list(editorsDict)[:100]:
            count += 1
    return count

def getEditorsRank(crew):
    if len(crew) < 1:
        return len(editorsDict)
    rank = 0
    for c in crew[:5]:
        if c in list(editorsDict):
            rank += list(editorsDict.keys()).index(c)
    if rank == 0:
        rank = len(editorsDict)
    return rank / len(crew)
# %%

df['topLeadEditor'] = df['editors'].apply(lambda x: checkLeadEditor(x))

df['numTopEditors'] = df['editors'].apply(lambda x: checkTopEditors(x))

df['editorsRank'] = df['editors'].apply(lambda x: getEditorsRank(x))
# %%
train_features.append('topLeadDirector')
log_features.append('numTopDirectors')
log_features.append('directorsRank')

train_features.append('topLeadExecProd')
log_features.append('numTopExecProd')
log_features.append('execProdRank')

train_features.append('topLeadProducer')
log_features.append('numTopProducers')
log_features.append('producersRank')

train_features.append('topLeadComposer')
log_features.append('numTopComposers')
log_features.append('composersRank')

train_features.append('topLeadDirectorPhoto')
log_features.append('numTopDirectorsPhoto')
log_features.append('directorsPhotoRank')

train_features.append('topLeadEditor')
log_features.append('numTopEditors')
log_features.append('editorsRank')




# %% Feature Engineering
def getAvgStudioRev(movie):
    if movie.budget < 10000: 
        if len(movie.production_companies_reform) > 0:
            studios = movie.production_companies_reform
            median_revs = []
            for studio in studios:
                if studio in companies_df.index:
                     median_revs.append(float(companies_df.loc[studio]['median_rev']))
            if(len(median_revs) > 0) and mean(median_revs) > 10000:
                movie.budget_processed = mean(median_revs)
            else:
                movie.budget_processed = df.budget.median()
        else:
            movie.budget_processed = df.budget.median()
        
        
    
    if 'revenue' in movie and movie.revenue < 10000 and len(movie.production_companies_reform) > 0:
        studios = movie.production_companies_reform
        median_revs = []
        for studio in studios:
            if studio in companies_df.index:
                 median_revs.append(float(companies_df.loc[studio]['median_rev']))
        if(len(median_revs) > 0) and mean(median_revs) > 10000:
            movie.revenue = mean(median_revs)
        else:
            movie.revenue = df.revenue.median()
        return movie
    else:
        return movie
# %%
df = df.apply(getAvgStudioRev, axis=1)
# %%
f, ax = plt.subplots(2, figsize=(10,6))

sns.set(rc={'figure.figsize':(12,8)})
sns.distplot(a=df.revenue, kde = False, ax = ax[0], color='indianred', hist_kws=dict(alpha=1))
ax[0].set_title("Histogram of adjusted revenue variable")
sns.distplot(a=df.budget_processed, kde = False, ax = ax[1], color='mediumaquamarine', hist_kws=dict(alpha=1))
ax[1].set_title("Histogram of adjusted budget_processed variable")
f.tight_layout()

# %%
df['budget_to_year_ratio'] = df['budget_processed'] / (df['year'] * df['year'])
# %%
df['runtime_to_year_ratio'] = df['runtime'] / (df['year'])
# %%
log_features.append('budget_to_year_ratio')
log_features.append('runtime_to_year_ratio')
# %%
len(df.columns)
# %%
print(drop_cols)
# %%
print(train_features)
# %%
len(train_features)
print(len(train_features))
# %%
print(log_features)
print(len(log_features))



# DONE PREPROCESSING-DATA
#%%
# STEP [4]================================== Prepare The Data For Machine Learning Algorithms ================================================
# %%
for feat in log_features:
    df["log_" + feat] = np.log1p(df[feat])
    if feat != "revenue":
        train_features.append("log_" + feat)
# %%
print(train_features)
print()
print("Number of features: ", len(train_features))
# %%
sns.set(rc={'figure.figsize':(30,35)})

corr = df[['log_revenue',*log_features]].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm')

plt.title("Correlation between numerical features")
#%%
corr = df[['log_revenue', *train_features]].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm')

plt.title("Correlation between numerical features")

#%%
create_clear_datasets = True
if create_clear_datasets:
    temp_df = pd.DataFrame()
    temp_df = df[train_features]
    temp_df['log_revenue'] = df['log_revenue']
    pd.DataFrame.to_csv(temp_df, '../datasets/films_revenue_dataset_clear.csv')
# %% MODELING
# STEP [5]================================== Train and Evaluate Models ================================================
X = np.array(df[train_features])
y = np.array(df['log_revenue'])
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)
# %%
def get_evaluate(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    errors = abs(y_pred - y_test)
    mae = np.mean(errors)
    mape = 100 * (errors / y_test)
    print('Mean Squared Error: ', round(mse, 4))
    print('Root Mean Squared Error: ', round(rmse, 4))
    print('Mean Absolute Error: ', round(mae, 4))
    test_accuracy = 100 - np.mean(mape)
    print('Test Set Accuracy (from Mean Absolute Percentage Error):{:.3f}%'.format(test_accuracy))


#%%
# Try Random Forest Regression
new_run = True
if new_run:
    rfr_reg = RandomForestRegressor(n_estimators=120, max_depth=60)
    rfr_reg.fit(X_train, y_train)
    joblib.dump(rfr_reg, '../models/rfr_reg')
else:
    rfr_reg = joblib.load('../models/rfr_reg')

#%%
# Try Extra-Trees Regression
new_run = True
if new_run:
    extra_trees_reg = ExtraTreesRegressor(n_estimators=50, random_state=41)
    extra_trees_reg.fit(X_train, y_train)
    joblib.dump(extra_trees_reg, '../models/extra_trees_reg')
else:
    extra_trees_reg = joblib.load('../models/extra_trees_reg')
# %%
# Try Extreme Gradient Boost
new_run=True
if new_run:
    xgb_reg = XGBRegressor()
    xgb_reg.fit(X_train, y_train)
    joblib.dump(xgb_reg, '../models/xgb_reg')
else:
    xgb_reg = joblib.load('../models/xgb_reg')
# %%
# Try KNN Regression
new_run = True
if new_run:
    knn_reg = KNeighborsRegressor(n_neighbors=4, weights='distance')
    knn_reg.fit(X_train, y_train)
    joblib.dump(knn_reg, '../models/knn_reg')
else:
    knn_reg = joblib.load('../models/knn_reg')
# %%
# Try Light Gradient Boost Regression
new_run = True
if new_run:
    lgb_reg = LGBMRegressor()
    lgb_reg.fit(X_train, y_train)
    joblib.dump(lgb_reg, '../models/lgb_reg')
else:
    lgb_reg = joblib.load('../models/lgb_reg')
# %%
# Try Decision Tree Regression
new_run = True
if new_run:
    decision_tree_reg = DecisionTreeRegressor(random_state=41, max_depth=50)
    decision_tree_reg.fit(X_train, y_train)
    joblib.dump(decision_tree_reg, '../models/decision_tree_reg')
else:
    decision_tree_reg = joblib.load('../models/decision_tree_reg')


#%%
# Evaluate Models
#%%
print("Random Forest: \n")
get_evaluate(y_train, rfr_reg.predict(X_train))
print("====================")
#%%
print("Extra-Trees: \n")
get_evaluate(y_train, extra_trees_reg.predict(X_train))
print("====================")
#%%
print("Extreme Gradient Boost: \n")
get_evaluate(y_train, xgb_reg.predict(X_train))
print("====================")
#%%
print("KNN Regression: \n")
get_evaluate(y_train, knn_reg.predict(X_train))
print("====================")
#%%
print("Light Gradient Boost: \n")
get_evaluate(y_train, lgb_reg.predict(X_train))
print("====================")
#%%
print("Decision Tree: \n")
get_evaluate(y_train, decision_tree_reg.predict(X_train))



# %%
# STEP [6]================================== Fine-tune Your Model ================================================
#%%
# Random Forest
param_grid_rfr = {
    'n_estimators': [50, 120, 200],
    'max_depth': [50, 120, 20]
}
rfr_reg_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=41), param_grid=param_grid_rfr)
rfr_reg_grid.fit(X_train, y_train)
print(f"Best params: \n{rfr_reg_grid.best_params_}")
get_evaluate(rfr_reg_grid.predict(X_train), y_train)
#%%
# Extra-Trees
param_grid_extree = {
    'n_estimators': [50, 120, 150],
    'max_depth': [20,120 , 150, 200]
}
extrees_reg = ExtraTreesRegressor(random_state=41)
extrees_reg_grid = GridSearchCV(estimator=extrees_reg, param_grid=param_grid_extree)
extrees_reg_grid.fit(X_train, y_train)
print(f"Best params: \n{extrees_reg_grid.best_params_}")
get_evaluate(extrees_reg_grid.predict(X_train), y_train)
#%%
# Decision Tree
param_grid_decision_tree = {
    'max_depth': [10, 15, 20, 50, 100, 1000]
}
decision_tree_reg_grid = GridSearchCV(estimator=DecisionTreeRegressor(random_state=41), param_grid=param_grid_decision_tree)
decision_tree_reg_grid.fit(X_train, y_train)
print(f"Best params: \n{decision_tree_reg_grid.best_params_}")
get_evaluate(decision_tree_reg_grid.predict(X_train), y_train)



# %%
# STEP [7]================================== Analyze And Test Your Solution ================================================
#%%
# Voting Regression with Random Forest, Extra-Trees and Light GBM
voting_reg = VotingRegressor(
    estimators=[('random-forest',RandomForestRegressor(max_depth=20, n_estimators=200 ,random_state=41)),
    ('extra-trees',ExtraTreesRegressor(max_depth=20, n_estimators=150,random_state=41)),
    ('light GBM',LGBMRegressor(random_state=41))]
)
voting_reg.fit(X_train, y_train)
voting_pred = voting_reg.predict(X_train)
get_evaluate(voting_pred, y_train)

#%%
# Voting Regression with Random Forest, Light GBM, XGB
voting_reg2 = VotingRegressor(
    estimators=[('random-forest',RandomForestRegressor(max_depth=20, n_estimators=200 ,random_state=41)),
    ('xgb',XGBRegressor()),
    ('light GBM',LGBMRegressor(random_state=41))]
)
voting_reg2.fit(X_train, y_train)
voting_pred = voting_reg.predict(X_train)
get_evaluate(voting_pred, y_train)
# %%
# Bagging with 10 Light GBM
bagging_reg = BaggingRegressor(
    base_estimator=LGBMRegressor(random_state=41),
    n_estimators=10
)
bagging_reg.fit(X_train, y_train)
bagging_pred = bagging_reg.predict(X_train)
get_evaluate(bagging_pred, y_train)
# %%
# Stacking with Extra-Trees and Random Forest, final is Light GBM
stack_reg = StackingRegressor(
    estimators=[('extra-trees',ExtraTreesRegressor(max_depth=20, n_estimators=150, random_state=41)),
    ('random-forest',RandomForestRegressor(max_depth=20, n_estimators=200 ,random_state=41))],
    final_estimator=LGBMRegressor(random_state=41)
)
stack_reg.fit(X_train, y_train)
stack_pred = stack_reg.predict(X_train)
get_evaluate(stack_pred, y_train)
# %%
# AdaBoost
ada_reg = AdaBoostRegressor(
    base_estimator=LGBMRegressor(random_state=41),
    n_estimators=7
)
ada_reg.fit(X_train, y_train)
ada_pred = ada_reg.predict(X_train)
get_evaluate(ada_pred, y_train)


#%%
# ============ Test Models on Test Set ===================
#%%
# ========= Base Models ===============
#%%
# Random Forest
get_evaluate(y_test, rfr_reg.predict(X_test))
# %%
# Extra-Trees
get_evaluate(y_test, extra_trees_reg.predict(X_test))
# %%
# XGB
get_evaluate(y_test, xgb_reg.predict(X_test))
# %%
# Light GBM
get_evaluate(y_test, lgb_reg.predict(X_test))
# %%
# KNN
get_evaluate(y_test, knn_reg.predict(X_test))
# %%
# Decision Tree
get_evaluate(y_test, decision_tree_reg.predict(X_test))

#%%
# ================== Tuned Models ================
# %%
# Random Forest tuned
get_evaluate(y_test, rfr_reg_grid.predict(X_test))
# %%
# Extra-Trees tuned
get_evaluate(y_test, extrees_reg_grid.predict(X_test))
# %%
# Decision Tree tuned
get_evaluate(y_test, decision_tree_reg_grid.predict(X_test))
# %%
# ================ Ensemble Models ===================
#%%
# Voting Regression with Random Forest, Extra-Trees and Light GBM
get_evaluate(y_test, voting_reg.predict(X_test))
# %%
# Voting Regression with Random Forest, Light GBM, XGB
get_evaluate(y_test, voting_reg2.predict(X_test))
# %%
# Bagging with 10 Light GBM
get_evaluate(y_test, bagging_reg.predict(X_test))
# %%
# AdaBoost
get_evaluate(y_test, ada_reg.predict(X_test))
# %%
# Stacking with Extra-Trees and Random Forest, final is Light GBM
get_evaluate(y_test, stack_reg.predict(X_test))


#%%
# =========== Choose Model ===================
# Best Model: Voting Regression with Random Forest, Extra-Trees and Light GBM
# Code:
''' voting_reg = VotingRegressor(
    estimators=[('random-forest',RandomForestRegressor(max_depth=20, n_estimators=200 ,random_state=41)),
    ('extra-trees',ExtraTreesRegressor(max_depth=20, n_estimators=150,random_state=41)),
    ('light GBM',LGBMRegressor(random_state=41))]
) '''
# %%
# STEP [8]================================== Launch, monotor and maintain your system ================================================




#%%

# -----------------------------------------------------DONE--------------------------------------------------------------------





# DATASET LINK:
# https://www.kaggle.com/c/tmdb-box-office-prediction


# REFERECE LINK:
# [1]: https://github.com/nickmitch21/TMDB-Box-Office-Prediction
# [2]: https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
# [3]: scikit-learn.org
# %%
