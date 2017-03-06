import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

age_dist_census = pd.read_csv(
    '../niger_election_data/data/external/tabula-ETAT_STRUCTURE_POPULATION.csv')
age_dist_census = age_dist_census[~(age_dist_census.AGE == 'Total')]
age_dist_census = age_dist_census[~(age_dist_census.Region == 'Niger')]

age_dist_census_adults = age_dist_census[age_dist_census.AGE.astype(int) > 17]


voters_data = pd.read_csv(
    '../niger_election_data/data/processed/voters_list.csv', encoding="ISO-8859-1")
voters_data.region = voters_data.region.str.title()

voters_data.age = np.round(voters_data.age)
print(len(voters_data))
voters_data = voters_data[(voters_data.age < 99) & (voters_data.age > 17)]
voters_data = voters_data[~(voters_data.region == 'Diaspora')]
print(len(voters_data))

voters_data.region[voters_data.region == 'Tillaberi'] = 'Tillabery'


def census_sum_total(data_census):
    region = data_census.Region.unique()[0]
    n_total = sum(data_census.Total)
    out = data_census['Total'] / n_total
    age_str = data_census['AGE'].astype(str) + ' Ans'
    return pd.DataFrame({'region': region, 'percentage': out.tolist(), 'Age': age_str})

by_region_census_total = age_dist_census_adults.groupby(
    ['Region']).apply(census_sum_total)


def voters_age_distrib(data):
    """
    Function to get the distribution of voters by age in a dataset. Age is censored at 100.
    """
    data.age_str = data.age.astype(int).astype(str) + ' Ans'
    out = data.age_str.value_counts() / len(data)
    out = out.reset_index()
    out.columns = ['age', 'percentage']
    return out

by_region_voters_total = voters_data.groupby(
    'region').apply(voters_age_distrib).reset_index()
by_region_voters_total['source'] = 'Voters'
by_region_census_total['source'] = 'Census'

by_region_voters_total = by_region_voters_total.drop('level_1', 1)

by_region_voters_total.columns = ['region', 'age', 'percentage', 'source']
by_region_census_total.columns = ['age', 'percentage', 'region', 'source']

data = by_region_voters_total.append(by_region_census_total)

data = data.reset_index()

data['age_num'] = 0
for i in range(len(data)):
    data.loc[i, 'age_num'] = data.iloc[i]['age'][0:2]

data['age_num'] = data['age_num'].astype(int)

def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_palette(get_colors())

def get_colors():
    return np.array([
        [0.7 , 0.7 , 0.7] ,                 # Light Grey
        [0.086, 0.478, 0.188],              # Green
        [0.984375, 0.7265625, 0],           # Yellow
        [0.078, 0.337, 0.396],              # Blue
        [0.639, 0.176, 0.114],              # Red
        [0.639, 0.376, 0.114],              # Brun
    ])

def set_size(fig):
    fig.set_size_inches(6, 3)
    plt.tight_layout()

sns.palplot(get_colors())

set_style()
sns.set_context("paper", font_scale=1.5)
plt.figure(figsize=(6, 3))
kws = dict(s=20, linewidth=.5)
g = sns.FacetGrid(data, col="region", col_wrap=4, hue="source", ylim=(-0.001, 0.08) , hue_kws=dict(marker=["^", "v"])
            ).map(plt.scatter,  'age_num',  'percentage', **kws).set(xlabel='Age', ylabel = 'Age Distribution').add_legend()

g.savefig('figure/age_structure_comparison.pdf', dpi=1200)
