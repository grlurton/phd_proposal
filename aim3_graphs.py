import pandas as pd
import datetime
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

store = pd.HDFStore('../orbf_data_quality/data/processed/orbf_benin.h5')
data = store['data']
store.close()


def make_timeline_data(data):
    out = len(data)
    data['correction'] = (data.indicator_claimed_value ==
                          data.indicator_verified_value)

    perc_right = sum(data['correction']) / len(data)

    return pd.DataFrame([[out, perc_right]])

timeline_data = data.groupby(['entity_name', 'entity_type', 'period']).apply(
    make_timeline_data).reset_index()
timeline_data.columns = ['entity_name', 'entity_type',
                         'date', 'level_1', 'value', 'percent_right']


def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_palette(get_colors())


def get_colors():
    return np.array([
        [0.078, 0.337, 0.396],            # Blue
        [0.639, 0.176, 0.114],            # Red
        [0.086, 0.478, 0.188],            # Green
        [0.639, 0.376, 0.114],            # Brun
        [0.984375, 0.7265625, 0],       # Yellow
    ])


def set_size(fig):
    fig.set_size_inches(6, 3)
    plt.tight_layout()

sns.palplot(get_colors())

set_style()
sns.set_context("paper")
plt.figure(figsize=(6, 3))
ax = sns.distplot(timeline_data.percent_right[
                  timeline_data.percent_right < 1], bins=12)
ax.set(xlabel='% of correct indicators by report')
sns.plt.xlim(0, 1.1)

ax.figure.savefig("figure/correct_dist.pdf", dpi=1200)

# Dollar Value Correction


def correct_amount(data):
    data.indicator_claimed_value = sum(data.indicator_claimed_value)
    data.difference_indicateur = sum(
        data.indicator_verified_value - data.indicator_claimed_value)
    data.claimed_montant = sum(
        data.indicator_claimed_value * data.indicator_tarif)
    data.difference_montant = sum(
        data.indicator_montant - data.claimed_montant)
    return pd.DataFrame([[data.claimed_montant.tolist(),
                          data.difference_montant.tolist(),
                          data.difference_indicateur.tolist(),
                          data.indicator_claimed_value.tolist()[0],
                          data.indicator_verified_value.tolist()[0],
                          data.indicator_tarif.tolist()[0]]])


verif_data_report = data.groupby(
    ['entity_name', 'indicator_label', 'date']).apply(correct_amount)
verif_data_report = verif_data_report.reset_index()
verif_data_report.columns = ['entity_name', 'indicator_label', 'date', 'level_3', 'claimed_montant',
                             'difference_montant', 'difference_indicateur', 'claimed_indicateur', 'indicator_verified_value', 'indicator_tarif']

verif_data_report['perc_variation_montant'] = verif_data_report[
    'difference_montant'] / verif_data_report['claimed_montant']
verif_data_report['perc_variation_indicateur'] = verif_data_report[
    'difference_indicateur'] / (verif_data_report['claimed_indicateur'] + 0.2)
verif_data_report_plot = verif_data_report[(np.abs(verif_data_report.difference_montant) < 10000000000) &
                                           ~(verif_data_report.perc_variation_montant == np.inf) & ~(verif_data_report.perc_variation_indicateur == np.inf) & (np.abs(verif_data_report.difference_montant) < 10000000000) &
                                           (verif_data_report.indicator_verified_value > 0) &
                                           (verif_data_report.indicator_tarif > 0) &
                                           ~(verif_data_report.perc_variation_indicateur == 0)]

#verif_data_report_plot[verif_data_report_plot.indicator_tarif == 330]
verif_data_report_plot.indicator_tarif = verif_data_report_plot.indicator_tarif.astype('str').astype(
    'category').cat.set_categories(['330.0', '420.0', '3000.0', '5890.0', '15000.0', '27000.0'], ordered=True)


set_style()
sns.set_context("paper")
plt.figure(figsize=(20, 6))
g = sns.FacetGrid(verif_data_report_plot, col="indicator_label", col_wrap=3,
                  sharex=True, sharey=False, size=4)
g.map(sns.distplot,  "perc_variation_indicateur", hist=False, rug=True)
g.add_legend()
g.set_titles('')

ax.set(xlabel='Correction of payment after data validation')
ax.figure.savefig("figure/payment_correction.pdf", dpi=1200)

# Facility Examples
sns.despine(left=True)

verif_data_report_plot2 = verif_data_report_plot[verif_data_report_plot.entity_name.isin(
    ['Csa Dekanme Cs', 'Tohoue Csc', 'Bembe Di', 'Zounta Mi'])]

set_style()
sns.set_context("paper")
plt.figure(figsize=(6, 3))
g = sns.FacetGrid(verif_data_report_plot2, col="entity_name",
                  col_wrap=2, size=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "difference_montant")
g.fig.savefig("figure/facility_correct.pdf", dpi=1200)
