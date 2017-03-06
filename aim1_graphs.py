import pyodbc
import pandas as pd
import datetime
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; '
    r'DBQ=J:\LIMITED_USE\PROJECT_FOLDERS\KEN\ART_ABCE\10890 Nyahururu\CCC Patient Application Database - Anonymized.mdb;'
    )
connexion = pyodbc.connect(conn_str)

visits_to_keep = ['visit_id' , 'patient_id' , 'visit_date' , 'weight' , 'weight_not_taken' , 'height' , 'height_not_taken' , 'cotrim' , 'fluc' , 'art_regimen' , 'cd4' , 'cd4date' , 'cd4_result' , 'hb' , 'hb_result' , 'next_visit_date' , 'WHOstage' , 'ViralLoad' , 'ViralLoad_result' , 'DateEntered']

sql = 'select ' + ' , '.join(visits_to_keep) + ' from tblvisit_information'

visits_1 = pd.read_sql(sql , connexion)

patients_to_keep = ['patient_id' , 'dob' , 'age' , 'sex' , 'marital_status' , 'patient_source' , 'artstart_date' , 'previous_arv' , 'hiv_pos_date' , 'hiv_care_date' , 'who_stage' , 'art_eligibility_date' , 'date_entered' , 'DateEntered']

sql = 'select ' + ' , '.join(patients_to_keep) + ' from tblpatient_information'

patients_1 = pd.read_sql(sql , connexion)

visits_to_keep = [item.lower() for item in visits_to_keep]
patients_to_keep = [item.lower() for item in patients_to_keep]

visits_1.columns = visits_to_keep
patients_1.columns = patients_to_keep

### Facility 2
visits_2 = pd.read_csv("J:\\LIMITED_USE\\PROJECT_FOLDERS\\KEN\\ART_ABCE\\11936 Athi River\\11936_athi_visit.csv"  , encoding = "ISO-8859-1")
patients_2 = pd.read_csv("J:\\LIMITED_USE\\PROJECT_FOLDERS\\KEN\\ART_ABCE\\11936 Athi River\\11936_athi_patient.csv"  , encoding = "ISO-8859-1")
visits_2 = visits_2[visits_to_keep]
patients_2 = patients_2[patients_to_keep]

### Facility 4
visits_3 = pd.read_csv("J:\\LIMITED_USE\\PROJECT_FOLDERS\\KEN\\ART_ABCE\\12626 Mwingi Dist Hosp\\12626_mwingi_visit.csv"  , encoding = "ISO-8859-1")
patients_3 = pd.read_csv("J:\\LIMITED_USE\\PROJECT_FOLDERS\\KEN\\ART_ABCE\\12626 Mwingi Dist Hosp\\12626_mwingi_patient.csv"  , encoding = "ISO-8859-1")
visits_3 = visits_3[visits_to_keep]
patients_3 = patients_3[patients_to_keep]

visits_1.dateentered = pd.to_datetime(visits_1.dateentered)
visits_1.visit_date[visits_1.visit_date == datetime.datetime(2111, 3, 7, 0, 0)] = datetime.datetime(2011, 3, 7, 0, 0)

visits_1.visit_date = pd.to_datetime(visits_1.visit_date , errors='coerce')




visits_1.next_visit_date = pd.to_datetime(visits_1.next_visit_date , errors='coerce')

#################################################################################################################
def make_patient_data(data):
    data['first_visit'] = min(data.visit_date)
    data['time_follow_up'] = data.visit_date - data.first_visit
    data = data.sort('visit_date' , ascending = True)
    data['last_given_appointment'] = [pd.NaT] + data.next_visit_date[1:len(data)].tolist()
    return data

visits_1 = visits_1.groupby('patient_id').apply(make_patient_data)

#################################################################################################################
import numpy as np
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)
%matplotlib inline


visits_1['visit_entry'] = visits_1.dateentered -visits_1.visit_date
visits_1['visit_entry'] = visits_1['visit_entry'].astype('timedelta64[D]').astype(int , raise_on_error=False)
visit_entry_toplot = visits_1['visit_entry']
visit_entry_toplot = visit_entry_toplot[~(pd.isnull(visit_entry_toplot)) & (visit_entry_toplot < 365) & (visit_entry_toplot >= 0)]

import matplotlib
import matplotlib.pyplot as plt
def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_palette(get_colors())

def get_colors():
    return np.array([
        [0.639,0.176,0.114],            # Red
        [0.078,0.337,0.396],            # Blue
        [0.086,0.478,0.188],            # Green
        [0.639,0.376,0.114],            # Brun
        [0.984375, 0.7265625, 0],       # Yellow
    ])

def set_size(fig):
    fig.set_size_inches(6, 3)
    plt.tight_layout()

sns.palplot(get_colors())

set_style()
sns.set_context("paper")
plt.figure(figsize=(6, 3))
ax = sns.distplot(visit_entry_toplot , kde = False )
ax.set(xlabel='Days between visit and data entry')
sns.plt.xlim(0,)
ax.figure.savefig("figure/data_entry_time.pdf", dpi=1200)

visits_1['time_schedule']  = visits_1.next_visit_date -visits_1.visit_date
visits_1['time_schedule'] = visits_1['time_schedule'].astype('timedelta64[D]').astype(int , raise_on_error=False)
visit_schedule_toplot = visits_1['time_schedule']
visit_schedule_toplot  = visit_schedule_toplot[(~pd.isnull(visit_schedule_toplot)) & (visit_schedule_toplot < 150) & (visit_schedule_toplot > 0)]

plt.figure(figsize=(6, 3))
ax = sns.distplot(visit_schedule_toplot , kde = False )
ax.set(xlabel='Days between visit and next scheduled appointment')
sns.plt.xlim(0,)
ax.figure.savefig("figure/time_to_appointment.pdf", dpi=1200)

late_to_appointment = (visits_1.last_given_appointment-visits_1.visit_date).astype('timedelta64[D]').astype(int , raise_on_error=False)
late_to_appointment = late_to_appointment[(~pd.isnull(late_to_appointment)) & (np.abs(late_to_appointment) < 250)] % 28

plt.figure(figsize=(6, 3))
ax = sns.distplot(late_to_appointment , kde = False)
ax.set(xlabel='Days late to appointment')
sns.plt.xlim(0,)
ax.figure.savefig("figure/time_late_to_appointment.pdf", dpi=1200)
