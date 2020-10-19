#
# A better data cleaning function
#

import pandas as pd
import numpy as np

def clean_data(data):
    age_range = [(17,100), [0,1]]
    price_idx_range = [(90,120), (0,1)]
    conf_idx_range = [(-50,0), (0,1)]
    euribor_range = [(0,6), (0,1)]
    nr_employed_range = [(4500,5500), (0,1)]
    
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    # x_df = data.copy()
    
    # call duration is not known at the start of a call therefore should not be part of the model
    # the 'month' feature should be discarded as data is over multiple years and socioeconomic features correlate with month
    x_df.drop(columns=['duration','month'], inplace=True)
    # age
    x_df['age'] = np.interp(x_df.age, *age_range)
    # job
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df = x_df.join(jobs)
    x_df.drop(["job","job_unknown"], inplace=True, axis=1)
    # marital
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    # education
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df = x_df.join(education)
    x_df.drop(["education","education_unknown"], inplace=True, axis=1)
    # default
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    # housing
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    # loan
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    # contact
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df = x_df.join(contact)
    x_df.drop(["contact","contact_telephone"], inplace=True, axis=1)
    # day_of_week
    # *very* slightly higher percentage of calls are successful mid-week (12% vs 10-11% on mon/fri)
    days = pd.get_dummies(x_df.day_of_week.apply(lambda s: 'mid-week' if s in ['tue','wed','thu'] else 'not'), prefix='day')
    x_df = x_df.join(days)
    x_df.drop(["day_of_week","day_not"], inplace=True, axis=1)
    # campaign
    campaign = pd.get_dummies(pd.cut(x_df.campaign,[1,2,3,5,10,1e4]), prefix='campaign')
    x_df = x_df.join(campaign)
    x_df.drop(["campaign","campaign_(10.0, 10000.0]"], inplace=True, axis=1)
    # pdays
    dayvalues = [0,2,3,5,7,14,28,999]
    daylabels = [f'{x}days' if x<999 else 'never' for x in dayvalues[1:]]
    pdays = pd.get_dummies(pd.cut(x_df.pdays, dayvalues, labels=daylabels), prefix='pdays')
    x_df = x_df.join(pdays)
    x_df.drop(["pdays","pdays_never"], inplace=True, axis=1)
    # previous
    x_df['previous'] = x_df.previous.apply(lambda s: 1 if s > 0 else 0)
    # poutcome (previous outcome)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)
    # emp.var.rate
    x_df['emp.var.rate'] = x_df['emp.var.rate'].apply(lambda s: 1 if s < -1 else 0)
    # cons.price.idx
    x_df['cons.price.idx'] = np.interp(x_df['cons.price.idx'], *price_idx_range)
    # cons.conf.idx
    x_df['cons.conf.idx'] = np.interp(x_df['cons.conf.idx'], *conf_idx_range)
    # euribor3m
    x_df['euribor3m'] = np.interp(x_df['euribor3m'], *euribor_range)
    # nr.employed
    empvalues = [0,5050,5150,5250]
    emplabels = [str(x) for x in empvalues[1:]]
    nr_emp = pd.get_dummies(pd.cut(x_df['nr.employed'], empvalues, labels=emplabels), prefix='empl')
    x_df = x_df.join(nr_emp)
    x_df.drop(["nr.employed","empl_5050"], inplace=True, axis=1)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    return x_df, y_df
