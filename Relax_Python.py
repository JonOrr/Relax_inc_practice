# -*- coding: utf-8 -*-
"""
Spyder Editor

Relax inc Take-Home

We define an adopted user as one who has logged into the product on three 
separate days in at least one seven-day period. We want to identify which 
factors predict future user adoption


By: Jonathan Orr
"""
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up filenames
engagement_file = 'takehome_user_engagement.csv'
users_file = 'takehome_users.csv'

# Read in the csv files as dataframes
e_df = pd.read_csv(engagement_file, encoding='latin-1', header = 0)
u_df = pd.read_csv(users_file, encoding='latin-1', header = 0)


# Maximum user id is 12000
unique_users = e_df.max(axis = 1).max() #12000


# For each number in range(1, 12001) 
#   If there are no results for the user_id, add the id to no_logins list
#   Return all times where the user visited
#   Check if any are three times within the same week
#   If so add the user id to the adopted users list
#   If not add them to the non_adopted users list

no_login_users = []
has_login_users = []
adopted_users = []
non_adopted_users = []
classification = []

e_id_set = set(e_df['user_id'])

for i in range(1, 12001):
    print('Currently working on user_id:', i) # This line is for debugging purposes
    if i in e_id_set:
        has_login_users.append(i)
        # cu_df is current user df
        cu_df = e_df.loc[e_df['user_id'] == i]
        cu_dates = cu_df['time_stamp']
        df = pd.to_datetime(cu_dates)
        df.index = np.arange(0,len(df))
        if len(df) > 2:
            for j in range(2,len(df)):
                timespan = df[j] - df[j-2]
                if timespan.days < 7:
                    adopted_users.append(i)            
    else:
        no_login_users.append(i)
         
# Conduct set operations and order the lists
no_login_users_set = set(no_login_users)
has_login_users_set = set(has_login_users)
adopted_users_set = set(adopted_users)
non_adopted_users_set = has_login_users_set - adopted_users_set

adopted_users = list(adopted_users_set)
adopted_users = sorted(adopted_users)


non_adopted_users = list(non_adopted_users_set)
non_adopted_users = sorted(non_adopted_users)

# Classification
classification = []
for i in range(0,12000):
    if i in adopted_users:
        classification.append(1)
    else:
        classification.append(0)

# Now that the users are separated we can look at the users dataframe (u_df)
# And build visualizations that separate the two groups
# Then we will build a Random Forest Classifier to predict the two groups
# From that we will return importance values


au_df = u_df.loc[adopted_users]
au_source = au_df['creation_source']
au_source.index = np.arange(0,len(au_df))
au_source_nums = []
for i in range(len(au_source)):
    if au_source[i] == 'PERSONAL_PROJECTS':
        au_source_nums.append(1)
    elif au_source[i] == 'GUEST_INVITE':
        au_source_nums.append(2)
    elif au_source[i] == 'ORG_INVITE':
        au_source_nums.append(3)
    elif au_source[i] == 'SIGNUP':
        au_source_nums.append(4)
    else:
        au_source_nums.append(5)


au_mailing = au_df['opted_in_to_mailing_list']
au_marketing = au_df['enabled_for_marketing_drip']

au_small_df = au_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
au_small_df.index = np.arange(0,len(au_small_df))
au_small_df['creation_source'] = au_source_nums

# Repeat for Non-Adopted Users (nau)

nau_df = u_df.loc[non_adopted_users]
nau_source = nau_df['creation_source']
nau_source.index = np.arange(0,len(nau_df))
nau_source_nums = []
for i in range(len(nau_source)):
    if nau_source[i] == 'PERSONAL_PROJECTS':
        nau_source_nums.append(1)
    elif nau_source[i] == 'GUEST_INVITE':
        nau_source_nums.append(2)
    elif nau_source[i] == 'ORG_INVITE':
        nau_source_nums.append(3)
    elif nau_source[i] == 'SIGNUP':
        nau_source_nums.append(4)
    else:
        nau_source_nums.append(5)

nau_mailing = nau_df['opted_in_to_mailing_list']
nau_marketing = nau_df['enabled_for_marketing_drip']


nau_small_df = nau_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
nau_small_df.index = np.arange(0,len(nau_small_df))
nau_small_df['creation_source'] = nau_source_nums

# Few variables hist
print('Few variables Adopted Users')
au_small_group_hist  = au_small_df.hist(bins = 5)
print('Few variables Non-Adopted Users')
nau_small_group_hist = nau_small_df.hist(bins = 5)

# All variables hist
print('All variables Adopted Users')
au_group_hist  = au_df.hist(bins = 5)
print('All variables Non-Adopted Users')
nau_group_hist = nau_df.hist(bins = 5)


# Only the ones in few variables should be relevant. 
# The other insight one could gather in a larger scale project would be if the 
# user id's that referred individuals were adopted users. 

# Add a classifier column to the original dataframe that indicates
# an adopted or a non-adopted user.
# Then create a random forest classifier
# Then get importances

u_source = u_df['creation_source']
u_source_nums = []
for i in range(len(u_source)):
    if u_source[i] == 'PERSONAL_PROJECTS':
        u_source_nums.append(1)
    elif u_source[i] == 'GUEST_INVITE':
        u_source_nums.append(2)
    elif u_source[i] == 'ORG_INVITE':
        u_source_nums.append(3)
    elif u_source[i] == 'SIGNUP':
        u_source_nums.append(4)
    else:
        u_source_nums.append(5)
u_df['creation_source'] = u_source_nums
u_df['classification'] = classification

# Create small u_df   
su_df = u_df[['classification', 'creation_source', 
              'opted_in_to_mailing_list', 
              'enabled_for_marketing_drip']] 

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

X = su_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
y = su_df['classification']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     

clf = RandomForestClassifier()
print('Training Model: ')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Finished Predictions for small df')
print(accuracy_score(y_test, y_pred))  

feature_imps = clf.feature_importances_
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances)
importances.plot.bar()
# =============================================================================
# print('Results for small users DataFrame:')
# print('\n')
# print('Feature importance results: ')
# print('\n')
# print('Importance of creation source: ', feature_imps[0])
# print('\n')
# print('Importance of opted_in_to_mailing_list: ', feature_imps[1])
# print('\n')
# print('Importance of enabled_for_marketing_drip: ', feature_imps[2])
# =============================================================================



# Now look at predict proba
probability = clf.predict_proba(X)

df_new = su_df.copy()
df_new['Probability_of_Adoption'] = probability[:,1]

Q1 = [] # [0.0432, 0.121]
Q2 = [] # (0.121, 0.123]
Q3 = [] # (0.123, 0.125]
Q4 = [] # (0.125, 0.133]
Q5 = [] # (0.133, 0.142]
Q6 = [] # (0.142, 0.148]
Q7 = [] # (0.148, 0.152]
Q8 = [] # (0.152, 0.222]

for i in range(len(df_new)):
    if 0.0432 <= df_new['Probability_of_Adoption'][i] <= 0.121:
        Q1.append(i)
        
    elif 0.121 <= df_new['Probability_of_Adoption'][i] <= 0.123:
        Q2.append(i)
        
    elif 0.123 < df_new['Probability_of_Adoption'][i] <= 0.125:
        Q3.append(i)
        
    elif 0.125 < df_new['Probability_of_Adoption'][i] <= 0.133:
        Q4.append(i)
    
    elif 0.133 < df_new['Probability_of_Adoption'][i] <= 0.142:
        Q5.append(i)   
        
    elif 0.142 < df_new['Probability_of_Adoption'][i] <= 0.148:
        Q6.append(i)
        
    elif 0.148 < df_new['Probability_of_Adoption'][i] <= 0.152:
        Q7.append(i)
        
    elif 0.152 < df_new['Probability_of_Adoption'][i] <= 0.222:
        Q8.append(i)
        
q1_df = df_new.loc[Q1]
q2_df = df_new.loc[Q2]
q3_df = df_new.loc[Q3]
q4_df = df_new.loc[Q4]
q5_df = df_new.loc[Q5]
q6_df = df_new.loc[Q6]
q7_df = df_new.loc[Q7]
q8_df = df_new.loc[Q8]

# Note, we fix the random state below to solve the issue of quantiles being empty 
# Depending on different train test splits.

# In this split quantile 2 and 7 are empty in this random_state = 10 example

# -----------
#
# Quantile 1
#
# -----------
print('\n')
print('\n')

if q1_df.empty:
    print('Quartile 1 is empty:')
else:
    X = q1_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q1_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 1')
    print(accuracy_score(y_test, y_pred))  
    
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q1_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 1 [0.0432, 0.121]')
#     plt.show()
#     plt.clf()
# =============================================================================

# -----------
#
# Quantile 2 is empty in this random_state
#
# -----------+
print('\n')
    
if q2_df.empty:
    print('Quartile 2 is empty:')
else:    
    X = q2_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q2_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 2')
    print(accuracy_score(y_test, y_pred))  
    
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q2_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 2 (0.121, 0.123]')
#     plt.show()
#     plt.clf()
# =============================================================================
# -----------
#
# Quantile 3 is empty
#
# -----------
print('\n')

if q3_df.empty:
    print('Quartile 3 is empty:')
else:    
    X = q3_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q3_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train) 
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 3')
    print(accuracy_score(y_test, y_pred))  
    
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q3_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 3 (0.123, 0.125]')
#     plt.show()
#     plt.clf()
# =============================================================================
# -----------
#
# Quantile 4 
#
# -----------
print('\n')

if q4_df.empty:
    print('Quartile 4 is empty:')
else:
    X = q4_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q4_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 4')
    print(accuracy_score(y_test, y_pred))  
        
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q4_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 4 (0.125, 0.133]')
#     plt.show()
#     plt.clf()
# =============================================================================
# -----------
#
# Quantile 5
#
# -----------
print('\n')

if q5_df.empty:
    print('Quartile 5 is empty:')
else:    
    X = q5_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q5_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 5')
    print(accuracy_score(y_test, y_pred))  
    
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q5_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 5 (0.133, 0.142]')
#     plt.show()
#     plt.clf()
# =============================================================================
# -----------
#
# Quantile 6
#
# -----------
print('\n')

if q6_df.empty:
    print('Quartile 6 is empty:')
else:
    X = q6_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q6_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 6')
    print(accuracy_score(y_test, y_pred))  
    
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q6_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 6 (0.142, 0.148]')
#     plt.show()
#     plt.clf()
# =============================================================================
# -----------
#
# Quantile 7
#
# -----------
print('\n')

if q7_df.empty:
    print('Quartile 7 is empty:')
else:    
    X = q7_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q7_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 7')
    print(accuracy_score(y_test, y_pred))  
    
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q7_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 7 (0.148, 0.152]')
#     plt.show()
#     plt.clf()
# =============================================================================
# -----------
#
# Quantile 8 
#
# -----------
print('\n')

if q8_df.empty:
    print('Quartile 8 is empty:')
else:
    X = q8_df[['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip']]
    y = q8_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)     
    
    clf = RandomForestClassifier()
    print('Training Model: ')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('Finished Predictions for Quantile 8')
    print(accuracy_score(y_test, y_pred))  
    
    feature_imps = clf.feature_importances_
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    q8_importances = importances
# =============================================================================
#     importances.plot.bar()
#     plt.title('Quantile 8 (0.152, 0.222]')
#     plt.show()
#     plt.clf()
# =============================================================================


# Over several runs the only consistent truth was that the lower
# probabilities of adopting were correlated with opting into the mailing list
# This would imply one of two things:
# 1.) The mailing list makes logging in not worthwhile
# 2.) It is spammy or is being filtered as spam
    
# Meanwhile enabled for marketing drip increases
# So the marketing drip is working well while the mailing list is not.