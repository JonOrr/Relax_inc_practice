# Relax_inc_practice

This projects looks at both user engagement (takehome_user_engagement.csv) and user website interactions (takehome_users.csv) among 12,000 users. 

The task is to determine what contributes to users adopting website visits to Relax.inc as a part of their weekly routine. A user is considered to have adopted the service once they login three times within the same week. After some data wrangling the users were separated into distinct lists of adopted and non-adopted users, which were then used to created two separate dataframes au_df and nau_df. Plots of the variables in these files showed that only three varaibles would be of key interest to differentiate the groups: 

creation_source: This refers to where the user's account originated from. Be it google, a referral, or directly from the site. 
opted_in_to_mailing_list: This indicates whether or not the user opted into the mailing list.
enabled_for_marketing_drip: Whether or not they are on the regular email marketing drip. 

After constructing a Random Forest Classifier using a 70/30 train/test split on su_df (This is the users data frame with the columns mentioned above) it was apparent that creation source was the most important variable, followed by opted_in_to_mailing_list followed by enabled_for_marketing_drip. 

While this informs us that the creation_source is the most important for determining who is likely to become an adopted member, what is indicative of the different degrees of probability that a user will adopt the service. The logical hypothesis would be the creation_source, but by taking quartiles of the predicited probabilities for becoming an adopted member we see another story. 

After breaking down the quartiles and constructing Random Forest Classifiers on each subset of data we learn that in the least probable bracket whether or not the user opted into the mailing list is the most important. Meanwhile, in the most probable set the most important variable was whether or not they were in the marketing drip. 

This implies that the marketing drip is more effective than the mailing list. Additionally it could point to the mailing list being filtered out as spam by various email servies, or perhaps the users do not like the mailing list content. I would recommend that Relax.Inc do a user survey regarding their mailing list and marketing drip. 
