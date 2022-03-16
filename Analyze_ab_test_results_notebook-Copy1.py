#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[44]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[45]:


df = pd.read_csv('ab_data.csv')


# b. Use the cell below to find the number of rows in the dataset.

# In[46]:


df.shape


# c. The number of unique users in the dataset.

# In[47]:


df.nunique()


# d. The proportion of users converted.

# In[48]:


len(df.index)


# In[49]:


len(df.query('converted==1'))/len(df.index)


# e. The number of times the `new_page` and `treatment` don't match.

# In[50]:


dont_line_up = df.query('(group == "treatment" and landing_page == "old_page")' + 
                        'or (group == "control" and landing_page == "new_page")')
len(dont_line_up)


# f. Do any of the rows have missing values?

# In[51]:


df.isnull().sum()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[52]:


df2 = df.copy()
df2 = df2[~df2.index.isin(dont_line_up.index)]
df2.head()


# In[53]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[54]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[55]:


df2[df2.user_id.duplicated()].user_id


# c. What is the row information for the repeat **user_id**? 

# In[56]:


df2[df2.user_id.duplicated(keep=False)]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[57]:


df2.drop_duplicates(subset='user_id', keep="last", inplace=True)


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[58]:


conv_rate = df2.converted.mean()
conv_rate


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[59]:


control_converted = df2.query('group == "control"')['converted'].mean()
control_converted


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[60]:


treatment_converted = df2.query('group == "treatment"')['converted'].mean()
treatment_converted


# d. What is the probability that an individual received the new page?

# In[61]:


len(df2.query('landing_page == "new_page"'))/len(df2)


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Although the old page has a higher conversion probability than the new page, the difference between the old and new page are statistically negligible.  The old page has a probablity of 0.1204 while the new page is 0.1188.  More time is necessary to find a definitive answer.  If we allow the test to continue, we will be able to find more conclusive data on what page has a higher conversion rate.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# The null hypothesis is that the old page has a higher conversion rate than the new page.  The alternative hypothesis is that the new page will have a higher conversion rate than the pold page.

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[62]:


p_new = conv_rate
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[63]:


p_old = conv_rate
p_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[64]:


n_new = len(df2.query('group == "treatment"'))
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[65]:


n_old = len(df2.query('group == "control"'))
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[66]:


new_page_converted = np.random.choice(2, size=n_new, p=[1-p_new, p_new])


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[67]:


old_page_converted = np.random.choice(2, size=n_old, p=[1-p_old, p_old])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[68]:


new_page_converted.mean() - old_page_converted.mean()


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[69]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.choice(2, size=n_new, p=[1-p_new, p_new])
    old_page_converted = np.random.choice(2, size=n_old, p=[1-p_old, p_old])
    p_diffs.append(new_page_converted.mean() - old_page_converted.mean())


# In[70]:


p_diffs = np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[71]:


plt.hist(p_diffs)
plt.title('Sampling Distribution for The Difference in Conversion Rates From The Null')
plt.xlabel('Difference in Conversion Rates')
plt.ylabel('Frequency');


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[73]:


plt.hist(p_diffs)
plt.title('Sampling Distribution for The Difference in Conversion Rates From The Null')
plt.xlabel('Difference in Conversion Rates')
plt.ylabel('Frequency');


# In[ ]:


(obs_diff < p_diffs).mean()


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **The P value is the probabability a statistical measure of an assumed probablity distribution will be greater than or equal to the null hypothesis or it will be less than or equal to the null hypothesis. If the P value is greater than or eaqual to.05 the P value is  not statistically significant and the test hypothesis can be rejected, if less than .05 the P value is statitically significant and the test hypothesis can be accepted. Since in our case the P value is greater than .05 the null hypothesis can be accepted.** 

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[32]:


import statsmodels.api as sm

convert_old = len(df2[(df2.group == 'control') & (df2.converted == 1)])
convert_new = len(df2[(df2.group == 'treatment') & (df2.converted == 1)])
n_old = len(df2.query('group == "control"'))
n_old
n_new = n_new = len(df2.query('group == "treatment"'))
n_new

print(convert_old)
print(n_old)
print(convert_new)
print(n_new)


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[33]:


z, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
z, p_value   


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **There is not a statististical difference between the two pages conversion rates but just like the P value showed above, we still accept the null hypothesis.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic regression.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[34]:


# create a column for the intercept
df2['intercept'] = 1

# create the necessary dummy variables
df2[['control', 'ab_page']] = pd.get_dummies(df['group'])
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[35]:


logit_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = logit_mod.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[42]:


results.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **Even when running a logistic regression we still accept the null hypothesis (.189) as it is greater than .05.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **adding other factors and using them in the model could help give us more precise results.  It could give us more insights into why they converted, demographics like user's age, gender, etc may play a role. 
# We do have to watch out for multicollinearity as adding too many factors could cause the x-variables to be related to one another.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[74]:


countries_df = pd.read_csv('./countries.csv')
# join the dfs on user_id
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()


# In[75]:


# check the counrty values
df_new.country.value_counts()


# In[76]:


# create the necessary dummy variables
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new.head()


# In[87]:


## logistic regression model
logit_mod = sm.Logit(df_new['converted'], df_new[['CA', 'UK']])
results = logit_mod.fit()
results.summary2()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[82]:


# create columns for the interaction between page and country 
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new.head()


# In[89]:


# logistic regression model
logit_mod = sm.Logit(df_new['converted'], df_new[['CA', 'UK']])
results = logit_mod.fit()
results.summary2()


# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

