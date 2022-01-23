# EDA_of_Preferances_of_Students_regarding_universities

import re
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


df = pd.read_excel('universities.xlsx')
#Check the columns with 20% or more with NaN values
type(df)
df.shape
#df.info(max_cols = len(df))
df.isna().sum().sort_values(ascending=False)
perc = df.isna().sum()/len(df)*100
ax = perc[perc>=20].sort_values(ascending=False).plot.bar(title = 'Percentage of NaN values',figsize=(12,5))
ax.set_ylabel('% of NaN elements')


# Drop columns with NaN values
col_off = df.isna().sum()[df.isna().sum()>=(0.2*len(df))] 
list_col_off = col_off.index.to_list()
dfnw = df.copy(deep=True)
dfnw.drop(list_col_off,axis=1,inplace=True)



int_col=['Name', 'year', 'Highest degree offered', "Offers Bachelor's degree",
       "Offers Master's degree",
       "Offers Doctor's degree - research/scholarship",
       "Offers Doctor's degree - professional practice", 'Applicants total',
       'Admissions total', 'Enrolled total', 'Estimated enrollment, total',
       'Tuition and fees, 2013-14',
       'Total price for in-state students living on campus 2013-14',
       'Total price for out-of-state students living on campus 2013-14',
       'State abbreviation', 'Control of institution', 'Total  enrollment',
       'Full-time enrollment', 'Part-time enrollment',
       'Undergraduate enrollment', 'Graduate enrollment',
       'Full-time undergraduate enrollment',
       'Part-time undergraduate enrollment',
       'Percent of total enrollment that are women',
       'Percent of undergraduate enrollment that are women',
       'Percent of graduate enrollment that are women',
       'Graduation rate - Bachelor degree within 4 years, total',
       'Graduation rate - Bachelor degree within 5 years, total',
       'Graduation rate - Bachelor degree within 6 years, total'
       ]
       
       
       #Drop rows with NaN values

dfnw = dfnw[int_col]
dfnw[dfnw['Total  enrollment'].isna()]
a = dfnw[dfnw['Name']=='University of North Georgia'].index[0]
b = dfnw[dfnw['Name']=='Texas A & M University-Galveston'].index[0]
dfnw.drop([a,b],inplace=True)


#print("The dataframe has now {} rows instead of {} rows and {} columns instead of {} columns".
      #format(dfnw.shape[0],df.shape[0],dfnw.shape[1],df.shape[1]))

col = dfnw.select_dtypes(include=['float64','int64']).columns

#Check the presence of negative numbers
lst = []
for i in col:
    y = any(x<0 for x in dfnw[i])
    if y ==True:
        lst.append(y)
print('There are {} number of negative numbers in data frame'.format(len(lst)))




#Replace 0 values with NaN
dfnw = dfnw.replace(0,np.nan)

dfnw[['Name','Applicants total']].sort_values('Applicants total').head()


#Replace charachters in headers
def funct(list_headers,chars):
    new_headers = []
    for head in list_headers:
        for char in chars:
            if char in head:
                head = head.replace(char,'_')
            head = head
        new_headers.append(head)
    return new_headers



def func(headers,chars):
    new_headers=list()
    for header in headers:    
        for char in chars:
            if char in header:
                header=header.replace(char,'')
            header=header   
        new_headers.append(header)
    return new_headers   
            
headers = funct(dfnw.columns,[' - ',' '] )
headers = func(headers,["'",',',':','-','/'])

lst_new_headers = []

for head in headers:
    head = head.casefold()
    
    if 'degrese' in head:
        head = head.replace('degrese','degrees')
        
    lst_new_headers.append(head)
    
dfnw.columns = lst_new_headers
        
            
dfnw = dfnw.rename(columns={'state_abbreviation':'state'})
print(headers)


df = dfnw
high_app = df[['name','applicants_total']].sort_values('applicants_total',ascending=False).head()
plt.figure(figsize=(12,8))
sns.barplot(x = 'applicants_total',y = 'name',data=high_app)
plt.title('Top 5 american universities with higher applicants')
plt.xlabel('Number of applicants')



#  To find out if the universities with the most applications are the preferred ones
# Find some relationships with the number of admissions and enrollments
plt.figure(figsize=(16,6))

plt.subplot(1,3,1)
sns.histplot(df.applicants_total,bins=50)

# Should be in 3 parenthesis
plt.title('''Histogram of Number of Applications. 
          Mean: {:.1f}, Median: {:.1f}'''.format(df.applicants_total.mean(),df.applicants_total.median()))
plt.xlabel('Number of Applicants')
plt.axis([0,30000,0,350])
plt.xticks(rotation=10)
plt.grid()

plt.subplot(1,3,2)
sns.histplot(df.admissions_total,bins=50)
plt.title('''Histogram of Number of Admissions.
          Mean: {:.1f},Median: {:.1f}'''.format(df.admissions_total.mean(),df.admissions_total.median()))
plt.xlabel('Number of Admissions')
plt.axis([0,10000,0,350])
plt.xticks(rotation=10)
plt.grid()

plt.subplot(1,3,3)
sns.histplot(df.enrolled_total,bins=50)
plt.title('''Histogram of Number of Enrollments.
          Mean:{:.1f},Median:{:.1f}'''.format(df.enrolled_total.mean(),df.enrolled_total.median()))
plt.xlabel('Number of Enrollments')
plt.axis([0,5000,0,350])
plt.xticks(rotation=10)
plt.grid()

plt.tight_layout(pad=2)


# Could the number of applications tell us that a university is one of the most preferred by students?
# Do students prefer a university where it is easier for them to be admitted?

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
plt.title('Applications vs Admissions')
sns.scatterplot(y=df.admissions_total,x=df.applicants_total,hue=df.control_of_institution)
plt.ylabel('Number of Admissions')
plt.xlabel('Number of Applicants')
plt.grid()

plt.subplot(1,2,2)
plt.title('Admissions vs Enrollments')
sns.scatterplot(x = df.admissions_total,y=df.enrolled_total,data=df,hue=df.control_of_institution)
plt.xlabel('Number of Admissions')
plt.ylabel('Number of Enrollments')
plt.grid()

plt.tight_layout(pad=2)


# Acceptance rate
df['acceptance_rate'] = ((df.admissions_total/df.applicants_total)*100).round(2)

# Enrollment rate
df['enrollment_rate'] = ((df.enrolled_total/df.applicants_total)*100).round(2)


plt.figure(figsize=(12,5))
sns.scatterplot(x='applicants_total',y='enrollment_rate',data=df)
plt.title('Applications vs Enrollment rate')
plt.ylabel('Enrollment rate %')
plt.xlabel('Number of Applications')


plt.figure(figsize=(16,6))
sns.scatterplot(x='acceptance_rate',y='enrollment_rate',data=df,hue=df.control_of_institution)
plt.title('Acceptance vs enrollment rate')
plt.xlabel('Acceptance rate %')
plt.ylabel('Enrollment rate %')



high_accept = df[df['acceptance_rate'].notnull()][['name','acceptance_rate','enrollment_rate']].sort_values('acceptance_rate',ascending=False).head(24)
low_accept = df[df['acceptance_rate'].notnull()][['name','acceptance_rate','enrollment_rate']].sort_values('acceptance_rate',ascending=False).tail(24)



plt.figure(figsize=(16,4))

plt.subplot(1,2,1)
ind=np.arange(len(high_accept)) #number of universities
width = 0.35 #space

plt.bar(ind,high_accept.acceptance_rate,width,label='Acceptance Rate')
plt.bar(ind+width,high_accept.enrollment_rate,width,label='Enrollment Rate')
plt.title('''Acceptance and Enrollment rates.
          25 universities with the highest enrollment rates''')

plt.ylabel('Rates %')
plt.xticks(ind+width, high_accept.name.values,rotation =90)
plt.legend(loc='best')


plt.subplot(1,2,2)
ind = np.arange(len(low_accept)) #number of universities
width = 0.35

plt.bar(ind,low_accept.acceptance_rate,width,label='Acceptance Rate')
plt.bar(ind+width,low_accept.enrollment_rate,width,label='Enrollment Rate')
plt.title('''Acceptance and Enrollment rates.
          25 universities with the lowest Acceptance Rate''')
plt.ylabel('Rates %')
plt.xticks(ind+width,low_accept.name.values,rotation = 90)
plt.legend(loc='best')




# Do students prefer public or private universities?

public_univ = df[df['control_of_institution']=='Public']
public_univ = public_univ[public_univ['applicants_total'].notnull()]

private_univ = df[df['control_of_institution']=='Private not-for-profit']
private_univ = private_univ[private_univ['applicants_total'].notnull()]


plt.figure(figsize=(16,7))

plt.subplot(1,2,1)
plt.hist([public_univ['applicants_total'],private_univ['applicants_total']],stacked=True,bins=25)
plt.axis([0,31000,0,700])
plt.title('Distributions of Applications')
plt.xlabel('Number of Applications')
plt.ylabel('Number of Universities')
plt.legend(['Public universities.({})'.format(len(public_univ)),'Private universities.({})'.format(len(private_univ))])

plt.subplot(1,2,2)
sns.barplot(x = df.control_of_institution,y = df.applicants_total)
plt.title('''Average and Variation of Applications 
          According to the type of control''')
plt.xlabel('')
plt.ylabel('Number of Applications')
plt.tight_layout(pad=1)



print('The minimum number of applications for private universities was {}; whereas, for public universities was {}.'.format(int(private_univ.applicants_total.min()),int(public_univ.applicants_total.min())))
print('The maximum number of applications for private universities was {}; whereas, for public universities was {}.'.format(int(private_univ.applicants_total.max()), int(public_univ.applicants_total.max())))



g = sns.jointplot(x=df.enrollment_rate,y=df.applicants_total,hue=df.control_of_institution)
g = (g.set_axis_labels('Enrollment rate %','Applications'))



# Do students prefer universities with low tuition and fees?

r = sns.jointplot(x=df.tuition_and_fees_201314,y=df.applicants_total,hue=df.control_of_institution)
r = (r.set_axis_labels('Tuition and Fees $','Applications'))


r = sns.jointplot(x=df.tuition_and_fees_201314,y=df.enrollment_rate,hue=df.control_of_institution,height=9)
r = (r.set_axis_labels('Tuition and Fees $','Enrollment rate'))



#  Do students prefer a university for its low cost of on-campus living?
plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
sns.barplot(y = df.total_price_for_instate_students_living_on_campus_201314,x=df.control_of_institution)
plt.title('''Average and variation of the Cost for 
In-State Students Living on Campus (2013-2014)''')
plt.xlabel('')
plt.ylabel('Cost of living on campus $')

plt.subplot(1,2,2)
sns.scatterplot(x = df.total_price_for_instate_students_living_on_campus_201314,y=df.enrollment_rate,hue=df.control_of_institution)
plt.title('''Cost for In-State Students Living 
on Campus vs Enrollment Rate (2013-2014)''')
plt.xlabel('Cost of living on campus $')
plt.ylabel('Enrollment rate')

plt.tight_layout(pad=2)



plt.figure(figsize=(16,7))
plt.subplot(1,2,2)
sns.scatterplot(x = df.total_price_for_outofstate_students_living_on_campus_201314,y=df.enrollment_rate,hue=df.control_of_institution)
plt.title('''Cost for Out-State Students Living 
on Campus vs Enrollment Rate (2013-2014)''')
plt.xlabel('Cost of living on campus $')
plt.ylabel('Enrollment rate')


plt.subplot(1,2,1)
sns.barplot(y= df.total_price_for_outofstate_students_living_on_campus_201314,x=df.control_of_institution)
plt.title('''Average and variation of the Cost 
for Out-State Students Living on Campus (2013-2014)''')
plt.xlabel('')
plt.ylabel('Cost of living on campus $')

plt.tight_layout(pad=2)



