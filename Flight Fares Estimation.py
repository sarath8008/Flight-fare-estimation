#!/usr/bin/env python
# coding: utf-8

# **Flight Fares Estimation**
# 
# 
# 
# ![flight gif url](https://media.giphy.com/media/3o6nV8OYdUhiuKja1i/giphy.gif)
# 
# 

# Understanding the flight fares estimation(Business Understanding)
# 
# 
# 
# 
# 
# *To estimate the flight fares there will be several factors which will include like  Date of journey, Duration, Total stops, Additional info 

# **Features and column info**
# 
# 
# 
# 
# 

# variable|variable info
# -----------|--------------------
# Airline|Type of Airline
# Date of journey|reprsents the journey date
# Source|journey starts from
# Destination|journey ends to
# Route|from which place to which place
# Dep Time|time of departure
# Arrival time|time of arrival
# Duration|total journey duration
# Total stops|how many stops in entire duration of journey
# Additional info|any extra information during the journey
# Price|total fare for entire journey

# **DATA**

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
from IPython.display import Image


# In[2]:


rawdata=pd.read_excel('C:\\Users\\sarat\\Downloads\\Data_Train.xlsx')


# In[3]:


pwd


# In[4]:


rawdata.head(10)


# In[5]:


rawdata.dtypes


# In[6]:


rawdata.shape


# In[7]:


rawdata.isnull().sum() 


# In[8]:


rawdata=rawdata.dropna()


# **Data Cleaning**

# In[9]:


rawdata.columns


# In[10]:


rawdata['Airline'].apply(str.lower)


# In[11]:


rawdata['Date_of_Journey']


# In[12]:


rawdata['Date_of_Journey']=rawdata['Date_of_Journey'].str.replace('/','-')


# In[13]:


rawdata['Date_of_Journey'][20:50]


# extracting day,month,weekday

# In[14]:


rawdata['Journey_Day'] = pd.to_datetime(rawdata.Date_of_Journey, format='%d-%m-%Y').dt.day
rawdata['Journey_Month'] = pd.to_datetime(rawdata.Date_of_Journey, format='%d-%m-%Y').dt.month
rawdata['weekday']= pd.to_datetime(rawdata.Date_of_Journey, format='%d-%m-%Y').dt.weekday


# In[15]:


rawdata=rawdata.drop(['Date_of_Journey'],axis=1)


# In[16]:


rawdata.head()


# In[17]:


rawdata['Route']


# Dropping the route column because iam going to considering source & destination

# In[18]:


rawdata=rawdata.drop(['Route'], axis=1)


# In[19]:


rawdata['Duration'][0:30]


# In[20]:


def duration(test):
    test = test.strip()
    total=test.split(' ')
    to=total[0]
    hrs=(int)(to[:-1])*60
    if((len(total))==2):
        mint=(int)(total[1][:-1])
        hrs=hrs+mint
    test=str(hrs)
    return test
rawdata['Duration']=rawdata['Duration'].apply(duration)
rawdata['Duration']


# In[21]:


rawdata['Duration']


# 
# Extracting whether its a morning,evening,night or afternoon flight from departure time & arrival time 
# 
# 
# 5 am to 11 am(morning)
# 
# 11 am to 5pm(afternoon)
# 
# 5pm to 9pm(evening)
# 
# 9pm to 5am(night)

# In[22]:



def deparrtime(x):
    x=x.strip()
    tt=(int)(x.split(':')[0])
    if(tt>=16 and tt<21):
        x='Evening'
    elif(tt>=21 or tt<5):
        x='Night'
    elif(tt>=5 and tt<11):
        x='Morning'
    elif(tt>=11 and tt<16):
        x='Afternoon'
    return x
rawdata['Dep_Time']=rawdata['Dep_Time'].apply(deparrtime)
rawdata['Arrival_Time']=rawdata['Arrival_Time'].apply(deparrtime)


# In[23]:


rawdata.head()


# In[24]:


rawdata['Total_Stops'][0:30]


# converting total stops into 0,1,2----

# In[25]:


def stops(x):
    if(x=='non-stop'):
        x=str(0)
    else:
        x.strip()
        stps=x.split(' ')[0]
        x=stps
    return x
rawdata['Total_Stops']=rawdata['Total_Stops'].apply(stops)
rawdata['Total_Stops']


# In[26]:


pd.options.mode.chained_assignment = None 
for i in range(rawdata.shape[0]):
    if(rawdata.iloc[i]['Additional_Info']=='No info'):
        rawdata.iloc[i]['Additional_Info']='No Info' 
        


# In[27]:


rawdata['Additional_Info'][0:40]


# In[28]:


rawdata['Price']


# In[29]:


rawdata.head(10)


# In[30]:


rawdata[rawdata.duplicated()]


# In[31]:


rawdata=rawdata.drop_duplicates().reset_index(drop=True)


# In[32]:


rawdata.head()


# In[33]:


cleaned_data=rawdata[['Airline','Source','Destination','Dep_Time','Arrival_Time','Duration','Total_Stops','Additional_Info','Price','Journey_Day','Journey_Month','weekday']]


# In[34]:


cleaned_data=cleaned_data.reset_index(drop=True)


# In[35]:


cleaned_data.head()


# In[36]:


rawdata["Duration"] = rawdata["Duration"].astype(int)
rawdata["Journey_Day"] = rawdata["Journey_Day"].astype(object)
rawdata["Journey_Month"] = rawdata["Journey_Month"].astype(object)
rawdata["weekday"] =rawdata["weekday"].astype(object)


# In[37]:


rawdata.info()


# journey month is in object but in column it is in numbers,so replacing numbers into month names by doing copy()

# In[38]:


df1=rawdata.copy()


# In[39]:


df1["Journey_Month"]=df1["Journey_Month"].replace({3:"March",4:"April",5:"May",6:"June"})
df1["Journey_Month"]=df1["Journey_Month"].astype(object)


# In[40]:


rawdata['Journey_Day']=rawdata['Journey_Day'].astype(int)
rawdata['Total_Stops']=rawdata['Total_Stops'].astype(int)
rawdata['weekday']=rawdata['weekday'].astype(int)                               


# In[41]:


rawdata.head()


# In[42]:


df1.head()


# In[43]:


rawdata.info()


# In[44]:


df1["Journey_Month"].unique()


# In[45]:


#rawdata["Journey_Month"]=rawdata["Journey_Month"].replace({3:"March",4:"April",5:"May",6:"June"})
#rawdata["Journey_Month"]=rawdata["Journey_Month"].astype(object)

 #only 3,4,5,6 months are there in data                                                  
rawdata


# In[46]:


rawdata.head()


# In[47]:


rawdata.info()
rawdata.head()
rawdata['Journey_Month'].unique()


# In[48]:


cleaned_data.to_csv(' Train Flight Fares Cleaned Data.csv',index=False)
rawdata.head()


# **Test dada**

# In[49]:


import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

import  seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[50]:


test=pd.read_excel('C:\\Users\\sarat\\OneDrive\\Desktop\\Test_set.xlsx')


# In[51]:


test.head()


# Date of journey

# In[52]:


test['Date_of_Journey'][0:20]


# extracting date of journey to journey day, journey month, week day

# In[53]:


test['Journey_Day']=pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.day
test['Journey_Month']=pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.month
test['weekday']=pd.to_datetime(test.Date_of_Journey,format='%d/%m/%Y').dt.weekday


# In[54]:


test.drop(['Date_of_Journey'],axis=1,inplace=True)


# In[ ]:





# In[ ]:





# In[55]:


test.head()


# Dropping the Route column because iam going to consider source & destination

# In[56]:


#test.drop(['Route'],axis=1,inplace=True)
test.head()


# Dep time, Arrival time are converting into morning,evening,night and afternoon

# In[57]:



def deparrtime(x):
    x=x.strip()
    tt=(int)(x.split(':')[0])
    if(tt>=16 and tt<21):
        x='Evening'
    elif(tt>=21 or tt<5):
        x='Night'
    elif(tt>=5 and tt<11):
        x='Morning'
    elif(tt>=11 and tt<16):
        x='Afternoon'
    return x
test['Dep_Time']=test['Dep_Time'].apply(deparrtime)
test['Arrival_Time']=test['Arrival_Time'].apply(deparrtime)


# In[58]:


test.head()


# converting duration hours into minutes

# In[59]:


def duration(test):
    test = test.strip()
    total=test.split(' ')
    to=total[0]
    hrs=(int)(to[:-1])*60
    if((len(total))==2):
        mint=(int)(total[1][:-1])
        hrs=hrs+mint
    test=str(hrs)
    return test
test['Duration']=test['Duration'].apply(duration)
test['Duration']


# Total stops

# In[60]:


def stops(x):
    if(x=='non-stop'):
        x=str(0)
    else:
        x.strip()
        stps=x.split(' ')[0]
        x=stps
    return x
test['Total_Stops']=test['Total_Stops'].apply(stops)
test['Total_Stops']


# In[61]:



test=test.drop(['Route'],axis=1)


# In[62]:


test['Journey_Day']=test['Journey_Day'].astype(int)
test['Total_Stops']=test['Total_Stops'].astype(int)
test['weekday']=test['weekday'].astype(int)


# In[63]:


cleaned_data.to_csv(' Test Flight Fares Cleaned Data.csv',index=False)
test.head()


# In[64]:


#cleaned_data.to_csv('flight fares cleaned data original.csv',index=False)


# **EDA**(insights)
# 
# 
# EDA, Visualizations: Qualitative and Quantitative analysis

# ### Data Analysis can be done after Data validation and cleaning
# 
# * This can be either,
# 
#     - **Getting Insights (study of present column/s data) - Descriptive Analysis**
#     - **Getting Predictions (predictions of future values of a column data w.r.t other columns) - Predictive analysis**

# EDA is divided into three types of analysis
# 
# 
# Uni-Variate|Bi-Variate|Multi-Variate
# ---|---|----
# Data study of single column|Data study between two columns|Data Study b/w three or more columns

# Visualization Libraries

# In[65]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Uni-Variate 

# In[66]:


data=pd.read_csv('C:\\Users\\sarat\\Downloads\\Flight Fares Cleaned Data.csv')


# data

# In[67]:


data[data.duplicated()]


# **Data Info**
# 
# * In Stats Columns are called as variables
# * for our data we do have dependent (y) - (price) and independent variables(x) - (other columns)
#     
#     **note:**
#     - if the x and y not mentioned for the insights , we need to do analysis for each and every column
#     - if x & y mentioned , then we can have a reference analysis with y for the x

# ** Insights Using EDA**
# 
# * We can get insights on data using Exploratory Data Analysis (EDA) methods
# 
# * EDA will follow two things, 
#     - Stats
#         - Descriptive
#         - Inferential
#     - Visual Analysis

#  Variable Types of data

# Quantitative(Numerical)|Qualitative(Categorical)
# -----------------|---------------
# Dep_Time|Airline
# Arrival_Time|Source
# Duration|Destination
# Total_Stops|Additional_Info
# Price|
# Journey_Day|
# Journey_Month|
# weekday|
# 

# In[68]:


data.head()


# In[69]:


data.shape


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[71]:


from  simple_colors import *

for i in data.columns:
    if data[i].dtype == 'object':
        print()
        print(green("Airline:Source:Destination:Additional_Info",['bold']), i)
        print("=====================================================")
        print(green("Uni-Variate Descriptive Stats:",['bold']))
        print(black("Classes:", ['bold']), data[i].unique())
        print(black("Number of Classes:",['bold']), data[i].nunique())
        print(black('Class Percent:',['bold']))
        print((data[i].value_counts()/data[i].value_counts().sum())*100)
        print("---------------------------------------------------")
        print(black("Mode Value:",['bold']), data[i].mode()[0])
        if i!='Amenities_Cleaned' and i!='Flooring':
            print()
            print(magenta("Visual Analysis:",['bold']))
            print("-----------------------------------------------------")
            plt.figure(figsize = (8,6))
            data[i].value_counts().plot(kind = 'pie')
            plt.show()        
        
    elif data[i].dtype != 'object':
        print()
        print(blue("Numerical Column:",['bold']), i)
        print("=====================================================")
        print(green("Uni-Variate Descriptive Stats:",['bold']))
        print(round(data[i].describe(),))
        print("-----------------------------------------------------")
        print(black("Skewness & Kurtosis:",['bold']), data[i].skew(), ",", data[i].kurt())
        print()
        print(cyan("Visual Analysis:",['bold']))
        print("-----------------------------------------------------")
        plt.figure(figsize = (8,6))
        sns.distplot(data[i])
        #data[i].plot(kind = 'density')
        plt.show()


# Insights(uni-varaite)

# -Among all the airline, Air India has the highest number of passengers
# 
# -Delhi is the highest source city it having 41% of passengers
# 
# -Cochin is the highest destination city it having 41% of passengers
# 
# -According to the data morning is the most preferred departure time
# 
# -Acoording to the data night is the most preferred arrival time
# 
# -50% of the flight duration is 505 minutes
# 
# -75% of the flights are 1 stop
# 
# -78% additional info doesnt have any information
# 
# -most of the passengers preferred journey day is saturday
# 
# -most of the passengerd preferred journey month is june
# 

# Bi-variate

# In[72]:


data.head()


# In[ ]:





# Taking journey month vs price (n-n)

# In[73]:


v1=sns.barplot(x='Journey_Month',y='Price',data=df1)
v1.set_title('Price of Month')
v1.set_xlabel('Month of Booking')
v1.set_ylabel('Price')


# By seeing the bar plot in the month of march price is high compare to other months.

# In[74]:


monthly_avg=df1.groupby(['Journey_Month']).agg({'Price':np.mean}).reset_index()


# In[75]:


monthly_avg.plot(x='Journey_Month',y='Price')


# Taking Destination vs Price( c-n )

# In[76]:


sns.catplot(y='Price',x='Destination',data=df1)


# As per the catplot New Delhi(Destination) is showing the highest price.
# 

# Taking Source vs Price (c - n)

# In[77]:


sns.catplot(x='Source',y='Price',data=df1)


# As per the catplot Banglore (source destination) is having the highest price.

# Taking Airline vs Price (c-n)

# In[78]:


sns.barplot(y='Airline',x='Price',data=df1,orient='h')


# From the above diagram jet airways having the highest price compare to other airways

# Taking Duration vs Price (n-n)

# In[79]:


sns.scatterplot(x='Duration',y='Price',data=df1)


# By seeing the above diagram we are unable write the insight so that we can consider other columns.

# Taking Departure time vs Price (c-n)

# In[80]:


sns.barplot(x='Dep_Time',y= 'Price',data=df1)


# Above barplot shows evening departure time having the highest  price

# Taking Arrival time vs price (c-n)

# In[81]:


sns.barplot(x='Arrival_Time',y= 'Price',data=df1)


# Above barplot shows evening Arrival time having the highest  price

# Taking total stops vs price(n-n)

# In[82]:


sns.scatterplot(x='Total_Stops',y='Price',data=df1)


# so we all know that direct flight fares are high compare to multiple stop flights

# Taking weekday vs price (n-n)

# In[83]:


sns.scatterplot(x='weekday',y='Price',data=df1)


# As we see that 4(friday) are the most expensive price compare to other days

# **Mutli-variate**

# In[84]:


df1.corr()


# In[85]:


plt.figure(figsize=(10,6))
sns.heatmap(df1.corr(),annot=True)


# In[86]:


sns.pairplot(df1.iloc[0:500])


# In[87]:


df1.groupby(['Airline','Source','Destination']).agg({'Price':np.mean}).sort_values(by='Price',ascending=False)


# In[88]:


rawdata


# **Insights**
# 
# 
# Uni- variate

# Numerical
# 
# -Among all the airline, Air India has the highest number of passengers
# 
# -Delhi is the highest source city it having 41% of passengers
# 
# -Cochin is the highest destination city it having 41% of passengers
# 
# -According to the data morning is the most preferred departure time
# 
# -Acoording to the data night is the most preferred arrival time
# 
# 
# Categorical
# 
# -50% of the flight duration is 505 minutes
# 
# -75% of the flights are 1 stop
# 
# -78% additional info doesnt have any information
# 
# -most of the passengers preferred journey day is saturday
# 
# -most of the passengerd preferred journey month is june
# 
# 

# Bi- variate

# -By seeing the bar plot in the month of march price is high compare to other months.
# 
# -As per the catplot New Delhi(Destination) is showing the highest price.
# 
# -As per the catplot Banglore (source destination) is having the highest price.
# 
# -From the above diagram jet airways having the highest price compare to other airways.
# 
# -By seeing the above diagram we are unable write the insight so that we can consider other columns.
# 
# -Above barplot shows evening departure time having the highest  price.
# 
# -Above barplot shows evening Arrival time having the highest  price.
# 
# -so we all know that direct flight fares are high compare to multiple stop flights.
# 
# -As we see that 4(friday) are the most expensive price compare to other days
# 
# 
# 
# 
# 
# 
# 

# In[89]:


df1.info()


# **Predictive modeling**

# input(x) and output (y)

# In[90]:


df1.head()


# we are predicting price values ,so our output column will be price,remaining are input columns

# In[91]:


x=df1.drop(['Price'],axis=1)
y=df1['Price']


# In[92]:


x.head(5)


# In[93]:


y.head(5)


# Feature Engineering(select proper x data for modeling)
# 
# * Feature selection/Deletion
# 
# The above step can be done through the help of EDA and Business decisions
# 
# * Feature Modification (Changing the Values)
# * Feature Generation (Creating New Features)

# In[94]:


x.head()


# In[ ]:





# In[95]:


x.columns


# In[96]:


correlation=df1['Duration'].corr(df1['Price'])


# In[97]:


correlation


# Checking missing values percentage

# In[98]:


x.isnull().sum()/10372


# In[99]:


x.shape


# In[100]:


x.head()


#  Feature Modification/Generation**
# 
# * We have two features **(------------)** which values are in list, need to convert them to regular values by one hot encoding after replacing the missing values

# Dividing Data (x,y) into train and test (Data Validation)**
# 
# * for this we will use sklearn module
# * we can go with 70,30 or 80,20 or 75,25 ratios

# In[101]:


from sklearn.model_selection import train_test_split


# In[102]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25,random_state=123)


# In[103]:


xtrain.head()


# In[104]:


xtrain['Journey_Day']=xtrain["Journey_Day"].astype(int)
xtrain['Total_Stops']=xtrain['Total_Stops'].astype(int)
xtrain['weekday']=xtrain['weekday'].astype(int)


# In[105]:


xtest['Journey_Day']=xtest['Journey_Day'].astype(int)
xtest['Total_Stops']=xtest['Total_Stops'].astype(int)
xtest['weekday']=xtest['weekday'].astype(int)


# In[106]:


xtest.head()


# In[107]:


xtrain=xtrain.reset_index(drop=True)
xtest=xtest.reset_index(drop=True)
ytrain=ytrain.reset_index(drop=True)
ytest=ytest.reset_index(drop=True)


# In[108]:


xtrain.head()


# In[109]:


ytrain.shape,ytest.shape


# Handling Missing Values & Outliers for rawdata and test data

# In[110]:


xtrain.isnull().sum()


# xtrain

# In[111]:


indx = xtrain[xtrain.isnull().sum(axis=1)>=7].index


# In[112]:


xtrain.drop(indx, axis = 0, inplace = True)


# In[113]:


xtrain.shape


# xtest

# In[114]:


indxtest = xtest[xtest.isnull().sum(axis=1)>=7].index


# In[115]:


xtest.drop(indxtest, axis = 0, inplace = True)


# In[116]:


xtest.shape


# In[117]:


def drop(x,y):
    indxs = y.index
    for i in indxs:
        if i not in x.index:
            y.drop(i, axis = 0, inplace = True)


# In[118]:


drop(xtrain, ytrain)
drop(xtest, ytest)


# In[119]:


xtrain.shape, ytrain.shape


# In[120]:


xtest.shape, ytest.shape


# **Handling Na column wise**

# In[121]:


xtrain.isnull().sum()


# In[122]:


xtest.isnull().sum()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Import label encoder 
# colnames = list(rawdata.columns)
# from sklearn import preprocessing 
# 
# # label_encoder object knows how to understand word labels. 
# label_encoder = preprocessing.LabelEncoder() 
#   
# for col in colnames:
#     if rawdata[col].dtype==object:
#         rawdata[col]= label_encoder.fit_transform(rawdata[col]) 

# cat_var=["Airline","Source","Destination","Dep_Time","Arrival_Time","Total_Stops","Additional_Info","Journey_Day","Journey_Month","weekday"] 
# catdf=rawdata[cat_var]

# In[123]:


#catdf.head()


# 

# In[ ]:





# In[ ]:





# outliers for numeric data

# In[124]:


def outlier_detect(df):
    for i in df.describe().columns:
        print("Column:",i)
        print("------------------------------------------------")
        Q1 = df.describe().at['25%',i]
        Q3 = df.describe().at['75%',i]
        IQR = Q3 - Q1
        LTV = Q1 - 1.5 * IQR
        UTV = Q3 + 1.5 * IQR
        
#         fifth = df[i].quantile(0.05)
#         ninetyfifth = df[i].quantile(0.95)
        
        
        print("Lower Outliers:")
        print()
        lowerout = list(df[df[i]<LTV][i])
        lowerout.sort()
        print(lowerout)
        print()
        # mask method is used to replace the values
        df[i] = df[i].mask(df[i]<LTV, round(LTV,0)) # replacing the outlier with ltv (25% value)
        
        print("Upper Outliers:")
        print()
        upperout = list(df[df[i]>UTV][i])
        upperout.sort()
        print(upperout) 
        print()
        
        # mask method is used to replace the values
        df[i] = df[i].mask(df[i]>UTV, round(UTV,0)) # replacing the outlier with utv (75% value)
    return df


# In[125]:




rawdata = outlier_detect(test)


# In[126]:


outlier_detect(rawdata), outlier_detect(test)


# In[127]:


xtrain.head()


# In[128]:


xtest.head()


# In[129]:


ytrain.head()


# In[130]:


ytest.head()


# Data Pre-Processing (rawdata, test)

# Categorical to Numerical (Encoding)
# 
# Machine needs data in numeric format, so we need to convert categorical to numerical, while observing the number of classes , because it will increase the dimensionality if we are converting them to one hot encoding.
# 
# * Label Encoding for ordinal
#     - lets assume cat column data : platinum, gold, silver
#     - ordinal - platinum>gold>silver
#                 3>2>1
# * One hot encodig for nominal
#     - lets assume cat column data: a, b, c
# 
# a|b|c
# --|--|--
# 1|0|0
# 1|0|0
# 0|1|0
# 0|0|1

# nominal data- Airlines,Source, Destination,Additional info
# 
# ordinal data-Dep time, Arrival time,Journey month

# Dep_Time(Ordinal)

# In[131]:


xtrain.Dep_Time.value_counts()


# In[132]:


xtest.Dep_Time.value_counts()


# In[133]:


pd.concat([xtest, ytest], axis = 1).groupby('Dep_Time')['Price'].mean()


# In[134]:


pd.concat([xtrain, ytrain], axis = 1).groupby('Dep_Time')['Price'].mean()


# In[135]:


xtrain.Dep_Time.replace({'Evening':9412, 'Afternoon':9156, 'Morning':9140, 'Night':7956}, inplace = True)


# In[136]:


xtest.Dep_Time.replace({'Evening':8934, 'Afternoon':8761, 'Morning':8927, 'Night':9811}, inplace = True)


# In[137]:


xtrain.Dep_Time.head()


# In[138]:


xtest.Dep_Time.head()


# Arrival_Time(ordinal)

# In[139]:


pd.concat([xtrain, ytrain], axis = 1).groupby('Arrival_Time')['Price'].mean()


# In[140]:


pd.concat([xtest, ytest], axis = 1).groupby('Arrival_Time')['Price'].mean()


# In[141]:


pd.concat([xtrain, ytrain], axis = 1).groupby('Arrival_Time')['Price'].mean()


# In[142]:


pd.concat([xtest, ytest], axis = 1).groupby('Arrival_Time')['Price'].mean()


# In[143]:


xtrain.Arrival_Time.replace({'Evening':10007, 'Night':8876, 'Morning':8478, 'Afternoon':8428}, inplace = True)


# In[144]:


xtest.Arrival_Time.replace({'Evening':10061, 'Night':8780, 'Morning':8616, 'Afternoon':8399}, inplace = True)


# In[145]:


xtrain.head()


# In[146]:


xtrain.Journey_Month.value_counts()


# In[147]:


xtest.Journey_Month.value_counts()


# In[148]:


xtrain.Journey_Month.replace({'May':5, 'June':6, 'March':3, 'April':4}, inplace = True)


# In[149]:


xtest.Journey_Month.replace({'May':5, 'June':6, 'March':3, 'April':4}, inplace = True)


# In[150]:


xtrain.head()


# In[151]:


xtest.head()


# Airlines,source,destination,additional info(nominal)

# In[152]:


from sklearn.preprocessing import OneHotEncoder


# In[153]:


ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)


# using one hot encoder will remember the data , also it will be used to ignore on the unknown data classes

# we will be using the command fit_transform to train and convert the classes into one hot encoding

# In[154]:


ohedata_train = ohe.fit_transform(xtrain[['Airline','Source','Destination','Additional_Info']]).toarray()


# In[155]:


ohedata_train.shape


# In[156]:


# Converting the one hot data to a data frame with col names

ohedata_train = pd.DataFrame(ohedata_train, columns = ohe.get_feature_names_out())


# In[157]:


xtrain=pd.concat([xtrain,ohedata_train],axis=1)


# In[158]:


xtrain.head()


# In[159]:


xtrain=xtrain.drop(['Airline','Source','Destination','Additional_Info'],axis=1)


# we will be using trained **ohe** to transform test data

# In[160]:


ohedata_test = ohe.transform(xtest[['Airline','Source','Destination','Additional_Info']]).toarray()


# In[161]:


ohedata_test = pd.DataFrame(ohedata_test, columns = ohe.get_feature_names_out())


# In[162]:


xtrain.head()


# In[163]:


xtest=pd.concat([xtest,ohedata_test],axis=1)


# In[164]:


xtest=xtest.drop(['Airline','Source','Destination','Additional_Info'],axis=1)


# In[165]:


xtest.head()


# Scaling 

# In[166]:


from sklearn.preprocessing import StandardScaler


# In[167]:


sc = StandardScaler()


# In[168]:


xtrain.iloc[:,0:3]


# In[169]:


xtrain.iloc[:,0:3] = sc.fit_transform(xtrain.iloc[:,0:3])


# In[170]:


xtrain.dtypes


# xtest

# In[171]:


xtest.iloc[:,0:3] = sc.transform(xtest.iloc[:,0:3])


# In[172]:


xtest.head()


# Selecting The Predictive Model
# 
#     * y data is a numeric data , we will be using regression algorithms
# 
#     * Linear Algorithms (when the data is linear to output (having correlation))
#         - Linear Regression
#         - Polynomial Regression
#         - Lasso & Ridge Regression
# 
#     * Non-Linear Algorithms (when the data is non-linear to output (not having correlation))
#         - RandomForest Regressor
#         - SVM Regressor
#         - Knn regressor

# Importing Libraries and Define Models

# In[173]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# In[174]:


# Multiple Linear Regression 

mlr = LinearRegression()

# Polynomial Regression

polyfeat = PolynomialFeatures(degree = 2)  # degree is hyperparam

poly = LinearRegression()

# Lasso (L1) & Ridge (L2)

lasso = Lasso(alpha = 5) # alpha - hyperparam - penalty

ridge = Ridge(alpha = 5)

# Random Forest regressor

rf = RandomForestRegressor(n_estimators=50) # n_estimators - hyperparam - number of decision trees

# Xgb

xgb = XGBRegressor()


# Training
# 
# * Using xtrain, ytrain data
# * Using fit command to train the defined model with xtrain, ytrain

# linear regression

# In[175]:


mlr.fit(xtrain, ytrain)


# Randomforest Regressor

# In[176]:


rf.fit(xtrain, ytrain)


# Model Performance
# 
# * Loss (RMSE - root mean squared error)
# * Performance (R2score)

# In[177]:


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score


# generating predictions for xtest data
# 
# * Using predict method in trained model to get ypredictions on test data
# * Comparing ypred values with ytest for loss and performance

# MLR

# In[178]:


ypred = mlr.predict(xtest)


# In[179]:


ypred


# RF

# In[180]:


ypred = rf.predict(xtest)


# Lasso and Ridge

# In[181]:


lasso.fit(xtrain,ytrain), ridge.fit(xtrain,ytrain)


# Xgb Regressor

# In[182]:


xgb.fit(xtrain, ytrain)


# polynomial

# In[183]:


x_train_p = xtrain.iloc[:,0:8]
x_test_p = xtest.iloc[:,0:8]


# In[184]:


# converting x data to poly features

x_train_poly = polyfeat.fit_transform(x_train_p)

x_test_poly = polyfeat.transform(x_test_p)


# In[185]:


poly.fit(x_train_poly, ytrain)


# In[186]:


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score


# In[187]:


names = ['Multiple Linear Regression','Polynomial Regression','Lasso Regression',
        'Ridge Regression','RandomForest Regressor','Xgboost Regressor']

models = {'mlr':mlr, 'poly':poly, 'lasso':lasso, 'ridge':ridge, 'rf':rf, 'xgb':xgb}


# In[188]:


trainRMSE = []
testRMSE = []

trainscore = []
testscore = []

fit = []

crossvalscore = []


# In[189]:


X = pd.concat([xtrain, xtest], axis = 0)
y = pd.concat([ytrain, ytest], axis = 0)


# In[190]:


xtest.dtypes


# In[191]:


for name, model in models.items():
        
    if name == 'poly':
        
        ptrain = xtrain.iloc[:,0:8]
        ptest = xtest.iloc[:,0:8]
        
        pftrain = polyfeat.transform(ptrain)
        pftest = polyfeat.transform(ptest)
        
        # RMSE , R2score
        
        trainRMSE.append(np.sqrt(mean_squared_error(ytrain, models[name].predict(pftrain))))
        testRMSE.append(np.sqrt(mean_squared_error(ytest, models[name].predict(pftest))))
        trainscore.append(r2_score(ytrain, models[name].predict(pftrain)))
        testscore.append(r2_score(ytest, models[name].predict(pftest)))
        trscore = r2_score(ytrain, models[name].predict(pftrain))
        tescore = r2_score(ytest, models[name].predict(pftest))
        
        # Bias-Variance Trade off
        
        if trscore<0.50 and tescore<0.50:
            fit.append("Underfit")
        elif trscore>0.70 and tescore<0.60:
            fit.append("Overfit")
        else:
            fit.append("Goodfit")
        
        # Cross validation

        scores = cross_val_score(models[name], X.iloc[:,0:8], y, cv=3)
        crossvalscore.append(scores.mean())
        
    else:
        
        # RMSE, R2score

        trainRMSE.append(np.sqrt(mean_squared_error(ytrain, models[name].predict(xtrain))))
        testRMSE.append(np.sqrt(mean_squared_error(ytest, models[name].predict(xtest))))
        trainscore.append(r2_score(ytrain, models[name].predict(xtrain)))
        testscore.append(r2_score(ytest, models[name].predict(xtest)))
        trscore = r2_score(ytrain, models[name].predict(xtrain))
        tescore = r2_score(ytest, models[name].predict(xtest))
        
        # Bias-Variance Trade off
        
        if trscore<0.50 and tescore<0.50:
            fit.append("Underfit")
        elif trscore>0.70 and tescore<0.60:
            fit.append("Overfit")
        else:
            fit.append("Goodfit")

        # Cross-val score
        
        scores = cross_val_score(models[name], X, y, cv=3)
        crossvalscore.append(scores.mean())


# In[192]:


pd.DataFrame({'Model':names, 'Trainedmodel':models.values(), 'TrainRMSE':trainRMSE, 'TestRMSE':testRMSE,
             'Trainscore':trainscore, 'Testscore':testscore, 'Crossvalscore':crossvalscore, 'Fit':fit})


# Real time prediction
# 
# using best model (RF) to predict future given data

# In[193]:


x.columns


# taking sample data can be given by user
# 
# Single row

# In[194]:


data=['Jet Airways','Banglore','New Delhi',675,678,'2h 30m',2,'No info',23,3,5]


# In[195]:


data


# In[196]:


data = pd.DataFrame([data], columns = x.columns)


# In[197]:


data


# In[198]:


def duration(test):
    test = test.strip()
    total=test.split(' ')
    to=total[0]
    hrs=(int)(to[:-1])*60
    if((len(total))==2):
        mint=(int)(total[1][:-1])
        hrs=hrs+mint
    test=str(hrs)
    return test
data['Duration']=data['Duration'].apply(duration)
data['Duration']


# onehot encoding

# In[199]:


ohedata= ohe.transform(data[['Airline','Source','Destination','Additional_Info']]).toarray()


# In[200]:


ohedata = pd.DataFrame(ohedata, columns = ohe.get_feature_names_out())


# In[201]:


ohedata.head()


# In[202]:


data=pd.concat([data,ohedata],axis=1)


# In[203]:


data=data.drop(['Airline','Source','Destination','Additional_Info'],axis=1)


# In[204]:


data.dtypes


# In[205]:


data.columns


# In[206]:


rf.predict(data)[0]


# In[207]:


data1=['IndiGo','Banglore','Kolkata',679,888,'3h 40m',2,'No info',22,1,1]


# In[208]:


data1


# In[209]:


data1 = pd.DataFrame([data1], columns = x.columns)


# In[210]:


data1


# In[211]:


def duration(test):
    test = test.strip()
    total=test.split(' ')
    to=total[0]
    hrs=(int)(to[:-1])*60
    if((len(total))==2):
        mint=(int)(total[1][:-1])
        hrs=hrs+mint
    test=str(hrs)
    return test
data1['Duration']=data1['Duration'].apply(duration)
data1['Duration']


# In[212]:


ohedata= ohe.transform(data1[['Airline','Source','Destination','Additional_Info']]).toarray()


# In[213]:


ohedata = pd.DataFrame(ohedata, columns = ohe.get_feature_names_out())


# In[214]:


ohedata.head()


# In[215]:


data1=pd.concat([data1,ohedata],axis=1)


# In[216]:


data1=data1.drop(['Airline','Source','Destination','Additional_Info'],axis=1)


# In[217]:


rf.predict(data1)[0]


# In[ ]:




