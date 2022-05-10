from gym import spaces
import numpy as np
import requests
import pandas as pd
import json
import time
import urllib.request
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os
import dateutil
import configparser
from dateutil.relativedelta import relativedelta
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from jproperties import Properties

# print("DEBUGGING. STATEMENT1")

configs = Properties()
with open('keywords.properties', 'rb') as read_prop:
    configs.load(read_prop)

IsStore = ""
prop_view = configs.items()
for item in prop_view:
    if ((item[0]) == "IsStore"):
        IsStore = item[1].data

print("store--->: ",IsStore)
if(IsStore == "yes"):
#     print("entered store line")
    m=50    	#max capacity of warehouse
    K=3      	#constant part of order cost (K in document), can be cost of fuel
    c=4      	#variable part of order cost (c(a_t) in document)
    h=0.0025    #holding cost 
    p=4.5      	#selling price of product is such that PROFIT = 12.5%
    R=K		 	#return cost = K because cost of fuel is same for to and fro journeys
    n=40        #max capacity of store
    storeFlag = True 

if(IsStore == "no"):
    print("entered True line")
    m=50    	#max capacity
    K=3      	#constant part of order cost (K in document), can be cost of fuel
    c=4      	#variable part of order cost (c(a_t) in document)
    h=0.0025    #holding cost 
    p=4.5      	#selling price of product is such that PROFIT = 12.5%
    R=K		 	#return cost = K because cost of fuel is same for to and fro journeys
    storeFlag = False


# Initialized Values
lamda_mon=""
lamda_tue=""
lamda_wed=""
lamda_thu=""
lamda_fri=""
lamda_sat=""
lamda_sun=""

# START: EXTRACT basic parameters from Properties File

# configs = Properties()
# with open('keywords.properties', 'rb') as read_prop:
#     configs.load(read_prop)
    
# prop_view = configs.items()
# for item in prop_view:
#     if ((item[0]) == "m"):
#         m = item[1].data
#     elif ((item[0]) == "K"):
#         K = item[1].data
#     elif((item[0]) == "c"):
#         c = item[1].data
#     elif((item[0]) == "h"):
#         h = item[1].data
#     elif((item[0]) == "p"):
#         p = item[1].data
#     elif((item[0]) == "R"):
#         R = item[1].data
#     elif((item[0]) == "n"):
#         n = item[1].data
#     elif((item[0]) == "storeFlag"):
#         storeFlag = item[1].data 

# print("m:",m)
# print("type(m):",type(m))
# print("K:",K)
# print("c:",c)
# print("h:",h)
# print("p:",p)
# print("R:",R)
# print("n:",n)
# print("storeFlag:",storeFlag)
        
        
# END



#START: DEMAND CODE#

store = pd.read_csv('data_preprocessed.csv', sep=';')
# store.head(10)
store['DEMAND_DATE'] = pd.to_datetime(store['DEMAND_DATE'])

##########################################################################################################################################
#STEP 1: EDA Analysis
##########################################################################################################################################
store = store[store.TOTAL_FISCHPROD_DEMAND_T1>0]
store.loc[store['YEAR_2013']==1 ,['DEMAND_DATE','TOTAL_FISCHPROD_DEMAND_T1']].plot(x='DEMAND_DATE',y='TOTAL_FISCHPROD_DEMAND_T1',title='Demand for Day T1',figsize=(16,4))

store = store[store.TOTAL_FISCHPROD_DEMAND_T1>0]
store.loc[store['YEAR_2014']==1 ,['DEMAND_DATE','TOTAL_FISCHPROD_DEMAND_T1']].plot(x='DEMAND_DATE',y='TOTAL_FISCHPROD_DEMAND_T1',title='Demand for Day T1',figsize=(16,4))

store = store[store.TOTAL_FISCHPROD_DEMAND_T1>0]
store.loc[store['YEAR_2015']==1 ,['DEMAND_DATE','TOTAL_FISCHPROD_DEMAND_T1']].plot(x='DEMAND_DATE',y='TOTAL_FISCHPROD_DEMAND_T1',title='Demand for Day T1',figsize=(16,4))

#Year wise Demand

filtered_store = store.loc[(store['DEMAND_DATE']>= '2013-01-01') & (store['DEMAND_DATE']<= '2013-12-31')]
filtered_store
Year2013Total = filtered_store['TOTAL'].sum()
print("Year 2013:",Year2013Total)

filtered_store = store.loc[(store['DEMAND_DATE']>= '2014-01-01') & (store['DEMAND_DATE']<= '2014-12-31')]
filtered_store
Year2014Total = filtered_store['TOTAL'].sum()
print("Year 2014:",Year2014Total)

filtered_store = store.loc[(store['DEMAND_DATE']>= '2015-01-01') & (store['DEMAND_DATE']<= '2015-12-31')]
filtered_store
Year2015Total = filtered_store['TOTAL'].sum()
print("Year 2015:",Year2015Total)

# Weekday wise Demand
filtered_store = store.loc[(store['Montag']== 1)]
filtered_store
MontagTotal = filtered_store['TOTAL'].sum()
print("Montag:", MontagTotal)

filtered_store = store.loc[(store['Dienstag']== 1)]
filtered_store
DienstagTotal = filtered_store['TOTAL'].sum()
print("Dienstag:", DienstagTotal)

filtered_store = store.loc[(store['Mittwoch']== 1)]
filtered_store
MittwochTotal = filtered_store['TOTAL'].sum()
print("Mittwoch:", MittwochTotal)

filtered_store = store.loc[(store['Donnerstag']== 1)]
filtered_store
DonnerstagTotal = filtered_store['TOTAL'].sum()
print("Donnerstag:", DonnerstagTotal)

filtered_store = store.loc[(store['Freitag']== 1)]
filtered_store
FreitagTotal = filtered_store['TOTAL'].sum()
print("Freitag:", FreitagTotal)

filtered_store = store.loc[(store['Samstag']== 1)]
filtered_store
SamstagTotal = filtered_store['TOTAL'].sum()
print("Samstag:", SamstagTotal)

filtered_store = store.loc[(store['Sonntag']== 1)]
filtered_store
SonntagTotal = filtered_store['TOTAL'].sum()
print("Sonntag:", SonntagTotal)

df = pd.DataFrame({'Days':['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag'], 'Day wise Inventory':[MontagTotal,DienstagTotal,MittwochTotal,DonnerstagTotal,FreitagTotal, SamstagTotal,SonntagTotal]})
ax = df.plot.bar(x='Days', y='Day wise Inventory')

#saturday inventory is higher, sunday inventory is lower

# Month wise Demand
filtered_store = store.loc[(store['MONTH_JAN']== 1)]
filtered_store
MonthJanTotal = filtered_store['TOTAL'].sum()
print("MonthJanTotal:", MonthJanTotal)

filtered_store = store.loc[(store['MONTH_FEB']== 1)]
filtered_store
MonthFebTotal = filtered_store['TOTAL'].sum()
print("MonthFebTotal:", MonthFebTotal)

filtered_store = store.loc[(store['MONTH_MAR']== 1)]
filtered_store
MonthMarTotal = filtered_store['TOTAL'].sum()
print("MonthMarTotal:", MonthMarTotal)

filtered_store = store.loc[(store['MONTH_APR']== 1)]
filtered_store
MonthAprTotal = filtered_store['TOTAL'].sum()
print("MonthAprTotal:", MonthAprTotal)

filtered_store = store.loc[(store['MONTH_MAY']== 1)]
filtered_store
MonthMayTotal = filtered_store['TOTAL'].sum()
print("MonthMayTotal:", MonthMayTotal)

filtered_store = store.loc[(store['MONTH_JUN']== 1)]
filtered_store
MonthJunTotal = filtered_store['TOTAL'].sum()
print("MonthJunTotal:", MonthJunTotal)

filtered_store = store.loc[(store['MONTH_JUL']== 1)]
filtered_store
MonthJulTotal = filtered_store['TOTAL'].sum()
print("MonthJulTotal:", MonthJulTotal)


filtered_store = store.loc[(store['MONTH_AUG']== 1)]
filtered_store
MonthAugTotal = filtered_store['TOTAL'].sum()
print("MonthAugTotal:", MonthAugTotal)


filtered_store = store.loc[(store['MONTH_SEP']== 1)]
filtered_store
MonthSepTotal = filtered_store['TOTAL'].sum()
print("MonthSepTotal:", MonthSepTotal)


filtered_store = store.loc[(store['MONTH_OCT']== 1)]
filtered_store
MonthOctTotal = filtered_store['TOTAL'].sum()
print("MonthOctTotal:", MonthOctTotal)


filtered_store = store.loc[(store['MONTH_NOV']== 1)]
filtered_store
MonthNovTotal = filtered_store['TOTAL'].sum()
print("MonthNovTotal:", MonthNovTotal)


filtered_store = store.loc[(store['MONTH_DEC']== 1)]
filtered_store
MonthDecTotal = filtered_store['TOTAL'].sum()
print("MonthDecTotal:", MonthDecTotal)

df = pd.DataFrame({'Months':['MONTH_JAN', 'MONTH_FEB', 'MONTH_MAR', 'MONTH_APR', 'MONTH_MAY', 'MONTH_JUN', 'MONTH_JUL', 'MONTH_AUG', 'MONTH_SEP', 'MONTH_OCT', 'MONTH_NOV', 'MONTH_DEC'], 'Month wise Inventory':[MonthJanTotal,MonthFebTotal,MonthMarTotal,MonthAprTotal,MonthMayTotal,MonthJunTotal,MonthJulTotal,MonthAugTotal,MonthSepTotal,MonthOctTotal,MonthNovTotal,MonthDecTotal]})
ax = df.plot.bar(x='Months', y='Month wise Inventory')

# October Inventory is higher, September Inventory is lower

# Seafoodwise Inventory
CALAMARI = store['CALAMARI'].sum()
print("CALAMARI Inventory:", CALAMARI)

FISCH = store['FISCH'].sum()
print("FISCH Inventory:", FISCH)

GARNELEN = store['GARNELEN'].sum()
print("GARNELEN Inventory:", GARNELEN)

HAEHNCHEN = store['HAEHNCHEN'].sum()
print("HAEHNCHEN Inventory:", HAEHNCHEN)

KOEFTE = store['KOEFTE'].sum()
print("KOEFTE Inventory:", KOEFTE)

LAMM = store['LAMM'].sum()
print("LAMM Inventory:", LAMM)

STEAK = store['STEAK'].sum()
print("STEAK Inventory:", STEAK)

TOTAL_FISCHPROD = store['TOTAL_FISCHPROD'].sum()
print("TOTAL_FISCHPROD Inventory:", TOTAL_FISCHPROD)

TOTAL_FLEISCH = store['TOTAL_FLEISCH'].sum()
print("TOTAL_FLEISCH Inventory:", TOTAL_FLEISCH)

df = pd.DataFrame({'Seafood Type':['CALAMARI', 'FISCH', 'GARNELEN', 'HAEHNCHEN', 'KOEFTE', 'LAMM', 'STEAK', 'TOTAL_FISCHPROD', 'TOTAL_FLEISCH'], 'Seafood wise Inventory':[CALAMARI,FISCH,GARNELEN,HAEHNCHEN,KOEFTE,LAMM,STEAK,TOTAL_FISCHPROD,TOTAL_FLEISCH]})
ax = df.plot.bar(x='Seafood Type', y='Seafood wise Inventory')

#Haenchen has highest inventory while calamari has lowest Inventory

##########################################################################################################################################
#STEP 2: Extract Demand Levels from Dataset
##########################################################################################################################################
# print("DEBUGGING. STATEMENT2")
# print("Entering STEP 2...")

configs = Properties()
with open('keywords.properties', 'rb') as read_prop:
    configs.load(read_prop)
    
prop_view = configs.items()
start_date = ""
end_date = ""
for item in prop_view:
    if ((item[0]) == "start_date"):
        start_date = item[1].data
    elif ((item[0]) == "end_date"):
        end_date = item[1].data
# print(start_date)
# print(end_date)


store = store[(store['DEMAND_DATE']> start_date) & (store['DEMAND_DATE']< end_date)]
display(store)

demand_mon = round((store["TOTAL_DEMAND_T1"].mean()))
# print("monday_demand: ",demand_mon)
demand_tue = round((store["TOTAL_DEMAND_T2"].mean()))
# print("tuesday_demand: ",demand_tue)
demand_wed = round((store["TOTAL_DEMAND_T3"].mean()))
# print("wednesday_demand: ",demand_wed)
demand_thu = round((store["TOTAL_DEMAND_T4"].mean()))
# print("thursday_demand: ",demand_thu)
demand_fri = round((store["TOTAL_DEMAND_T5"].mean()))
# print("friday_demand: ",demand_fri)
demand_sat = round((store["TOTAL_DEMAND_T6"].mean()))
# print("saturday_demand: ",demand_sat)
demand_sun = round((store["TOTAL_DEMAND_T7"].mean()))
# print("sunday_demand: ",demand_sun)

##########################################################################################################################################
#STEP3: Extact news from Portal in same duration as of Dataset
##########################################################################################################################################
# start = datetime.date(2013,1,1)
# end = datetime.date(2015,12,31)
start = start_date
end = end_date
# print('Start date: ' + str(start))
# print('End date: ' + str(end))

number_of_months = [x.split(' ') for x in pd.date_range(start, end, freq='MS').strftime("%Y %m").tolist()]
number_of_months

def send_request(date):
#     request to the NYT Archive API for a specific date.'''
    base_url = 'https://api.nytimes.com/svc/archive/v1'
    
    url = base_url + '/' + date[0] + '/' + date[1].lstrip('0') + '.json?api-key=' + API_KEY
#     print("url: ",url)
    try:
        response = requests.get(url, verify=False).json()
        print("response: ",response)
    except Exception:
        return None
    time.sleep(6)
    return response


def is_valid(article, date):
#     check if article is in date range and if it has a headline.
    is_in_date_range = date > start and date < end
    has_headline = type(article['headline']) == dict and 'main' in article['headline'].keys()
    return is_in_date_range and has_headline


def parse_response(response):
# Parses and returns response as pandas dataframe.
    data = {'headline': [],  
        'date': [], 
        'doc_type': [],
        'material_type': [],
        'section': [],
        'keywords': []}
    
    articles = response['response']['docs'] 
    for article in articles: # For each article, ensuring it falls within date range
        date = dateutil.parser.parse(article['pub_date']).date()
        if is_valid(article, date):
            data['date'].append(date)
            data['headline'].append(article['headline']['main']) 
            if 'section' in article:
                data['section'].append(article['section_name'])
            else:
                data['section'].append(None)
            data['doc_type'].append(article['document_type'])
            if 'type_of_material' in article: 
                data['material_type'].append(article['type_of_material'])
            else:
                data['material_type'].append(None)
            keywords = [keyword['value'] for keyword in article['keywords'] if keyword['name'] == 'subject']
            data['keywords'].append(keywords)
    return pd.DataFrame(data) 


def get_data(dates):
#     Sends and parses request/response to/from NYT Archive API for specific dates.
    total = 0
#     print('Date range: ' + str(dates[0]) + ' to ' + str(dates[-1]))
    if not os.path.exists('headlines'):
        os.mkdir('headlines')
    for date in dates:
#         print('Working on ' + str(date) + '...')
        csv_path = 'headlines/' + date[0] + '-' + date[1] + '.csv'
        if not os.path.exists(csv_path): # If we don't have data for a given month 
            response = send_request(date)
            if response is not None:
                df = parse_response(response)
                total += len(df)
                df.to_csv(csv_path, index=False)
                print('Saving ' + csv_path + '...')
#     print('Number of articles collected: ' + str(total))
    
get_data(number_of_months)

news = pd.read_csv('headlines/2013-04.csv', sep=',')
news.head(10000)

configs = Properties()
with open('keywords.properties', 'rb') as read_prop:
    configs.load(read_prop)
    
prop_view = configs.items()
high = ""
medium = ""
low = ""
for item in prop_view:
    if ((item[0]) == "high"):
        high = item[1].data
    elif ((item[0]) == "medium"):
        medium = item[1].data
    elif((item[0]) == "low"):
        low = item[1].data
# print(high)
# print(medium)
# print(low)


alert_level = ""
alert_text_high = news['headline'].str.contains(high,case=False)
alert_text_medium = news['headline'].str.contains(medium,case=False)
alert_text_low = news['headline'].str.contains(low,case=False)

if(news[alert_text_high].empty == False):
    alert_level = "high"
elif(news[alert_text_medium].empty == False):
    alert_level = "medium"
elif(news[alert_text_low].empty == False):
    alert_level = "low"

##########################################################################################################################################
#STEP4: Adjust Demand Levels as per Alerts extracted from NEWS
##########################################################################################################################################

configs = Properties()
with open('keywords.properties', 'rb') as read_prop:
    configs.load(read_prop)
    
prop_view = configs.items()

high_demand_increase = ""
medium_demand_increase = ""
low_demand_increase = ""

for item in prop_view:
    if ((item[0]) == "high_demand_increase"):
        high_demand_increase = item[1].data
    elif ((item[0]) == "medium_demand_increase"):
        medium_demand_increase = item[1].data
    elif((item[0]) == "low_demand_increase"):
        low_demand_increase = item[1].data

# print("high_demand_increase: ",high_demand_increase)
# print("medium_demand_increase: ",medium_demand_increase)
# print("low_demand_increase: ", low_demand_increase)

def adjusted_demand(day, percent_increase):
    demand_adjusted = int(round((int(day) +  (int(day) *  (int(percent_increase)/100))),0))
    return demand_adjusted

if (alert_level == "high"):
    demand_mon = adjusted_demand(demand_mon,high_demand_increase)
    demand_tue = adjusted_demand(demand_tue,high_demand_increase)
    demand_wed = adjusted_demand(demand_wed,high_demand_increase)
    demand_thu = adjusted_demand(demand_thu,high_demand_increase)
    demand_fri = adjusted_demand(demand_fri,high_demand_increase)
    demand_sat = adjusted_demand(demand_sat,high_demand_increase)
    demand_sun = adjusted_demand(demand_sun,high_demand_increase)
elif(alert_level == "medium"):
    demand_mon = adjusted_demand(demand_mon,medium_demand_increase)
    demand_tue = adjusted_demand(demand_tue,medium_demand_increase)
    demand_wed = adjusted_demand(demand_wed,medium_demand_increase)
    demand_thu = adjusted_demand(demand_thu,medium_demand_increase)
    demand_fri = adjusted_demand(demand_fri,medium_demand_increase)
    demand_sat = adjusted_demand(demand_sat,medium_demand_increase)
    demand_sun = adjusted_demand(demand_sun,medium_demand_increase)
elif(alert_level == "low"):
    demand_mon = adjusted_demand(demand_mon,low_demand_increase)
    demand_tue = adjusted_demand(demand_tue,low_demand_increase)
    demand_wed = adjusted_demand(demand_wed,low_demand_increase)
    demand_thu = adjusted_demand(demand_thu,low_demand_increase)
    demand_fri = adjusted_demand(demand_fri,low_demand_increase)
    demand_sat = adjusted_demand(demand_sat,low_demand_increase)
    demand_sun = adjusted_demand(demand_sun,low_demand_increase)
    
#STEP 5: Adjusted Demand as per News Alerts
print("monday's demand: ",demand_mon)
print("tuesday's demand: ",demand_tue)
print("wednesday's demand: ",demand_wed)
print("thursday's demand: ",demand_thu)
print("friday's demand: ",demand_fri)
print("saturday's demand: ",demand_sat)


#STEP 6: Scale Demand as per Inventory Size
print("scaled demand--->")
print("monday's demand: ",str(int(demand_mon/10)))
print("tuesday's demand: ",str(int(demand_tue/10)))
print("wednesday's demand: ",str(int(demand_wed/10)))
print("thursday's demand: ",str(int(demand_thu/10)))
print("friday's demand: ",str(int(demand_fri/10)))
print("saturday's demand: ",str(int(demand_sat/10)))


#END: DEMAND CODE#

day_mapping = {0:'mon',1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'}

class Env():

    def __init__(self):
        self.action_space = spaces.Discrete(m+1)
        self.inventory = np.random.choice(np.arange(0,m+1))       
        self.day = np.random.choice((0,1,2,3,4,5,6))
        self.state = (self.inventory,self.day)
        self.storeFlag = storeFlag
        self.IsStore = IsStore

        # Start the first round
        self.reset()

    def demand(self, day):
    	if day == 0:
    		return demand_mon
    	elif day == 1:
    		return demand_tue
    	elif day == 2:
    		return demand_wed
    	elif day == 3:
    		return demand_thu
    	elif day == 4:
    		return demand_fri
    	elif day == 5:
    		return demand_sat  
    	else:
    		return demand_sun
        

    def transition(self, x_t_1, a_t_1, d_t):

        if x_t_1[1] <6:
        	next_day = x_t_1[1]+1
        else:
        	next_day = 0    

        stock_after_sales = max(x_t_1[0] - d_t, 0)	#first this is calculated because this cannot go below 0
        stock_EOD = min(stock_after_sales + a_t_1,m)	#this is calculated second because this is added after the demand has been satisfied
        
        #note that state includes the order which was just delivered
        return (stock_EOD, next_day) 

    def reward(self, x_t_1, a_t_1, d_t):
        # x_t_1 = state(today-1), x_t = state(today)    #x_t_1 is the first element of the state tuple
        #Similarly for a = action and d = demand


        #1. EXPECTED INCOME
        expected_income = p * min(d_t,x_t_1)      #quantity sold=d,i.e.,demand. But if d>x, then quantity sold=x,i.e.,stock from last night
        
        #2. ORDER COST
        fixed_order_cost = K * (a_t_1 > 0)
        variable_order_cost = c
        order_cost = fixed_order_cost + variable_order_cost * a_t_1
        
        #3. HOLDING COST
        holding_cost = h * x_t_1
                
        #4. OPPORTUNITY COST
        actual_demand = d_t
        demand_satisfied = x_t_1 #because in d>x, we can only sell x
        profit = p - c
        #profit = 0.5
        opportunity_cost = profit * (actual_demand - demand_satisfied) * (actual_demand>demand_satisfied)
               
        #5. RETURN COST
        stock_after_sales = x_t_1 - d_t
        stock_arrived = a_t_1
        return_cost = R * (stock_after_sales + stock_arrived > m)    #can't use x_t_1 directly because it will be cut off at m

        #6. MONEY BACK
        stock_after_sales = x_t_1 - d_t
        stock_arrived = a_t_1
        money_back = c * (stock_after_sales + stock_arrived - m)  * (stock_after_sales + stock_arrived > m)

        r = expected_income - order_cost - holding_cost - opportunity_cost - return_cost + money_back
        return r

    def initial_step(self, state, action):
        assert self.action_space.contains(action)     #to check that action is a discrete value less than m
        obs = state

        if state[1]<6:
            demand = self.demand(state[1]+1)    
        else:
            demand = self.demand(0)        

        obs2 = self.transition(obs, action, demand)       #next_state

        return obs2



    def step(self, x_t_1, a_t_1):   
#         print("DEBUG..............self.action_space>:",self.action_space)
#         print("DEBUG..............a_t_1>:",a_t_1)
        assert self.action_space.contains(a_t_1)     #to check that action is a discrete value less than m
        obs = x_t_1             #at the beginning, state is picked up from the contructor. 


        if x_t_1[1]<6:
            d_t = self.demand(x_t_1[1]+1)    
        else:
            d_t = self.demand(0)

        obs2 = self.transition(obs, a_t_1, d_t)       #next_state


        reward = self.reward(x_t_1[0], a_t_1,  d_t)

        return obs2, reward



    def reset(self):
        return self.state
