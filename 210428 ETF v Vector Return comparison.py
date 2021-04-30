import numpy as np
import pandas as pd
import io
import yfinance as yf
from scipy.spatial.distance import cosine
import difflib
import datetime
from pandas.tseries.offsets import BDay

#import data
df = pd.read_pickle(
    r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210331\Holdings_small\Aggregate.pkl')
#df = pd.read_csv('/content/Aggregate.csv')
df = df.sort_values(['ETF', 'Weight (%)'], ascending=[True, False])

index = pd.read_table(r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210427 metadata 100 epoch.tsv', encoding='utf-8', header=None)
weights = pd.read_table(r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210427 weights 100 epoch.tsv', encoding='utf-8', header=None)
weights.index = index[0].values

del index

#get individual ETF names
etf_names = df['ETF'].value_counts().keys().to_numpy()

#initialize new etf weight sum vector
etf_vectors = pd.DataFrame()

for etf in etf_names:                       #for each etf
    etf_vector_sum = np.zeros((weights.shape[1]),)  #initialize net sum
    etf_df = df[df['ETF'] == etf]           #restrict by individual etf
    for ticker in etf_df['Ticker']:         #for every ticker
        if etf_df[df['Ticker'] == ticker].shape[0] != 1:   #check to make sure no dupes
            continue
        etf_vector_sum += weights.loc[ticker] * (etf_df[df['Ticker'] == ticker]['Weight (%)'] / 100).values #sum weighted sum of individual stock vectors
    etf_vector_sum.rename(etf, inplace=True)    
    etf_vectors = etf_vectors.append(etf_vector_sum)

#Save
etf_vectors.to_csv(r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210427 weighted etf vectors.csv')

#For google embedding projector
#For saving vectors and metadata
out_v = io.open(r'C:\Users\ryanj\Google Drive\ML\Ishares Data\etf_vectors.tsv', 'w', encoding='utf-8')
out_m = io.open(r'C:\Users\ryanj\Google Drive\ML\Ishares Data\etf_metadata.tsv', 'w', encoding='utf-8')

for etf in etf_vectors.index:
  out_v.write('\t'.join([str(x) for x in etf_vectors.loc[etf]]) + "\n")
  out_m.write(etf + "\n")

out_v.close()
out_m.close()

#get ETF Tickers
with open(r'C:\Users\ryanj\Downloads\BR ticker and name.txt') as file:
    etf_to_ticker = file.readlines()

etf_to_ticker = etf_to_ticker[0::10]  #Every tenth line
etf_to_ticker = [[item.split(' ')[0], ('-'.join(item.split(' ')[1:]).rstrip())] for
                                            item in etf_to_ticker]  #split into two items per line
etf_to_ticker = pd.DataFrame(etf_to_ticker, columns=['ETF Ticker', 'ETF Name'])  #change to DataFrame
etf_to_ticker['ETF Name'] = etf_to_ticker['ETF Name'].replace('Â®', '') #remove extra character

etf_tickers = pd.DataFrame()

for etf_name in etf_vectors.index:
    match = difflib.get_close_matches(etf_name, etf_to_ticker['ETF Name'], 1)
    #print(f'{etf_name} : {match}')
    etf_tickers.loc[etf_name, 'Ticker'] = etf_to_ticker[etf_to_ticker['ETF Name']
                                                        == match[0]]['ETF Ticker'].to_numpy().item()


#find closest cosine neighbors
top_k_tickers = 15
top_k_etfs = 5

#for labeling
column_names = [f'Top Stock {i+1}' for i in range(top_k_tickers)] + [f'Top ETF {i+1}' for i in range(top_k_etfs)]

nearest_neighbors = pd.DataFrame(columns = column_names)
idx_to_tick = {i: tick for i, tick in enumerate(weights.index)}
idx_to_etf = {i: etf.item() for i, etf in enumerate(etf_tickers.values)}

for etf_idx, etf_vector in enumerate(etf_vectors.values):
    tick_results = np.zeros((weights.shape[0], 1))
    etf_results = np.zeros((etf_vectors.shape[0], 1))
    for i, ticker in enumerate(weights.index):   #find cosine similarity of all stocks and find top k
        tick_results[i] = 1 - cosine(etf_vector, weights.loc[ticker])
    nearest_ticks = (-tick_results.reshape(-1)).argsort()[0:top_k_tickers] 
                    #gets negative of the cosine scores, lists the indexes of the most negative to positve scores
                    #then indexing the argsort lists the smallest numbers
    for i, test_etf in enumerate(etf_vectors.index):  #repeat for etfs
        etf_results[i] = 1 - cosine(etf_vector, etf_vectors.loc[test_etf])
    nearest_etfs = (-etf_results.reshape(-1)).argsort()[1:top_k_etfs + 1] #+1 to skip itself
    final_results = np.array([idx_to_tick[idx] for idx in nearest_ticks] + [idx_to_etf[idx] for idx in nearest_etfs])
    nearest_neighbors.loc[idx_to_etf[etf_idx],:] = final_results    

nearest_neighbors.to_csv(r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210427 nearest_neighbors.csv')


#Get return data
def get_return_data (nearest_neighbors, start='2020-12-31', end='2021-04-29'):
    columns =  (['ETF CumReturn', 'Blended CumReturn'] +     #create categories of info we want, including 
                [f'Top ETF CumReturn {i + 1}' for i in range(top_k_etfs)] +
                [f'Top ETF Ticker {i + 1}' for i in range(top_k_etfs)] +
                ['Blended STD'] +
                [f'Top ETF STD {i}' for i in range(top_k_etfs + 1)]) #+1 because need STD for loop etf too
    results = pd.DataFrame(columns = columns)
    for i, etf in enumerate(nearest_neighbors.index):
        print(f'Calculating Returns for: {i}. {etf}')
        loop_etf_name = nearest_neighbors.loc[etf].name
        yf_tickers = ' '.join(nearest_neighbors.loc[etf][:(top_k_tickers + top_k_etfs)].values) #put all nearest stock and etf tickers in a string
        yf_tickers = f'{loop_etf_name} ' + yf_tickers  #add ticker of etf to front
        data = yf.download(yf_tickers, start=start, end=end)  #get return data
        data = data['Adj Close']   #disregard other information
        try:
            cum_returns = (data.loc[(pd.to_datetime(end) - BDay(1)).strftime('%Y-%m-%d')]-  #converting end date to datetime then subtracting one day
                            data.loc[start]).div(data.loc[start])  #calculate simple cum return
        except:
            continue
        #to calculate ETF CumReturn and Blended CumReturn
        cum_returns = cum_returns.reindex(index = yf_tickers.split())  #reorder to original nearests
        stock_weights = [1/top_k_tickers] * top_k_tickers   #for future weight featuring
        blended_cum_return = np.sum(cum_returns[1:top_k_tickers+1] * stock_weights)  #skips loop etf returns
        #to calculate 'Blended STD'
        blended_returns = np.sum(stock_weights * data.pct_change().iloc[:, :top_k_tickers], axis=1)
        blended_std = blended_returns[1:].std()
        results.loc[loop_etf_name, 'Blended STD'] = blended_std
        #to calculate nearest ETFs STD
        etfs_stds = nearest_neighbors.loc[etf][-top_k_etfs:].to_list()   #get tickers for top similar etfs
        etfs_stds = [etf] + etfs_stds    #add loop etf to front ('Top ETF 0')
        etfs_stds = data.pct_change().std()[etfs_stds].to_list() #get actual STDs
        #concatenate together
        row = [cum_returns[0], blended_cum_return]  #ETF CumReturn and Blended CumReturn
        row += cum_returns[-top_k_etfs:].to_list()  #Top ETF CumReturn
        row += nearest_neighbors.iloc[i, -top_k_etfs:].to_list() #Top ETF Ticker
        row += [blended_std]   #Blended STD
        row += etfs_stds   #Already a list
        results.loc[loop_etf_name, :] = row
    return results

#Explore results
filtered_results = results[(results['Blended STD'] < results['Top ETF STD 0']) &
                            (results['ETF CumReturn'] < results['Blended CumReturn'])] #find all etfs with higher blended STD and lower cum return
filtered_results = filtered_results.loc[:, ['ETF CumReturn', 'Blended CumReturn', 'Top ETF STD 0', 'Blended STD']]
filtered_names = [etf_to_ticker[etf_to_ticker['ETF Ticker'] == ticker]['ETF Name'].values[0] for ticker in filtered_results.index]
filtered_results.insert(loc=0, column='Name', value=filtered_names)
filtered_results = filtered_results.rename(columns={"Blended CumReturn": "Blended Stock CumReturn", "Top ETF STD 0": "ETF STD"})
filtered_results


filtered_results['STD Difference'] = filtered_results['ETF STD'] - filtered_results['Blended STD']
filtered_results.sort_values(by=['STD Difference'], ascending=False)

nearest_neighbors.loc[filtered_results.index]
