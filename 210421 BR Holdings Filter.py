import os
import pandas as pd

holdings_csvs = r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210331\Holdings' + '\\'
outpath = r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210331\Holdings_small' + '\\'
if not os.path.exists(outpath):
    os.makedirs(outpath)

all_files = []

for root, dirs, files in os.walk(holdings_csvs):
    for name in files:
        if '.csv' in name:
            all_files.append(''.join([root, name]))

final_df = pd.DataFrame()

for i, file in enumerate(all_files):
    df = pd.read_csv(file)
    file_name = ''.join(file.split('\\')[-1:]).split('.')[0].split('_')[0]
    #Progress counter
    print(f'Processing {i}. {file_name}')
    #Restrictions only Equity and Location
    df = df[(df['Asset Class'] == 'Equity') & (df['Location'] == 'United States')]
    #Keeping only these columns
    keep_columns = ['Ticker', 'Name', 'Sector', 'Weight (%)', 'Market Value']
    #skip rest if not enough relevant entries
    if len(df) < 10:
        continue               
    #skip if keep columns does not exist
    for keep_column in keep_columns:
        if keep_column not in df.columns:
            continue
    #Reduce df
    df = df[keep_columns]    #filters out the relevant columns
    df.insert(loc=0, column='ETF', value=file_name)   #add name of the ETF
    df = df.reset_index(drop=True)  #resets index
    final_df = final_df.append(df, ignore_index=True)  #copies to final
    df.to_pickle(outpath + file_name + '_small.pkl')
    print(df.head())

final_df.to_pickle(outpath + 'Aggregate.pkl')



