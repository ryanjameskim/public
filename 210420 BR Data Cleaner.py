from __future__ import unicode_literals
import io
from simplified_scrapy import SimplifiedDoc, utils, req
import os
import pandas as pd

rawpath = r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210331' + '\\'
outpath = r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210331\Holdings' + '\\'
if not os.path.exists(outpath):
    os.makedirs(outpath)

all_files = []

for root, dirs, files in os.walk(rawpath):
    for name in files:
        if '.xls' in name:
            all_files.append(''.join([root, name]))

for file in all_files:
    file_name = ''.join(file.split('\\')[-1:]).split('.')[0]
    save_name = ''.join(outpath + file_name + '_holdings.csv')
    data = io.open(file, 'r', encoding='utf-8').read()
    doc = SimplifiedDoc(data)
    #convert from xml
    worksheets = doc.selects('ss:Worksheet') # Get all Worksheets
    for worksheet in worksheets:
       if worksheet['ss:Name'] == 'Holdings':
            rows = worksheet.selects('ss:Row').selects('ss:Cell>text()') # Get all rows
            #utils.save2csv(save_name, rows) # Save data to cs
    #convert into DataFrame
    df = pd.DataFrame(rows)[8:]                 #convert data
    df.columns = pd.DataFrame(rows).iloc[7]     #collect headers
    df = df.reset_index(drop=True)              #reset index
    df.columns.name = file_name                 #change column name
    #save
    df.to_csv(save_name)


