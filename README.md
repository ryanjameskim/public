# Ryan James Kim public portfolio
Examples of adaptations and code

TPT2017CNN.ipynb

* Adaptation of Tsantekidis, Passalis, Tefas, et al 2017 with few addition of Dropout layers

* skills: Pandas, Tensorflow, Keras, Convolutional Neural Networks


Reduced_Scale_LTSM_only.ipynb

* Limit order book time series analysis using double LSTM to predict 1 minute future average price (categorical)
* skills: Pandas, Data generator, Keras, Tensorflow, NLP architectures


Kaggle_Tabular_Mar_2021.ipynb

* Kaggle Tabular Playground Series - Mar 2021

* Complete ensemble (CatBoost, LightGBM) analysis of pre-cleaned database

* Adapted mainly from @AndresHG



210331 Selenium BR ishares downloader.py

* Scrapes all 300+ of holding data from Blackrock ishares website

* packages: selenium

* skills: web scraping



210420 BR Data Cleaner.py

* Converts 'corrupt' XML XLS downloaded sheets into clean csv tables of holdings


210421 BR Holdings Filter.py

* Reduces cleaned CSV files down to US equity only table


210422 ETF2Vec Keras Implementation.py

* concept with individual stock holdings from all of BlackRock's US Public Equity ETF holdings in order to dimensionalize 'ETF stock selection criteria'
into vector form.

* Proximity in holding size within an ETF is taken as primary context with a Zipf similiarity negative sample selection.

* skills: _keras_ functional API, _pandas_, deep learning, neural networks


210427 ETF2Vec Batch Implementation.py

* Repeated the implementation to run faster with batching and only one cosine similarity test (using final weight embeddings) at end.

* Added embedding projector visualization

* skills: _keras_ functional API, batching, tensorflow embedding projector
