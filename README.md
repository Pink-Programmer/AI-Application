# AI-Application
Group Assignment: Stock market prediction


In this project, artificial intelligence is used to predict the stock market. The goal is to train an artificial 
intelligence to make the decision whether to sell the stock or not. With the decisions made, it is 
desired that no losses are made on the sale. The training data is based on a dataset with a total of 
approx. 18000 data. The data such as date, open, high, low, close, volume, dividends and stock splits 
of each day of the share are shown in the dataset. The training will be done with the Random Forest 
technique. Depending on the result of the training, if necessary, the dataset will be expanded or the 
model will be compared with another technique in the field of artificial intelligence.


I.	INTRODUCTION


The stock market is a dynamic system that presents the economic well-being of companies. Investors and market participants tend to make decisions based on the traditional econometric models and technical analysis indicators. Due to the complexity of the past performance of the stock, the accuracy of the traditional method is not highly reliable. This uncertainty and the unpredictability of the stock market means investors might face serious financial risks in optimizing their investment strategy. 61% of Americans own stock because they are more open to buying stock (Caporal, 2023). There is a need for a sophisticated and data-driven solution that can provide more reliable and actionable predictions.

In this project, an AI model is proposed to predict the stock market trend and to capture the intricate patterns and subtle relationships within financial data. An ensemble learning model utilizing Random Forest and Decision Tree algorithms to analyze historical stock market data and price movements will be developed. The model will provide labels to indicate the trend of the stocks. The user could make better decisions based on our model. 

By providing more accurate and reliable predictions, this model can help investors make informed investment decisions, optimize portfolio management strategies, and mitigate financial risks. Financial institutions can benefit from improved risk assessment and asset allocation, leading to enhanced profitability and stability. Moreover, a more precise understanding of market trends can contribute to overall market efficiency and investor confidence. 


II.	DATASET


This chapter describes the datasets that were used for the realization of this project.

A.	TESLA Stock Dataset (TSLA)

The TESLA dataset includes approx. 10 years of the TSLA stock data from June 29, 2010 - February 3, 2020. The original dataset was slightly modified removing Adj Close for consistency reasons. Hence, the features that are included in the modified dataset are as follows: Date, Open, High, Low, Close, Volume, and Rate of Change. The prices are based on USD.

B.	SAMSUNG Stock Dataset (SSNLF)

The SAMSUNG dataset includes decades of SSNLF stock data from January 4, 2000 - May 23, 2022. The prices are also based on USD. It’s also modified as described in the TESLA Stock Dataset description.

C.	TWITTER Stock Dataset (TWTR)

The TWITTER dataset was chosen to be used because of its interesting history. Due to Elon Musk’s purchase of Twitter back in 2022, it became a private company, was delisted from the New York exchange, and is no longer in the stock market. This dataset contains Twitter’s history on the stock market from November 7, 2013 - October 27, 2022. The prices are also based on USD. It’s also modified as described in the TESLA Stock Dataset description.





