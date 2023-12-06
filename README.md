# AI-Application
Group Assignment: Stock market prediction


In this project, artificial intelligence is used to predict the stock market. The goal is to train an artificial 
intelligence to make the decision whether to sell the stock or not. With the decisions made, it is 
desired that no losses are made on the sale. The training data is based on a dataset with a total of 
approx. 18000 data. The data such as date, open, high, low, close, volume, dividends and stock splits 
of each day of the share are shown in the dataset. The training will be done with the Random Forest 
technique. Depending on the result of the training, if necessary, the dataset will be expanded or the 
model will be compared with another technique in the field of artificial intelligence.

--------
I.	INTRODUCTION


The stock market is a dynamic system that presents the economic well-being of companies. Investors and market participants tend to make decisions based on the traditional econometric models and technical analysis indicators. Due to the complexity of the past performance of the stock, the accuracy of the traditional method is not highly reliable. This uncertainty and the unpredictability of the stock market means investors might face serious financial risks in optimizing their investment strategy. 61% of Americans own stock because they are more open to buying stock (Caporal, 2023). There is a need for a sophisticated and data-driven solution that can provide more reliable and actionable predictions.

In this project, an AI model is proposed to predict the stock market trend and to capture the intricate patterns and subtle relationships within financial data. An ensemble learning model utilizing Random Forest and Decision Tree algorithms to analyze historical stock market data and price movements will be developed. The model will provide labels to indicate the trend of the stocks. The user could make better decisions based on our model. 

By providing more accurate and reliable predictions, this model can help investors make informed investment decisions, optimize portfolio management strategies, and mitigate financial risks. Financial institutions can benefit from improved risk assessment and asset allocation, leading to enhanced profitability and stability. Moreover, a more precise understanding of market trends can contribute to overall market efficiency and investor confidence. 

---------------------------------------------------------------------------------
II.	DATASET


This chapter describes the datasets that were used for the realization of this project.

A.	TESLA Stock Dataset (TSLA)

The TESLA dataset includes approx. 10 years of the TSLA stock data from June 29, 2010 - February 3, 2020. The original dataset was slightly modified removing Adj Close for consistency reasons. Hence, the features that are included in the modified dataset are as follows: Date, Open, High, Low, Close, Volume, and Rate of Change. The prices are based on USD.

B.	SAMSUNG Stock Dataset (SSNLF)

The SAMSUNG dataset includes decades of SSNLF stock data from January 4, 2000 - May 23, 2022. The prices are also based on USD. It’s also modified as described in the TESLA Stock Dataset description.

C.	TWITTER Stock Dataset (TWTR)

The TWITTER dataset was chosen to be used because of its interesting history. Due to Elon Musk’s purchase of Twitter back in 2022, it became a private company, was delisted from the New York exchange, and is no longer in the stock market. This dataset contains Twitter’s history on the stock market from November 7, 2013 - October 27, 2022. The prices are also based on USD. It’s also modified as described in the TESLA Stock Dataset description.

--------------------------------------------
III.	METHODOLOGY

The project focuses on the prediction of stock prices with the help of Decision Tree and Random Forest. The selection of these artificial intelligence methods is based on the analysis of the dataset and the conclusion which model is suitable in this case. This is discussed in chapter IV in more detail. This chapter refers to the general methodology of implementation.


A.	Original Dataset

These are the features of the original dataset:
Date - Includes the year, month, and day in that order (YYYY-MM-DD)

Open - This is the opening price of the day.

High - The highest price of the day.

Low - The lowest price of the day.

Close - This is the closing price of the day.

Volume – The total number of shares traded in the day.

Adj Close - This means the adjusted closing price of the day. It’s adjusted to better reflect the stock’s value after anything such as corporate actions would affect the stock price after the market closes. However, this was taken out for consistency reasons.

B.	Modified Dataset and Features

The dataset is extended with modified data to make the dataset bigger and create more data for the AI to train on. With the additional created data, possible patterns and relationship between different information will be recognized by the AI. In the following it is described how the data is going to be modified and extended.

Slope/rate of change - It is calculated using the formula below. The purpose is to train the AI to consider the influence that stock prices on the days prior have on that of the present day. This helps the AI recognize certain patterns happening in the given datasets. For instance, when calculating the rate of change for the opening price, y2 would be the opening price of the current day, y1 would be the previous day’s opening price, and x2 - x1 would be the number of days. The slope is calculated as follows:

slope=(y_2-y_1)/(x_2-x_1 )

This calculation would be repeated based on the fixed number of days. The following figure illustrates the calculation of the slope.

![image](https://github.com/Pink-Programmer/AI-Application/assets/148310919/fe18d035-311a-4452-8a4f-bf35a3c889f6)

Figure 1 - Calculation of the slope

Average slope - Using the outcome of the calculation of the slopes, the average of those single slopes can be calculated, allowing one to determine whether the price will rise or fall. If the number is positive, the value increases. If the number is negative, the value falls.

Number of rises and falls - In this new feature, the individual calculated slopes are taken into account again. The number of slopes with a positive or negative sign are counted. This means that the number of times the share price rises or falls in a certain time frame is counted. In this way, it is intended that the artificial intelligence in this work may be able to recognize patterns of behavior on the stock market in order to predict the further course. In the following illustration, the rising and falling of the price is shown with arrows.

![image](https://github.com/Pink-Programmer/AI-Application/assets/148310919/7d0bd5ee-3c3b-4d2f-9b3e-28571ab5d420)

Figure 2 - Stock price changes based on slope

Difference between High and Low - Here the difference between the High and Low prices are calculated. This is done for the current day and all past days within a certain time frame. The difference shows how much the price fluctuates within a day. This is intended to illustrate further behavior in the stock market in order to identify certain patterns.

Average difference between High and Low - The average difference between High and Low is calculated. This summarizes whether the difference in the days is generally high.

Difference between Open and Close – Here the difference between the Open and Close prices are calculated. This gives us more information on the stability of the price. This is intended to provide further insight that may be useful in exploring patterns for artificial intelligence.

Average difference between Open and Close - Similar to the High and Low values, the average difference between Open and Close is also calculated here over a specific time window.

C.	Creating a label

The labels 1 and 0 are used to predict a rising or falling price, respectively. The labels are determined on the basis of the average rate of the low value. The low value, which represents the lowest price on a day, is used to forecast based on a worst-case scenario. To check the quality of the labels, the stock price performance is visualized with 3 data points each for a fall and rise in the stock price. The visualization is shown in the figure below. The green dots represent the rise and the red dots the fall of the share price.

<img width="246" alt="image" src="https://github.com/Pink-Programmer/AI-Application/assets/148310919/94784f74-5888-4f4d-a45e-c3b7463f9ab9">

Figure 3 – Visualization of the labels

D.	Distribution of the dataset

In terms of training and testing, the modified dataset will be split up 70% of the data will be used for training and 30% used for testing. The amount of training data is distributed as such because there needs to be as much data as possible to train the AI. On the other hand, there must be enough testing data available to ensure an accurate testing result.

E.	Implementation of the code for modifying and labelling the dataset

Several functions were declared for the realization of the project and subsequently applied. Their use is described below. The exact code can be found in the Jupyter notebook file.

def combine_data(trade_datas):

This function is used to merge different datasets. A list of DataFrames is passed, which are combined into one at the end. Since datasets from different companies are used in this project, it is necessary to merge them in a standardized way.

def modify_data(trade_data,t_previous_days, t_label_days, comprimize_data):

This function is used to modify a dataset. The purpose of modifying the dataset is to extract more information from the given dataset and use it as an additional feature. The new features are used, for example, to recognize temporal patterns. The label is also created in this function. To execute the functions, the dataset is required as well as the number of days immediately before and after the respective data points that are to be analyzed. comprimize_data is a boolean which is used to determine whether all modified features should be used or a compromised version.

def plot_label_visualization(data):

This function displays the result of the automatically created labels. The progression of a price is displayed graphically. For each label, 3 data points are selected at random and displayed in the graph in the form of dots in different colors. A vertical line with the respective color is then drawn for each data point, making it clear which point in time is being considered for a forecast.

F.	Function of a Decision Tree

A Decision Tree classifier is a machine learning algorithm that can be used for classification purposes. The model resembles a tree that recursively splits the dataset into subsets decided by the values of parameters, that are regularly adjusted when training the model to achieve a higher level of accuracy. The following image illustrates the function of a Decision Tree:

<img width="233" alt="image" src="https://github.com/Pink-Programmer/AI-Application/assets/148310919/456a947d-60e9-48fa-88c6-26c6ef6e97b9">

Figure 4 - The structure of a Decision Tree

A Decision Tree consists of the following basic components:

     Root node – starting (top) node with all data points
     
     Decision node(s) – nodes with condition(s) to split data
     
     Leaf nodes – nodes with ideally only data points of a   
                           single class

Starting from the root node, the Decision Tree classifier decides on the most optimal way to split the dataset into subsets. There are conditions in the decision nodes for splitting the data. Based on the conditions, the data is distributed to two or more branches. For example, a threshold can be used as a possible condition with which the features in the dataset are compared. The value of the threshold is random at the beginning and is adjusted during training. Optimal conditions are achieved during training with the highest possible information gain.
