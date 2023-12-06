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
---

The stock market is a dynamic system that presents the economic well-being of companies. Investors and market participants tend to make decisions based on the traditional econometric models and technical analysis indicators. Due to the complexity of the past performance of the stock, the accuracy of the traditional method is not highly reliable. This uncertainty and the unpredictability of the stock market means investors might face serious financial risks in optimizing their investment strategy. 61% of Americans own stock because they are more open to buying stock (Caporal, 2023). There is a need for a sophisticated and data-driven solution that can provide more reliable and actionable predictions.

In this project, an AI model is proposed to predict the stock market trend and to capture the intricate patterns and subtle relationships within financial data. An ensemble learning model utilizing Random Forest and Decision Tree algorithms to analyze historical stock market data and price movements will be developed. The model will provide labels to indicate the trend of the stocks. The user could make better decisions based on our model. 

By providing more accurate and reliable predictions, this model can help investors make informed investment decisions, optimize portfolio management strategies, and mitigate financial risks. Financial institutions can benefit from improved risk assessment and asset allocation, leading to enhanced profitability and stability. Moreover, a more precise understanding of market trends can contribute to overall market efficiency and investor confidence. 


II.	DATASET
---

This chapter describes the datasets that were used for the realization of this project.

A.	TESLA Stock Dataset (TSLA)

The TESLA dataset includes approx. 10 years of the TSLA stock data from June 29, 2010 - February 3, 2020. The original dataset was slightly modified removing Adj Close for consistency reasons. Hence, the features that are included in the modified dataset are as follows: Date, Open, High, Low, Close, Volume, and Rate of Change. The prices are based on USD.

B.	SAMSUNG Stock Dataset (SSNLF)

The SAMSUNG dataset includes decades of SSNLF stock data from January 4, 2000 - May 23, 2022. The prices are also based on USD. It’s also modified as described in the TESLA Stock Dataset description.

C.	TWITTER Stock Dataset (TWTR)

The TWITTER dataset was chosen to be used because of its interesting history. Due to Elon Musk’s purchase of Twitter back in 2022, it became a private company, was delisted from the New York exchange, and is no longer in the stock market. This dataset contains Twitter’s history on the stock market from November 7, 2013 - October 27, 2022. The prices are also based on USD. It’s also modified as described in the TESLA Stock Dataset description.


III.	METHODOLOGY
---
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

---
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

Information gain is defined as the effectiveness of a splitting condition in reducing entropy or Gini impurity within the data subset. The information gain can be calculated by the changes in entropy or the Gini index when training the model.

Entropy is the measure of impurity of data points. The aim here is to achieve a low value. The lower the entropy value, the more homogeneous the data is after splitting. The entropy value is calculated as follows:

Entropy=- ∑_(i=1)^n▒〖p_i* log (p_i)〗  

p_i refers to the percentage of class i in the data subset at the node. Using the entropy determined, the information gain can be calculated as follows:

Information Gain= Entropy(parent)- ∑▒〖w_i* Entropy(〖child〗_i)〗

w_i refers to the weight of class i expressed as a fraction of the data subset at the node.

Compared to entropy, the Gini index concentrates on the probability that a randomly selected instance will be misclassified. The aim here is to achieve the lowest possible value. The lower the Gini index, the lower the probability that something will be classified incorrectly. The Gini index is calculated as follows:

Gini=1- ∑_(i=1)^j▒〖p_i〗^2 

The variable j is the total number of classes in the target variable and p_i refers to the percentage of class i in the data subset at the node. Similar to the use of entropy, the information gain can be calculated using the Gini index and the following formula:

Information Gain=Gini(parent)  - ∑▒〖w_i*Gini(〖child〗_i)〗

Once the best possible condition based on the highest information gain has been determined, the data is split based on this and the process is repeated for each new node. The process continues until the stop conditions are met. The stop conditions are described in the following:

    - All data points in a node belong to the same class
    
    - The maximum depth of the tree has been reached
    
    - The minimum number of data points in a node has been  
       reached.

As soon as the stop conditions are met, a leaf node is created instead of a decision node. The leaf node determines the class affiliation of the respective root in the Decision Tree.

After the Decision Tree has been created, a data point is classified based on the conditions of the decision nodes. Depending on the condition, a specific branch in the Decision Tree is followed. The leaf node that is reached at the end determines the predicted class. The following image shows how a class of a data point is determined.

<img width="246" alt="image" src="https://github.com/Pink-Programmer/AI-Application/assets/148310919/13a0d9c2-df45-4810-84af-4807b932d398">

Figure 5 - The process of prediction with a Decision Tree

G.	Function of a Random Forest

The Random Forest algorithm is an enhanced version of the Decision Tree classification model, making use of a collection of multiple Decision Trees. Decision Trees are highly sensitive to the training dataset and any modification to the training data can result in an entirely different Decision Tree. Random Forest can reduce the sensitivity towards the training data through generalization.

<img width="246" alt="image" src="https://github.com/Pink-Programmer/AI-Application/assets/148310919/e7d3c84d-1923-495c-973d-e4edee19a098">

Figure 6 - The function of a Random Forest

A Random Forest consists of several Decision Trees that are created independently of each other. Bootstrapping is used to train the trees. Bootstrapping is a process in which random data points are taken from the dataset for training for each of the trees, so that each of the trees is trained with different datasets, increasing the versatility of the forest. It can happen that the same data point occurs in several training datasets. It is also possible that the random selection means that certain data points do not appear in any tree in the training data. Bootstrapping prevents the use of the same dataset in the training process, reducing the sensitivities of trees towards the original dataset. Limiting the random feature selection reduces the correlation between trees, as trees are trained with a variety of decision nodes, hence increasing the variance in prediction outcomes.

To merge the predictions of all trees and create a single prediction of the Random Forest, an aggregation is performed. To make a prediction using Random Forest, the data point is passed through each tree to get prediction outputs from every tree. The final decision is then made through majority voting of all predicted outcomes. The structure of a Random Forest is shown in figure 6.

H.	Implementation of the code to create the model

In this project, the Decision Tree and Random Forest are both created with the library sklearn and programmed from scratch. As the models are each trained and compared with different parameters, functions are defined for the executions. The following functions have been created to summarize all the individual steps involved in creating the respective models. The general procedure of the functions is very similar. To execute the function, a previously modified dataset is provided. Firstly, the dataset is split into test and training data. Then the model is created and trained with the training data. At the end, the trained model is tested with the test data and a value for accuracy is created. At the end, the model, the predictions made during testing, the labels and the accuracy are returned. The declared functions of the respective models are presented in the following.

def run_decision_tree_classifier_by_sklearn(modified_ trade_data)

This function is about the creation of a Decision Tree classifier with the help of the module of the sklearn library. The following function by sklearn is used: DecisionTreeClassifier(max_depth=10,min_samples_split=5)

def run_random_forest_classifier_by_sklearn(modified_ trade_data):

This function is about the creation of a Random Forest classifier with the help of the module of the sklearn library. The following function by sklearn is used: RandomForestClassifier(n_estimators=10, random_state=42,max_depth=10, min_samples_split=5)

def run_decision_tree_from_scratch(modified_trade_ data)

This function involves the creation of a Decision Tree classifier whose algorithm was implemented within the scope of this project. The exact algorithm will be explained in more detail later.

def run_random_forest_from_scratch(modified_trade_ data)

This function involves the creation of a Random Forest classifier whose algorithm was implemented within the scope of this project. The exact algorithm will be explained in more detail later.

The implementation from scratch is described below, starting with the Decision Tree.

class Tree_Node():

The class Tree_Node represents a Node in a Decision Tree. It contains information about the splitting of the tree including the feature index, a threshold, and a subtree. Additionally, it contains the information gain resulting from the splitting of a tree. Furthermore, it contains the label if it is a leaf node.

class DecisionTree():

The DecisionTree class represents a Decision Tree classifier which is implemented from scratch in the purpose of this project. It contains the parameters like min_datas_branching and max_depth which describe how often a tree will be split and the maximum depth of the split. Furthermore, it contains the roots of the previous branches.

def create_tree(self, dataset, curr_depth=0)

The function create_tree builds a Decision Tree recursively based on the given dataset. It uses values like min_datas_branching and max_depth as conditional values to decide whether conditions are met to calculate the best split and build another subtree recursively. If the conditions aren't met, a leaf node will be created.

def get_best_branching(self, dataset, number_of_datas, number_of_features)

This method searches for the best split in a Decision Tree based on the dataset. It iterates through the features and their possible thresholds to calculate the best branching. 

It calculates the potential threshold for every feature, divides the dataset and calculates the information gain using whether the Gini index or the entropy. It updates the dictionary of the best branching based on the highest possible improvement in the information gain value.

def branch_tree(self, dataset, feature_index, threshold)

The function branch_tree divides a dataset based on a feature and a threshold. It creates two arrays in which the data smaller or bigger than the thresholds are separated.

def information_gain(self, parent, l_child, r_child, mode="entropy") 

The information gain function is used to calculate the information gain after a splitting of a branch. For the calculation it uses either the Gini index or the entropy, determined by the 'mode' parameter.

def entropy(self, y):   

With the entropy function, disorders and randomness within a set of labels can be calculated. By examining the different types of labels and quantifying how unpredictable they are, it calculates the probability.

def gini_index(self, y): 

With the function Gini index, similar to the entropy function, the impurity or disorder is going to be measured. By iterating through the labels, it calculates the Gini index based on the probability of the occurrence of each of the labels. After applying a specific mathematical formula, it returns the Gini index.

def calculate_leaf_value(self, label):

The function calculate_leaf_value detects the label of a leaf note. It predicts the value by doing a majority voting of the labels that the leaf contains. The most frequent label will be returned as its label. 

def fit(self, data, label):

With the fit function, the Decision Tree is going to be trained. For this, the dataset and the labels are merged before using the function create_tree to establish the nodes.

def predict(self, data):

The predict function is used to create predictions based on the given dataset. It iterates through each data point and creates a prediction with the function make_prediction function. 

def make_prediction(self, x, tree):

The make_prediction function is used within the predict function. It uses the Decision Tree structure and evaluates a single data point against the nodes of the tree. It compares the feature values with the nodes and finds the correct root through the tree. Finally, it finds the predicted label when reaching the leaf note and returns it.

The implementation of the Random Forest in scratch is now presented below. It should be mentioned that the Decision Tree programmed in scratch presented above is used to create the forest.

class RandomForest():

The class RandomForest is used to build a Random Forest model. It is implemented from scratch and creates Decision Trees that are, as described above, implemented inin scratch as well. The RandomForest class uses parameters like n_trees to describe the number of trees within the decision forest. Furthermore, it uses the parameters like max_depth and min_datas_branching which are necessary to create a Decision Tree. Furthermore, it contains a parameter which describes the number of features in the dataset and another which contains all trees in an array.

def fit(self, X, y):

The Random Forest gets trained by the function fit. It creates multiple Decision Trees for the forest. When creating the Decision Tree, it uses the functions that are explained in the Decision Tree class section. After the creation of the individual Decision Trees the function samples are going to be used to create a diverse subset for the input data of each tree. This way, the Random Forest model has a higher variability among a tree which leads to more robustness and a higher predictive performance.

def samples(self, X, y):

After each Decision Tree is created, the function samples create different training datasets for each tree. These datasets are created randomly out of the given dataset. This way, unique training datasets are created which leads to individual training of the trees.

def identify_most_common(self, y):

This function detects the most frequently occurring label within a set of labels. These labels are created by all of the Decision Trees in the Random Forest. The most common label will be returned.

def predict(self, X):

With the predict function, the created Random Forest model will be used to predict the label of a given test dataset. The data will run through all created Decision Trees in the forest. Each Decision Tree will predict the label of the data. Using the predictions of all trees, the most common predicted label is going to be taken as the final predicted label.

-------




