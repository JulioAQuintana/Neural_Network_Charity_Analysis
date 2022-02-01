# Neural Network Charity Analysis

## Background
Beks has come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

## Overview
  **_Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2._**

The repository purpose is to demostrate the unsupervised machine learning to analyze a database of cryptocurrencies and create a report including the traded cryptocurrencies
classified by group according to their features, after this classification we goint to perform a report to give to the bank the better option regarding to cryptocurrency
investment option using the following methods : 

* **Deliverable 1:** Preprocessing Data for a Neural Network Model
* **Deliverable 2:** Compile, Train, and Evaluate the Model
* **Deliverable 3:** Optimize the Model


## Resources

**_list resources used_**

* **Data Source:** crypto_data.csv and crypto_clustering_starter_code.ipynb
* **Software:** Jupyter Notebook 6.3.0, MELNV (python environment)

## Results

### **_Preprocessing the Data for PCA._**

#### Deliverables 1

   * All cryptocurrencies that are not being traded are removed
   * The IsTrading column is dropped
   * All the rows that have at least one null value are removed
   * All the rows that do not have coins being mined are removed
   * The CoinName column is dropped
   * A new DataFrame is created that stores all cryptocurrency names from the CoinName column and retains the index from the crypto_df DataFrame
   * The get_dummies() method is used to create variables for the text features, which are then stored in a new DataFrame, X
   * X DataFrame have been standardized using the StandardScaler fit_transform() function

   ![](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/X_df.png)

In this part we ensure the correct data transformation before supervised machine learning process, result will be a data set tranformed of cryptocurrency information.

### **_Reducing Data Dimensions Using PCA_**

Using your knowledge of how to apply the Principal Component Analysis (PCA) algorithm, you’ll reduce the dimensions of the X DataFrame to three principal components and place these dimensions in a new DataFrame.

####  Deliverables 2
   * The PCA algorithm reduces the dimensions of the X DataFrame down to three principal components 
   * The pcs_df DataFrame is created and has the following three columns, PC 1, PC 2, and PC 3, and has the index from the crypto_df DataFrame


   ![](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/PCA_reduceDAta.png)

in this part we reduced the dimmension of X DataFrame in 3 main components. 

### **_Clustering Crytocurrencies Using K-Means_**
 
 Using your knowledge of the K-means algorithm, you’ll create an elbow curve using hvPlot to find the best value for K from the pcs_df DataFrame created in Deliverable 2. Then, you’ll run the K-means algorithm to predict the K clusters for the cryptocurrencies’ data.
 
#### Deliverables 3
   * An elbow curve is created using hvPlot to find the best value for K
   * Predictions are made on the K clusters of the cryptocurrencies’ data
   * A new DataFrame is created with the same index as the crypto_df DataFrame and has the following columns: Algorithm, ProofType, TotalCoinsMined, TotalCoinSupply, PC 1, PC 2, PC 3, CoinName, and Class

   ![Elbow Curve](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/elbow%20curve.png)
 
through Elbow curve shows we got output of 4 clusters for cryptocurrencies categorization.

   ![Clustered Data Frame](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/clustered%20DF.png)
 
 
### **_Visualizing Cryptocurrencies Results_**
 Using your knowledge of creating scatter plots with Plotly Express and hvplot, you’ll visualize the distinct groups that correspond to the three principal components you created in Deliverable 2, then you’ll create a table with all the currently tradable cryptocurrencies using the hvplot.table() function.
 
####  Deliverables 4
   * The clusters are plotted using a 3D scatter plot, and each data point shows the CoinName and Algorithm on hover
   * A table with tradable cryptocurrencies is created using the hvplot.table() function
   * The total number of tradable cryptocurrencies is printed
   * A DataFrame is created that contains the clustered_df DataFrame index, the scaled data, and the CoinName and Class columns
   * A hvplot scatter plot is created where the X-axis is "TotalCoinsMined", the Y-axis is "TotalCoinSupply", the data is ordered by "Class", and it shows the CoinName when you hover over the data point

   ![3D Scatter](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/3DScatter.png)

3-D scatter shows that crytocurrencies dimensions was reduced to three principal components.

   ![ tradable cryptocurrencies](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/tradable%20cryptocurrencies.png)

in the table we got the class of every cryptocurrencies 

   ![ Plot DF](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/plotDF.png)

   ![ Total Coins Mined](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/TotalcoinsMined.png)

Total coins mined plot shows the classes  and we can cam see the differences between all cryptocurrencies. 

## Summary

After development we got 532 tadable cryptocurrencies, those have to be evalueted based in each performance in order to define potential interest for clients to decide invstment.

