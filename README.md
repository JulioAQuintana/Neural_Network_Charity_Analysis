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

The repository purpose is to demonstrate the unsupervised machine learning to analyze a database of cryptocurrencies and create a report including the traded cryptocurrencies
classified by group according to their features, after this classification we going to perform a report to give to the bank the better option regarding to cryptocurrency
investment option using the following methods: 

* **Deliverable 1:** Preprocessing Data for a Neural Network Model
* **Deliverable 2:** Compile, Train, and Evaluate the Model
* **Deliverable 3:** Optimize the Model


## Resources

**_list resources used_**

* **Data Source:** crypto_data.csv and crypto_clustering_starter_code.ipynb
* **Software:** Jupyter Notebook 6.3.0, MELNV (python environment)

## Results

Compiling, Training, and Evaluating the Model
This deep-learning neural network model is made of two hidden layers with 80 and 30 neurons respectively.
The input data has 43 features and 25,724 samples.
The output layer is made of a unique neuron as it is a binary classification.
To speed up the training process, we are using the activation function ReLU for the hidden layers. As our output is a binary classification, Sigmoid is used on the output layer.
For the compilation, the optimizer is adam and the loss function is binary_crossentropy.
The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.
To increase the performance of the model, we applied bucketing to the feature ASK_AMT and organized the different values by intervals.
We increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.
We also tried a different activation function (tanh) but none of these steps helped improve the model's performance.

### **_Data Preprocessing._**

#### Deliverable 1

   * **_What variable(s) are considered the target(s) for your model?_**
   The variables considered for my model was IS_SUCCESSFUL
   
   * **_What variable(s) are considered to be the features for your model?_**
   All columns are considered excluding IS_SUCCESSFUL due that is a target of our deep neural network
   * **_What variable(s) are neither targets nor features, and should be removed from the input data?_**
   I decided to drop EIN and NAME columns ID to be non- beneficial. code used, see below:
   
   ```sh
   application_df = application_df.drop(columns=["EIN", "NAME"], axis=1)
  ```
   
### **_Compiling, Training, and Evaluating the Model_**

####  Deliverables 2 & 3
   * How many neurons, layers, and activation functions did you select for your neural network model, and why?
   I used 80 and 30 neurons for 2 hidden layers. In the hidden layers I used the "relu" activation function and the activation function for the output layer was "sigmoid", I was looking for well accuracy up 75%.see code below: 
   ```sh
   # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
nodes_hidden_layer1 = 80
nodes_hidden_layer2 = 30
nn = tf.keras.models.Sequential()
# First hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="relu", input_dim=number_input_features))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
  ```
  
   * Were you able to achieve the target model performance?
Model was not able to reach the target 75%. The accuracy for my model was 72%.
   ![](https://github.com/JulioAQuintana/Cryptocurrencies/blob/main/Resources/PCA_reduceDAta.png)

   * What steps did you take to try and increase model performance?
   In the first part I also droped USE_CASE 
   ```sh
   # Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(columns=["EIN", "NAME", "USE_CASE"], axis=1)
application_df.head()

  ```  
  Add More neurons and hidden Layers as followin reference
   ```sh
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 100
hidden_nodes_layer2 = 50
hidden_nodes_layer3 = 20

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))


# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

  ``` 
and finally Used Different activation functions like "tanh for the hidden layers as following reference: 

   ```sh
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 100
hidden_nodes_layer2 = 50
hidden_nodes_layer3 = 20

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))


# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="tanh"))

# Check the structure of the model
nn.summary()

  ``` 
and reduce the epochs to 25 

   ```sh
# Train the model reducing 25 epochs
fit_model = nn.fit(X_train, y_train,epochs=25)

  ``` 
 **_I can't reach the target accuracy with the attempts, I got 63% as highest value._**



## Summary

After development of 3 different attempts and playing with some variants in the neural network model we can't reached the target value. probably we can use other variants and different mixed models trying to increase the accuracy or in other case use supervised machine learning model.
