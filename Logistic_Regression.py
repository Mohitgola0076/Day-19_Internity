                    # Understanding Logistic Regression :
             
Pre-requisite: Linear Regression
This article discusses the basics of Logistic Regression and its implementation in Python. Logistic regression is basically a supervised classification algorithm. In a classification problem, the target variable(or output), y, can take only discrete values for given set of features(or inputs), X.

Contrary to popular belief, logistic regression IS a regression model. The model builds a regression model to predict the probability that a given data entry belongs to the category numbered as “1”. Just like Linear regression assumes that the data follows a linear function, Logistic regression models the data using the sigmoid function.

g(z) = \frac{1}{1 + e^-^z}\ 

1. Low Precision/High Recall: In applications where we want to reduce the number of false negatives without necessarily reducing the number false positives, we choose a decision value which has a low value of Precision or high value of Recall. For example, in a cancer diagnosis application, we do not want any affected patient to be classified as not affected without giving much heed to if the patient is being wrongfully diagnosed with cancer. This is because, the absence of cancer can be detected by further medical diseases but the presence of the disease cannot be detected in an already rejected candidate.

2. High Precision/Low Recall: In applications where we want to reduce the number of false positives without necessarily reducing the number false negatives, we choose a decision value which has a high value of Precision or low value of Recall. For example, if we are classifying customers whether they will react positively or negatively to a personalised advertisement, we want to be absolutely sure that the customer will react positively to the advertisemnt because otherwise, a negative reaction can cause a loss potential sales from the customer.

Based on the number of categories, Logistic regression can be classified as:

binomial: target variable can have only 2 possible types: “0” or “1” which may represent “win” vs “loss”, “pass” vs “fail”, “dead” vs “alive”, etc.
multinomial: target variable can have 3 or more possible types which are not ordered(i.e. types have no quantitative significance) like “disease A” vs “disease B” vs “disease C”.
ordinal: it deals with target variables with ordered categories. For example, a test score can be categorized as:“very poor”, “poor”, “good”, “very good”. Here, each category can be given a score like 0, 1, 2, 3.


To do, so we apply the sigmoid activation function on the hypothetical function of linear regression. So the resultant hypothetical function for logistic regression is given below :

h( x ) = sigmoid( wx + b )

Here, w is the weight vector.
x is the feature vector. 
b is the bias.

sigmoid( z ) = 1 / ( 1 + e( - z ) )


             # Example : 
# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings( "ignore" )

# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression
# Logistic Regression
class LogitRegression() :
	def __init__( self, learning_rate, iterations ) :		
		self.learning_rate = learning_rate		
		self.iterations = iterations
		
	# Function for model training	
	def fit( self, X, Y ) :		
		# no_of_training_examples, no_of_features		
		self.m, self.n = X.shape		
		# weight initialization		
		self.W = np.zeros( self.n )		
		self.b = 0		
		self.X = X		
		self.Y = Y
		
		# gradient descent learning
				
		for i in range( self.iterations ) :			
			self.update_weights()			
		return self
	
	# Helper function to update weights in gradient descent
	
	def update_weights( self ) :		
		A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
		
		# calculate gradients		
		tmp = ( A - self.Y.T )		
		tmp = np.reshape( tmp, self.m )		
		dW = np.dot( self.X.T, tmp ) / self.m		
		db = np.sum( tmp ) / self.m
		
		# update weights	
		self.W = self.W - self.learning_rate * dW	
		self.b = self.b - self.learning_rate * db
		
		return self
	
	# Hypothetical function h( x )
	
	def predict( self, X ) :	
		Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )		
		Y = np.where( Z > 0.5, 1, 0 )		
		return Y


# Driver code

def main() :
	
	# Importing dataset	
	df = pd.read_csv( "diabetes.csv" )
	X = df.iloc[:,:-1].values
	Y = df.iloc[:,-1:].values
	
	# Splitting dataset into train and test set
	X_train, X_test, Y_train, Y_test = train_test_split(
	X, Y, test_size = 1/3, random_state = 0 )
	
	# Model training	
	model = LogitRegression( learning_rate = 0.01, iterations = 1000 )
	
	model.fit( X_train, Y_train )	
	model1 = LogisticRegression()	
	model1.fit( X_train, Y_train)
	
	# Prediction on test set
	Y_pred = model.predict( X_test )	
	Y_pred1 = model1.predict( X_test )
	
	# measure performance	
	correctly_classified = 0	
	correctly_classified1 = 0
	
	# counter	
	count = 0	
	for count in range( np.size( Y_pred ) ) :
		
		if Y_test[count] == Y_pred[count] :			
			correctly_classified = correctly_classified + 1
		
		if Y_test[count] == Y_pred1[count] :			
			correctly_classified1 = correctly_classified1 + 1
			
		count = count + 1
		
	print( "Accuracy on test set by our model	 : ", (
	correctly_classified / count ) * 100 )
	print( "Accuracy on test set by sklearn model : ", (
	correctly_classified1 / count ) * 100 )


if __name__ == "__main__" :	
	main()


                # Output : 
Accuracy on test set by our model       :   58.333333333333336
Accuracy on test set by sklearn model   :   61.111111111111114

############################################################################################################################

                # Linear Regression Vs. Logistic Regression : 
                
Linear regression gives you a continuous output, but logistic regression provides a constant output. An example of the continuous output is house price and stock price. Example's of the discrete output is predicting whether a patient has cancer or not, predicting whether the customer will churn. Linear regression is estimated using Ordinary Least Squares (OLS) while logistic regression is estimated using Maximum Likelihood Estimation (MLE) approach.

                # Maximum Likelihood Estimation Vs. Least Square Method : 
The MLE is a "likelihood" maximization method, while OLS is a distance-minimizing approximation method. Maximizing the likelihood function determines the parameters that are most likely to produce the observed data. From a statistical point of view, MLE sets the mean and variance as parameters in determining the specific parametric values for a given model. This set of parameters can be used for predicting the data needed in a normal distribution.



                                # Applications of Logistic Regression : 
 

1.) Logistic regression algorithm is applied in the field of epidemiology to identify risk factors for diseases and plan accordingly for preventive measures.
2.) Used to predict whether a candidate will win or lose a political election or to predict whether a voter will vote for a particular candidate.
3.) Used in weather forecasting to predict the probability of rain.
4.) Used in credit scoring systems for risk management to predict the defaulting of an account.






                        # Linear Regression	 : 
                        
Linear regression is used to predict the continuous dependent variable using a given set of independent variables.	
Linear Regression is used for solving Regression problem.
In Linear regression, we predict the value of continuous variables.	
In linear regression, we find the best fit line, by which we can easily predict the output.	
Least square estimation method is used for estimation of accuracy.	
The output for Linear Regression must be a continuous value, such as price, age, etc.	
In Linear regression, it is required that relationship between dependent variable and independent variable must be linear.	
In linear regression, there may be collinearity between the independent variables.	


                          # Logistic Regression : 

Logistic Regression is used to predict the categorical dependent variable using a given set of independent variables.
Logistic regression is used for solving Classification problems.
In logistic Regression, we predict the values of categorical variables.
In Logistic Regression, we find the S-curve by which we can classify the samples.
Maximum likelihood estimation method is used for estimation of accuracy.
The output of Logistic Regression must be a Categorical value such as 0 or 1, Yes or No, etc.
In Logistic regression, it is not required to have the linear relationship between the dependent and independent variable.
In logistic regression, there should not be collinearity between the independent variable.
