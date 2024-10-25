# Build Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

def buildLogisticRegressionModel(X, y):
    # Initialize the model (Logistic Regression)    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)
    model = LogisticRegression(max_iter=10000)
    
    # Apply RFECV (RFE with cross-validation) to automatically select the best number of features
    # Initialize RFECV with Logistic Regression and 5-fold cross-validation
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
    
    # Fit the RFECV model on the training data
    rfecv.fit(X_train, y_train)
    
    # Training Data Predictions: Useful for understanding how well the model has fit the training data, 
    # Test Data Predictions: Crucial for evaluating model performance and generalization
    test_accuracy = rfecv.score(X_test, y_test)
    print(f"Model Accuracy (using score method): {test_accuracy * 100:.2f}%")
    
    # Without PCA Model Accuracy (using score method): 83.35%
    # With PCA Model Accuracy (using score method): 78.56%

    return test_accuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

def buildLinearRegressionModel(X, y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)
    
    model = LinearRegression()
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    test_accuracy = model.score(X_test, y_test)
    print(test_accuracy)
    print(f"Model Accuracy (using score method): {test_accuracy * 100:.2f}%")
    
    # Calculate metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # Store results in list
    print('RMSE', rmse, 'R2 Score', r2)

    return test_accuracy