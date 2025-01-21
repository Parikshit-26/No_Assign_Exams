import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# use pandas to load real_estate_dataset.csv
df = pd.read_csv("real_estate_dataset.csv")

#get the number of samples and features from the csv 
n_samples, n_features = df.shape
print(f"Number of samples, features: {n_samples, n_features}")

#get the number of columns in the dataset
columns = df.columns 
np.savetxt('Dataset_lec_2/columns.txt', columns, fmt='%s')

#save the column names to a file for accessing later as a text file
column_names = ["Square_Feet", "Garage_Size", "Location_Score", "Distance_to_Center"]
np.savetxt('Dataset_lec_2/selected_columns.txt', column_names, fmt='%s')

#From the dataset use Square_Feet, Garage_size, Location_Score, Distance_to_center as feature for the model
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]

#Use Price as the target
y = df["Price"].values

print(f"Shape of X: {X.shape}\n")
print(f"data type of X: {X.dtypes}\n")

#get the number of samples and features in X
n_samples, n_features = X.shape

#Build a linear model to predict price from the four features in X 
#make an array of coefs of the size of features +1, initialize to 1
coefs = np.ones(n_features+1)

#predict the price for each sample in X
predictions_bydefn = X@ coefs[1:]+ coefs[0]

#Append a column of 1s to X 
X = np.hstack((np.ones((n_samples, 1)), X))

#Predict the price for each sample in X
predictions = X@coefs


#check if predictions and predictions_bydefn are the same
is_same=np.allclose(predictions, predictions_bydefn)  

#print whether the predictions are the same in both cases
if is_same:
    print("The predictions are the same in both cases")

#Calculate the mean squared error of the model
errors = y - predictions
relative_error = errors / y
squared_loss  = errors.T @ errors
mean_squared_error = squared_loss / n_samples

#print shape and norm of errors
print(f"Shape of errors: {errors.shape}")
print(f"Norm of errors: {np.linalg.norm(errors)}")
print(f"Mean Squared Error: {mean_squared_error}")
#print(f"Relative error: {relative_error}")
print(f"Shape of relative error: {relative_error.shape}")
print(f"Norm of relative error: {np.linalg.norm(relative_error)}")

#get the loss matrix for the given data
loss_matrix = (y - X@coefs).T @ (y - X@coefs)/n_samples

#Calculate the gradient of the loss function
gradient = -2*X.T @ (y - X@coefs)/n_samples

coefs = np.linalg.inv(X.T @ X) @ X.T @ y

#Save the coefficients to a file
np.savetxt('Dataset_lec_2/coefficients.txt', coefs, fmt='%s')

#Predict the price for each sample in X
optimal_solution_predictions = X@coefs

#error model 
opt_sol_error = y - optimal_solution_predictions

#Find relative error
opt_sol_relative_error = opt_sol_error / y

#Print the norm of the relative error
print(f"Norm of optimal solution relative error: {np.linalg.norm(opt_sol_relative_error)}")

#Use all the features in the dataset to build a linear model to predict price
y = df["Price"].values
X = df[['Square_Feet']].values
n_samples,n_features = X.shape

#Append a column of 1s to X
X = np.hstack((np.ones((n_samples, 1)), X.reshape(-1, 1)))
# coefs = np.ones(n_features+1)

#Calculate the optimal solution
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

#Predict the price for each sample in X
predictions_with_all_features = X @ coefs

#Calculate the error
errors_with_all_features = y - predictions_with_all_features

#print the norm of the error
print(f"Norm of errors with all features: {np.linalg.norm(errors_with_all_features)}")
print(f"Norm of relative error with all features: {np.linalg.norm(errors_with_all_features/y)}")

#save the coefficients to a text file
np.savetxt('Dataset_lec_2/coefficients_all_features.txt', coefs, fmt='%s')

#Solve the normal equations using the QR decomposition
Q, R = np.linalg.qr(X)

print(f"Shape of Q: {Q.shape}")
print(f"Shape of R: {R.shape}")

#write R to a file named R.txt
np.savetxt('Dataset_lec_2/R.txt', R, fmt='%s')

#Calculate Q.T@Q and save it to a file named Q_TQ.txt
Q_TQ = Q.T @ Q
np.savetxt('Dataset_lec_2/Q_TQ.txt', Q_TQ, fmt='%s')

b = Q.T @ y

#Solve the system Rx = b
coefs_back_subs = np.zeros(n_features+1)
for i in range(n_features, -1, -1):
    coefs_back_subs[i] = (b[i] - R[i, i+1:] @ coefs_back_subs[i+1:])/R[i, i]

#save the coefficients to a file named coefficients_back_substitution.txt
np.savetxt('Dataset_lec_2/coefficients_back_substitution.txt', coefs_back_subs, fmt='%s')

#Solve the normal equations using the SVD decomposition
U, S, Vt = np.linalg.svd(X)

#write U, S, Vt to files named U.txt, S.txt, Vt.txt
np.savetxt('Dataset_lec_2/U.txt', U, fmt='%s')
np.savetxt('Dataset_lec_2/S.txt', S, fmt='%s')
np.savetxt('Dataset_lec_2/Vt.txt', Vt, fmt='%s')

X_feature = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 10)
X_feature = np.hstack((np.ones((X_feature.shape[0], 1)), X_feature.reshape(-1, 1)))
plt.scatter(X[:, 1], y, color='blue')
plt.plot(X_feature[:,1], X_feature @ coefs, color='red')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.show()
plt.savefig('Dataset_lec_2/Price_vs_Square_Feet.png')







