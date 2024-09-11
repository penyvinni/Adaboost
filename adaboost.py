#Panagiota Vinni 
#A.M. : 1873
#Statistical Machine Learning
 
#Import necessary libraries
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#Load the diabetes dataset
diabetes = load_diabetes()
# Show the dataset's keys
#print(list(diabetes))

#Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

#Create an adaboost regressor
reg = AdaBoostRegressor(random_state=42)

#Fit the model
reg.fit(X_train, y_train)

#Predict the target values
y_pred = reg.predict(X_test)

#Calculate the accuracy of the model
acc = reg.score(X_test, y_test)

#Print the accuracy
print("Accuracy of Adaboost Algorithm: {}\n".format(acc))



'''
Ξεκινά με την εισαγωγή των απαραίτητων βιβλιοθηκών, όπως NumPy και scikit-learn. 
Στη συνέχεια φορτώνεται το σύνολο δεδομένων διαβήτη και χωρίζεται σε σύνολα εκπαίδευσης και δοκιμής χρησιμοποιώντας τη συνάρτηση train_test_split(). 
Δημιουργείται ένα αντικείμενο AdaBoostRegressor και το μοντέλο προσαρμόζεται στα δεδομένα εκπαίδευσης. 
Οι προβλέψεις γίνονται στο σύνολο δοκιμής και η ακρίβεια του μοντέλου υπολογίζεται με τη χρήση του συντελεστή προσδιορισμού. 
Τέλος, εκτυπώνεται η ακρίβεια του μοντέλου. 
Συνολικά, αυτός ο κώδικας παρουσιάζει τη διαδικασία εκπαίδευσης ενός μοντέλου παλινδρόμησης AdaBoost, 
την αξιολόγηση της ακρίβειάς του και τη δημιουργία προβλέψεων.
'''