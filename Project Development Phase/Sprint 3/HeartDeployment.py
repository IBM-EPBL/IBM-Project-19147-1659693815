# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('./Heartnew.csv')


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = clean_dataset(df)


#import pandas as pd
import matplotlib.pyplot as plt

# read-in data
#data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs

import seaborn as sns
sns.countplot(df['active'],label="Count")
plt.show()

# count occurences
#occurrences = df.loc[:, 'Outcome'].value_counts()

# plot histogram
#plt.bar(occurrences.keys(), occurrences)
#plt.show()


# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco']] = df_copy[['gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution

#df_copy['gender'].fillna(df_copy['gender'].mean(), inplace=True)
#df_copy['height'].fillna(df_copy['height'].mean(), inplace=True)
#df_copy['weight'].fillna(df_copy['weight'].mean(), inplace=True)
#df_copy['ap_hi'].fillna(df_copy['ap_hi'].mean(), inplace=True)
#df_copy['ap_lo'].fillna(df_copy['ap_lo'].mean(), inplace=True)
#df_copy['cholesterol'].fillna(df_copy['cholesterol'].mean(), inplace=True)
#df_copy['gluc'].fillna(df_copy['gluc'].mean(), inplace=True)
#df_copy['smoke'].fillna(df_copy['smoke'].mean(), inplace=True)
#df_copy['alco'].fillna(df_copy['alco'].mean(), inplace=True)





# Model Building
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'active'], df['active'], stratify=df['active'], random_state=0)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train, y_train)

score = mlp.score(X_test,y_test)
print(score)


print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))




filename = 'heart-prediction-rfc-model.pkl'
pickle.dump(mlp, open(filename, 'wb'))

# Creating a pickle file for the classifier
#filename = 'heart-prediction-rfc-model.pkl'
#pickle.dump(classifier, open(filename, 'wb'))


Prescription = ''

filename = 'heart-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))


data = np.array([[52,2,180,84,120,80,1,1,0,0]])
my_prediction = classifier.predict(data)
print(my_prediction[0])

if my_prediction == 1:
    Answer = 'Heart'




    print('Hello:According to our Calculations, You have Heart')

else:
   # Answer = 'No-Heart Di'
    msg = 'Congratulations!!  You DON T have Heart Disease'
    print('Congratulations!! You DON T have Heart Disease')
    Prescription = 'Nill'



