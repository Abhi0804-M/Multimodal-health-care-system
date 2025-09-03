import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('D:/DP/classification/Anemia.csv')

# Define classes for specific types of anemia
data['Anemia'] = np.where(data[' HGB '] < 11, 1, 0)  # General Anemia (1 if HGB < 11, 0 otherwise)
data['Iron_Deficiency_Anemia'] = np.where((data[' HGB '] < 11) & (data['MCV  '] < 80) & (data['MCH'] < 27), 1, 0)
data['Vitamin_Deficiency_Anemia'] = np.where((data[' HGB '] < 11) & (data['MCV  '] > 100) & (data['MCH'] > 30), 1, 0)
data['Sickle_Cell_Anemia'] = np.where((data['MCV  '] < 80) & (data[' HGB '] < 11), 1, 0)

# Separate anemia classification labels
anemia_labels = data['Anemia'].values
iron_deficiency_labels = data['Iron_Deficiency_Anemia'].values
vitamin_deficiency_labels = data['Vitamin_Deficiency_Anemia'].values
sickle_cell_labels = data['Sickle_Cell_Anemia'].values

# Separate features (X) for anemia classification
X_anemia = data.drop(['S. No.', 'Anemia', 'Iron_Deficiency_Anemia', 'Vitamin_Deficiency_Anemia', 'Sickle_Cell_Anemia','PCV',' PLT /mm3','TLC','  RBC    ',' MCHC  ',' RDW    ','Sex  ','Age      '], axis=1).values

# Normalize the input features to scale values between 0 and 1 for anemia classification
scaler = MinMaxScaler()
X_anemia_scaled = scaler.fit_transform(X_anemia)

# Reshape data to 3D tensor format for RNN
X_anemia_scaled = X_anemia_scaled.reshape((X_anemia_scaled.shape[0], 1, X_anemia_scaled.shape[1]))

# Build the anemia classification RNN model
anemia_model = Sequential()
anemia_model.add(LSTM(units=64, activation='relu', input_shape=(X_anemia_scaled.shape[1], X_anemia_scaled.shape[2])))
anemia_model.add(Dense(units=32, activation='relu'))
anemia_model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the anemia classification model
anemia_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set the number of folds for k-fold cross-validation
k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Define a dictionary to store the accuracy scores for each type of anemia
accuracy_scores = {
    'Anemia': [],
    'Iron_Deficiency_Anemia': [],
    'Vitamin_Deficiency_Anemia': [],
    'Sickle_Cell_Anemia': []
}

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X_anemia_scaled):
    X_train, X_test = X_anemia_scaled[train_index], X_anemia_scaled[test_index]
    y_train_anemia, y_test_anemia = anemia_labels[train_index], anemia_labels[test_index]
    y_train_iron_deficiency, y_test_iron_deficiency = iron_deficiency_labels[train_index], iron_deficiency_labels[test_index]
    y_train_vitamin_deficiency, y_test_vitamin_deficiency = vitamin_deficiency_labels[train_index], vitamin_deficiency_labels[test_index]
    y_train_sickle_cell, y_test_sickle_cell = sickle_cell_labels[train_index], sickle_cell_labels[test_index]

    # Train the anemia classification model for this fold
    anemia_model.fit(X_train, y_train_anemia, epochs=100, batch_size=32, verbose=1)
    anemia_model.save('anemia_model.h5')
    # Evaluate the anemia classification model on the test set for this fold
    y_anemia_pred_prob = anemia_model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test_anemia, y_anemia_pred_prob)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_anemia_pred_classes = (y_anemia_pred_prob >= optimal_threshold).astype(int)

    # Train the iron deficiency anemia classification model for this fold
    iron_deficiency_model = Sequential()
    iron_deficiency_model.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    iron_deficiency_model.add(Dense(units=32, activation='relu'))
    iron_deficiency_model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid activation for binary classification
    iron_deficiency_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    iron_deficiency_model.fit(X_train, y_train_iron_deficiency, epochs=100, batch_size=32, verbose=1)
    iron_deficiency_model.save('iron_deficiency_model.h5')  # Save the iron deficiency anemia model

    # Train the vitamin deficiency anemia classification model for this fold
    vitamin_deficiency_model = Sequential()
    vitamin_deficiency_model.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    vitamin_deficiency_model.add(Dense(units=32, activation='relu'))
    vitamin_deficiency_model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid activation for binary classification
    vitamin_deficiency_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    vitamin_deficiency_model.fit(X_train, y_train_vitamin_deficiency, epochs=100, batch_size=32, verbose=1)
    vitamin_deficiency_model.save('vitamin_deficiency_model.h5')  # Save the vitamin deficiency anemia model

    # Train the sickle cell anemia classification model for this fold
    sickle_cell_model = Sequential()
    sickle_cell_model.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    sickle_cell_model.add(Dense(units=32, activation='relu'))
    sickle_cell_model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid activation for binary classification
    sickle_cell_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    sickle_cell_model.fit(X_train, y_train_sickle_cell, epochs=100, batch_size=32, verbose=1)
    sickle_cell_model.save('sickle_cell_model.h5')  # Save the sickle cell anemia model

    # Evaluate the iron deficiency anemia classification model on the test set for this fold
    y_iron_deficiency_pred_prob = iron_deficiency_model.predict(X_test)
    y_iron_deficiency_pred_classes = (y_iron_deficiency_pred_prob >= 0.5).astype(int)

    # Evaluate the vitamin deficiency anemia classification model on the test set for this fold
    y_vitamin_deficiency_pred_prob = vitamin_deficiency_model.predict(X_test)
    y_vitamin_deficiency_pred_classes = (y_vitamin_deficiency_pred_prob >= 0.5).astype(int)

    # Evaluate the sickle cell anemia classification model on the test set for this fold
    y_sickle_cell_pred_prob = sickle_cell_model.predict(X_test)
    y_sickle_cell_pred_classes = (y_sickle_cell_pred_prob >= 0.5).astype(int)

    # Calculate and store the accuracy score for each type of anemia for this fold
    accuracy_scores['Anemia'].append(accuracy_score(y_test_anemia, y_anemia_pred_classes))
    accuracy_scores['Iron_Deficiency_Anemia'].append(accuracy_score(y_test_iron_deficiency, y_iron_deficiency_pred_classes))
    accuracy_scores['Vitamin_Deficiency_Anemia'].append(accuracy_score(y_test_vitamin_deficiency, y_vitamin_deficiency_pred_classes))
    accuracy_scores['Sickle_Cell_Anemia'].append(accuracy_score(y_test_sickle_cell, y_sickle_cell_pred_classes))

    # Generate the confusion matrix for anemia classification for this fold
    confusion_anemia = confusion_matrix(y_test_anemia, y_anemia_pred_classes)

    # Generate the confusion matrix for iron deficiency anemia classification for this fold
    confusion_iron_deficiency = confusion_matrix(y_test_iron_deficiency, y_iron_deficiency_pred_classes)

    # Generate the confusion matrix for vitamin deficiency anemia classification for this fold
    confusion_vitamin_deficiency = confusion_matrix(y_test_vitamin_deficiency, y_vitamin_deficiency_pred_classes)

    # Generate the confusion matrix for sickle cell anemia classification for this fold
    confusion_sickle_cell = confusion_matrix(y_test_sickle_cell, y_sickle_cell_pred_classes)

    # Display the confusion matrices for this fold
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_anemia, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Anemia Classification (Fold {})".format(kf.get_n_splits()))
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_iron_deficiency, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Iron Deficiency Anemia Classification (Fold {})".format(kf.get_n_splits()))
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_vitamin_deficiency, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Vitamin Deficiency Anemia Classification (Fold {})".format(kf.get_n_splits()))
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_sickle_cell, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Sickle Cell Anemia Classification (Fold {})".format(kf.get_n_splits()))
    plt.show()

# Calculate the mean and standard deviation of the accuracy scores for each type of anemia
for anemia_type, scores in accuracy_scores.items():
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    print(f'Mean Accuracy ({anemia_type}): {mean_accuracy:.4f}, Standard Deviation: {std_accuracy:.4f}')

# Now, let's predict anemia, iron deficiency anemia, vitamin deficiency anemia, and sickle cell anemia for new input data
# Assuming you have new input values for MCV, MCH, and HGB as a list
new_input_values = [75, 25, 9]

# Convert the input values to a numpy array and reshape into a single sample with 3 features
new_input_values = np.array(new_input_values).reshape(1, -1)

# Scale the new input features using the MinMaxScaler
new_input_scaled = scaler.transform(new_input_values)

# Reshape the scaled input data to 3D tensor format for RNN
new_input_rnn = new_input_scaled.reshape((new_input_scaled.shape[0], 1, new_input_scaled.shape[1]))

# Make predictions for anemia, iron deficiency anemia, vitamin deficiency anemia, and sickle cell anemia using the trained RNN models
anemia_probability = anemia_model.predict(new_input_rnn)
iron_deficiency_probability = iron_deficiency_model.predict(new_input_rnn)
vitamin_deficiency_probability = vitamin_deficiency_model.predict(new_input_rnn)
sickle_cell_probability = sickle_cell_model.predict(new_input_rnn)

# Check the probabilities for each type of anemia
print(f'Probability of Anemia: {anemia_probability[0][0]:.4f}')
print(f'Probability of Iron Deficiency Anemia: {iron_deficiency_probability[0][0]:.4f}')
print(f'Probability of Vitamin Deficiency Anemia: {vitamin_deficiency_probability[0][0]:.4f}')
print(f'Probability of Sickle Cell Anemia: {sickle_cell_probability[0][0]:.4f}')
