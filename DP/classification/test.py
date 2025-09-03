import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the trained models
iron_deficiency_model = load_model('iron_deficiency_model.h5')
vitamin_deficiency_model = load_model('vitamin_deficiency_model.h5')
sickle_cell_model = load_model('sickle_cell_model.h5')

# Define the MinMaxScaler used during training (Assuming it was saved with the models)
# Note: In practice, you should save and load the scaler used during training.
# For now, we create a new scaler instance and assume it was fitted on similar data ranges.
scaler = MinMaxScaler()
scaler.fit([[80, 27, 11], [100, 30, 11]])  # Example fitting range similar to training data

# Function to get input from the user and make predictions
def get_user_input_and_predict():
    # Get input values from the user
    mcv = float(input("Enter MCV value: "))
    mch = float(input("Enter MCH value: "))
    hgb = float(input("Enter HGB value: "))

    # Convert the input values to a numpy array and reshape into a single sample with 3 features
    new_input_values = np.array([mcv, mch, hgb]).reshape(1, -1)

    # Scale the new input features using the MinMaxScaler
    new_input_scaled = scaler.transform(new_input_values)

    # Reshape the scaled input data to 3D tensor format for RNN
    new_input_rnn = new_input_scaled.reshape((new_input_scaled.shape[0], 1, new_input_scaled.shape[1]))

    # Make predictions for iron deficiency anemia, vitamin deficiency anemia, and sickle cell anemia
    iron_deficiency_probability = iron_deficiency_model.predict(new_input_rnn)[0][0]
    vitamin_deficiency_probability = vitamin_deficiency_model.predict(new_input_rnn)[0][0]
    sickle_cell_probability = sickle_cell_model.predict(new_input_rnn)[0][0]

    # Combine probabilities into a single array
    probabilities = np.array([iron_deficiency_probability, vitamin_deficiency_probability, sickle_cell_probability])

    # Normalize the probabilities so they sum to 1
    normalized_probabilities = probabilities / probabilities.sum()

    # Display the probabilities for each type of anemia
    print(f'Probability of Iron Deficiency Anemia: {normalized_probabilities[0]:.4f}')
    print(f'Probability of Vitamin Deficiency Anemia: {normalized_probabilities[1]:.4f}')
    print(f'Probability of Sickle Cell Anemia: {normalized_probabilities[2]:.4f}')

# Run the function to get user input and predict
get_user_input_and_predict()
