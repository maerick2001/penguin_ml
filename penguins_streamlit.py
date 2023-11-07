# App to predict penguin species
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title('Penguin Classifier: A Machine Learning App') 

password_guess = st.text_input("What is the password?!")
if password_guess != st.secrets['password']:
  st.stop()

# Display the image
st.image('penguins.png', width = 400)

st.write("This app uses 6 inputs to predict the species of penguin using " 
         "a model built on the Palmer's Penguin's dataset. Use the form below" 
         " to get started!") 

# Import CSV file provided by user (if not, stop the app from running)
penguin_file = st.file_uploader("Select Your Local Penguins CSV")

# Reading the pickle files that we created before 
dt_pickle = open('decision_tree_penguin.pickle', 'rb') 
map_pickle = open('output_penguin.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
unique_penguin_mapping = pickle.load(map_pickle)
dt_pickle.close() 
map_pickle.close()

# Checking if these are the same Python objects that we used before
# st.write(clf)
# st.write(unique_penguin_mapping)

if penguin_file is not None:
    penguins_df = pd.read_csv(penguin_file) # User provided file
    # For categorical variables, using selectbox
    island = penguins_df['island']
    sex = penguins_df['sex']

    # For numerical variables, using number_input
    # NOTE: Make sure that variable names are same as that of training dataset
    bill_length_mm = penguins_df['bill_length_mm']
    bill_depth_mm = penguins_df['bill_depth_mm']
    flipper_length_mm = penguins_df['flipper_length_mm']
    body_mass_g = penguins_df['body_mass_g']

    # Putting sex and island variables into the correct format
    # so that they can be used by the model for prediction
    #island_Biscoe, island_Dream, island_Torgerson = 0, 0, 0
    st.dataframe(penguins_df)
    if island == 'Biscoe':
      penguins_df['island_Biscoe'] = 1
    elif island == 'Dream':
      penguins_df['island_Dream'] = 1
    elif island == 'Torgerson':
      penguins_df['island_Torgerson'] = 1

    sex_female, sex_male = 0, 0
    if sex == 'Female':
      sex_female = 1
    elif sex == 'Male':
      sex_male = 1

    # Using predict() with new data provided by the user
    new_prediction = clf.predict([[bill_length_mm, bill_depth_mm, flipper_length_mm,
                                  body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]])

    new_prediction_prob = clf.predict_proba([[bill_length_mm, bill_depth_mm, flipper_length_mm,
                                              body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]])

    a = np.argmax(new_prediction_prob)  # index of max probability
    b = np.amax(new_prediction_prob)  # max probability
    #st.write(new_prediction_prob)
    #st.write(a)
    final_pred = penguin_file

    # Map prediction with penguin species
    prediction_species = unique_penguin_mapping[new_prediction][0]

    # Show the predicted species on the app
    st.subheader("Predicting Your Penguin's Species")
    st.dataframe(final_pred)
else:
        # Adding Streamlit functions to get user input
        # For categorical variables, using selectbox
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])

    # For numerical variables, using number_input
    # NOTE: Make sure that variable names are same as that of training dataset
    bill_length_mm = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth_mm = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass_g = st.number_input('Body Mass (g)', min_value=0)

    # st.write('The user inputs are {}'.format([island, sex, bill_length, bill_depth, flipper_length, body_mass]))

    # Putting sex and island variables into the correct format
    # so that they can be used by the model for prediction
    island_Biscoe, island_Dream, island_Torgerson = 0, 0, 0
    if island == 'Biscoe':
      island_Biscoe = 1
    elif island == 'Dream':
      island_Dream = 1
    elif island == 'Torgerson':
      island_Torgerson = 1

    sex_female, sex_male = 0, 0
    if sex == 'Female':
      sex_female = 1
    elif sex == 'Male':
      sex_male = 1

    # Using predict() with new data provided by the user
    new_prediction = clf.predict([[bill_length_mm, bill_depth_mm, flipper_length_mm,
                                  body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]])

    new_prediction_prob = clf.predict_proba([[bill_length_mm, bill_depth_mm, flipper_length_mm,
                                              body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]])

    a = np.argmax(new_prediction_prob)  # index of max probability
    b = np.amax(new_prediction_prob)  # max probability
    #st.write(new_prediction_prob)
    #st.write(a)

    # Map prediction with penguin species
    prediction_species = unique_penguin_mapping[new_prediction][0]

    # Show the predicted species on the app
    st.subheader("Predicting Your Penguin's Species")
    st.write('We predict your penguin is of the {} species'.format(
        prediction_species), 'with a {}% probabilty.'.format(b*100))

    # Showing Feature Importance plot
    st.write('We used a machine learning model (Decision Tree) to '
            'predict the species, the features used in this prediction '
            'are ranked by relative importance below.')
    st.image('feature_imp.svg')



