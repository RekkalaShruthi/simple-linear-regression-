import streamlit as st
import pickle
import numpy as np

# load the saved model
model=pickle.load(open(r'c:\Users\SHRUTHI\.vscode\A VS CODE\Machien Learning\slr\lrm.pkl','rb'))

# set title of the streamlit app
st.title('Salary Prediction app')

# Add a brief description
st.write('This app predicts the salary based on years of experience using a simple linear regression model.')

#Add input widget for user to enter years of experience
years_experience=st.number_input('Enter Years of Experience:', min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# when the button is clicked make predictions
if st.button('Predict Salary'):
    #make predictions using trained model
    experience_input=np.array([[years_experience]]) # convert input into 2D array for prediction
    prediction=model.predict(experience_input)
    
    # Display the result
    st.success(f'The predicted salary for {years_experience} years of experience is: ${prediction[0]}')

# Display information about the model
st.write('The model was trained using a dataset of salaries and years of experience.')
