import pickle

# Correct relative path
file_path = "./models/emotion_classifier_pipe_lr.pkl"

# Load the pickle file
with open(file_path, 'rb') as file:
    model = pickle.load(file)

# Display model details
print(model)
