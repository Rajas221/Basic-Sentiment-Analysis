#import libraries
import numpy as np  
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import imdb #imdb dataset is pre tokenized and each word is represented by an integer
from tensorflow.keras.preprocessing.sequence import pad_sequences #Ensures that all sequences in the dataset have the same length
from tensorflow.keras.models import Sequential #Lets us stack layers in order to build the model
from tensorflow.keras.layers import Embedding, LSTM, Dense #Embedding converts the integer representation of words into dense vectors of fixed size, LSTM learns patterns in sequences, Dense is a fully connected layer for binary classification 

#Embedding - Turns word IDs into vector embeddings
#LSTM - Learns dependenceies in word sequences 
#Dense(Sigmoid) - Converts LSTM O/P into a probablity(0-1) for binary classification


#Load Dataset
vocab_size = 10000 #Number of unique words to consider in the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size) #Load the dataset and limit to the top 10,000 words



#Exploring the data
print(f"Training Samples: {len(x_train)}")
print(f"First review: {x_train[0]}")
print(f"Label (0=negative, 1=positive): {y_train[0]}")
#To see the actual words
word_index = imdb.get_word_index() #Get the word index mapping
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #Reverse the mapping to get words from IDs
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]]) #Decode the first review
print(f"Decoded review: {decoded_review}") #Print the decoded review



#Pad Sequences 
#Ensure all sequences have the same length
max_length = 200 #Maximum length of each review
x_train = pad_sequences(x_train, maxlen=max_length) #Pad the training data
x_test = pad_sequences(x_test, maxlen=max_length) #Pad the testing data
print(x_train.shape) #Print the shape of the training data



#Build Neural Network Model
model = Sequential() #Create a sequential model
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length)) #Add an embedding layer turns indices into 128-dimensional vectors
model.add(LSTM(64)) #Learns sequence based patterns
model.add(Dense(1, activation='sigmoid')) #outputs probablity 
#Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #Compile the model with binary crossentropy loss and adam optimizer



#Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2) #Train the model with 5 epochs and a batch size of 128


#Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test) #Evaluate the model on the test data
print (f"Test Accuracy : {accuracy:.2f}%") #Print the test accuracy



#Make Predictions 
prediction = model.predict(x_test) #Make predictions on the test data
print(f"Prediction(Probablity):{prediction[0]}")
label = 1 if prediction[0] >= 0.5 else 0 #Convert the probability to a label (1 for positive, 0 for negative)
print(f"Predicted sentiment : {'Positive' if label == 1 else 'Negative'}") #Print the predicted sentiment



#Plotting Results
plt.plot(history.history['accuracy'], label='accuracy') #Plot the training accuracy
plt.plot(history.history['val_accuracy'], label='val_accuracy') #Plot the validation accuracy
plt.xlabel('Epochs') #Label the x-axis
plt.ylabel('Accuracy') #Label the y-axis
plt.title('Model Accuracy') #Title of the plot
plt.legend() #Show the legend
plt.show() #Display the plot
