import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Input
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from sklearn.preprocessing import MultiLabelBinarizer
from keras.optimizers import Adam





# Step 1: Load the data from the JSON file
data=[{"input":"cama","output":["casa"]},
      {"input":"mesa","output":["casa","trabalho"]},
      {"input":"computador","output":["casa","trabalho"]},
      {"input":"cadeira","output":["casa","trabalho"]},
        {"input":"sala de reunião","output":["trabalho"]},     
]

#Step 2: Extract input and output sequences
input=[item['input'] for item in data]
output=[item['output'] for item in data]

print("Input\n",input)
print("Output\n",output)

# Step 3: Create a character set for input values
def create_charset(input_texts):
    all_texts = ''.join(input_texts)
    chars = sorted(set(all_texts))
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(chars)
    return char_to_index,vocab_size



def encode_text(text,char_to_index):
    return [char_to_index[char] for char in text]



# Step 4: Create a character set for input values
char_to_index,vocab_size=create_charset(input)
print("Char to index")
print(char_to_index)

#step 5: Encode the input text using the loaded char_to_index mapping
encoded_input = [encode_text(text,char_to_index) for text in input]
print("Encoded Input\n",encoded_input)

# Step 6: Pad the sequences to the same length
max_seq_len = max(len(seq) for seq in encoded_input)

input_sequences = pad_sequences(encoded_input, maxlen=max_seq_len, padding='post')
print('Input Sequences')
print(input_sequences)

# Step 7: Encode Output Labels
mlb = MultiLabelBinarizer()
categories = mlb.fit_transform(output)
num_classes = len(mlb.classes_)
print('Output Labels')
print(categories)

# Step 8: Define the LSTM model
model = Sequential([
    Input(shape=(max_seq_len,)),  # Input shape is the padded sequence length
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_seq_len),
    LSTM(5, return_sequences=False),          
    Dense(num_classes, activation='sigmoid')  # Predict the next character from the vocabulary
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#Step 10: create a custom callback to stop training when accuracy reaches 1.0
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):

        predictions = model.predict(input_sequences,verbose=0)
        predicted_labels = (predictions > 0.5).astype(int)

        erro=0
        length=len(predicted_labels)
        for i in range(len(predicted_labels)):
            if not np.array_equal(predicted_labels[i],categories[i]):
                erro+=1

        print("Epoch: ",epoch," Erro:",erro/length,". Accuracy:",1-erro/length)
        accuracy = 1-erro/length
        if accuracy ==1.0:
            print(f"Stopping training at epoch {epoch + 1} because accuracy reached 1.0")
            self.model.stop_training = True  # Stops training

accuracy_callback = MyCallback()

model.fit(input_sequences, categories, epochs=200,verbose=0,callbacks=[accuracy_callback])


# Step 6: Prediction and Decoding



predictions = model.predict(input_sequences,verbose=0)
predicted_labels = (predictions > 0.5).astype(int)
decoded_labels = mlb.inverse_transform(predicted_labels)

print(decoded_labels)

# Step 7: Save the model
model.save('cattest.h5')






