import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load IMDB dataset
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to ensure uniform input size
max_len = 200
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

# Define LSTM model
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=num_words, output_dim=128, input_length=max_len)(input_layer)
dropout = SpatialDropout1D(0.2)(embedding)
lstm_layer = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(dropout)
dense_layer = Dense(1, activation='sigmoid')(lstm_layer)

model = Model(input_layer, dense_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate model
y_pred = (model.predict(x_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
