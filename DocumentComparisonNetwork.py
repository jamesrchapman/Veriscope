import numpy as np
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, concatenate, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.sequence import pad_sequences

def create_siamese_model(vocab_size, embedding_dim, max_length):
    # Input layer
    input_a = Input(shape=(max_length,))
    input_b = Input(shape=(max_length,))

    # Embedding layer
    embedding = Embedding(vocab_size, embedding_dim)

    # Convolutional Neural Network
    cnn = Conv1D(128, 3, activation='relu')

    # Model branches for inputs A and B
    branch_a = cnn(embedding(input_a))
    branch_b = cnn(embedding(input_b))

    # Global max pooling
    pooled_a = GlobalMaxPooling1D()(branch_a)
    pooled_b = GlobalMaxPooling1D()(branch_b)

    # Concatenating the pooled outputs
    concatenated = concatenate([pooled_a, pooled_b])

    # Dense layers
    dense = Dense(128, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense)

    # Create the Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)
    siamese_model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=['accuracy'])
    
    return siamese_model

# Example usage
vocab_size = 10000  # Size of the vocabulary
embedding_dim = 100  # Dimension of the word embeddings
max_length = 100  # Maximum length of input sequences

# Generate some dummy data for demonstration purposes
num_samples = 1000
input_a = np.random.randint(0, vocab_size, size=(num_samples, max_length))
input_b = np.random.randint(0, vocab_size, size=(num_samples, max_length))
labels = np.random.randint(0, 2, size=(num_samples,))

# Pad sequences to a fixed length
input_a = pad_sequences(input_a, maxlen=max_length)
input_b = pad_sequences(input_b, maxlen=max_length)

# Create the Siamese model
model = create_siamese_model(vocab_size, embedding_dim, max_length)

# Train the model
model.fit([input_a, input_b], labels, epochs=10, batch_size=32)
