import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

N = 5
patterns = np.eye(N)

model = Sequential()
model.add(Dense(2, input_shape=(N,), activation='sigmoid'))
model.add(Dense(N, input_shape=(2,), activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mse', optimizer=opt)

model.fit(patterns, patterns, epochs=1000)

out = model.predict(patterns)

print(np.maximum(0.01,(out*100).round()/100))
