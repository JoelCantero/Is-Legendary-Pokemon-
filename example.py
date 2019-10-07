import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from datetime import datetime
import os
from tensorflow.keras.metrics import FalsePositives, Precision, AUC
# CSV columns in the input file.
data_pd = pd.read_csv("pokemon.csv")
labels_pd = data_pd['isLegendary']*1

# Tensorboard callback 
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S")))


pokemonNames = data_pd['Name']
pokemonTypes = pd.concat([data_pd['Type_1'], data_pd['Type_2']]).unique()[:18]

for i in pokemonTypes:
	data_pd[i] = np.logical_or((data_pd['Type_1'] == i),(data_pd['Type_2'] == i))*1


data_pd = data_pd.drop(['Name', 'Number', 'Type_1', 'Type_2', 'HP', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Attack', 'Generation', 'Color', 'Body_Style', 'hasMegaEvolution', 'isLegendary', 'hasGender', 'Egg_Group_1', 'Egg_Group_2', 'Pr_Male', 'Catch_Rate'], axis=1)

data_np = data_pd.to_numpy()
labels_np = labels_pd.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(data_np, labels_np, test_size=0.30, random_state=1234)

train_ds = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_ds = tf.data.Dataset.from_tensor_slices(y_train)

test_ds = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_ds = tf.data.Dataset.from_tensor_slices(y_test)

train_ds = tf.data.Dataset.zip((train_ds, train_labels_ds)) \
	.batch(batch_size=1000)

test_ds = tf.data.Dataset.zip((test_ds, test_labels_ds)) \
	.batch(batch_size=1000)


model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(26, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(256, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model.compile(optimizer=Adam(0.0001),
              loss='binary_crossentropy',
              metrics=[FalsePositives(thresholds=0.5), AUC()])
            
model.fit(train_ds, epochs=50, validation_data=test_ds, callbacks=[tensorboard_callback])
