#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import GPyOpt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart.csv')
X = data.drop(columns=['target']).values
y = data['target'].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

plt.hist(y, bins=2, edgecolor='k')
plt.xticks([0, 1])
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.title('Distribution of Target')
plt.show()

def objective_function(hyperparameters):
    lr, units, dropout, l2_reg, batch_size = hyperparameters[0]
    units = int(units)
    batch_size = int(batch_size)
    model = Sequential()
    model.add(Dense(units, activation='relu',
                    kernel_regularizer=l2(l2_reg),
                    input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout))
    model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    filename = f"checkpoint_lr{lr:.5f}_units{units}_dropout{dropout:.2f}_l2{l2_reg:.5f}_batch{batch_size}.h5"
    checkpoint = ModelCheckpoint(filename, monitor='val_accuracy', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )
    val_loss = min(history.history['val_loss'])
    return val_loss

domain = [
    {'name': 'lr', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'units', 'type': 'discrete', 'domain': (8, 16, 32, 64)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (0.0, 0.01)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (8, 16, 32, 64)}
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=domain,
    acquisition_type='EI'
)

optimizer.run_optimization(max_iter=30)

with open('bayes_opt.txt', 'w') as f:
    f.write(f"Best hyperparameters: {optimizer.x_opt}\n")
    f.write(f"Best validation loss: {optimizer.fx_opt}\n")

optimizer.plot_convergence()
plt.savefig('convergence_plot.png')
plt.show()

print("Best hyperparameters:", optimizer.x_opt)
print("Best validation loss:", optimizer.fx_opt)
