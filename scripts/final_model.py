import numpy as np
import tensorflow as tf
import pandas as pd
import json


# X_aug = np.load('../data/X_aug.npy')
# y_aug = np.load('../data/y_aug.npy')

# from sklearn.model_selection import train_test_split

# X_aug_train_val, X_aug_test, y_aug_train_val, y_aug_test = train_test_split(X_aug, y_aug, test_size=0.07, random_state=42)

# X_aug_train, X_aug_val, y_aug_train, y_aug_val = train_test_split(X_aug_train_val, y_aug_train_val, test_size=0.14, random_state=42)

classes_train_og = pd.read_csv('../data/train/_classes.csv', delimiter=',', index_col=0).to_numpy()
classes_test_og = pd.read_csv('../data/test/_classes.csv', delimiter=',', index_col=0).to_numpy()
classes_valid_og = pd.read_csv('../data/valid/_classes.csv', delimiter=',', index_col=0).to_numpy()

with open('../data/valid.json',"r") as f:
    data = json.load(f)
    valid_og = np.array(data)

with open('../data/test.json',"r") as f:
    data = json.load(f)
    test_og = np.array(data)

with open('../data/train.json',"r") as f:
    data = json.load(f)
    train_og = np.array(data)

X_train = train_og/255.0
y_train = classes_train_og

X_valid = valid_og/255.0
y_valid = classes_valid_og

X_test = test_og/255.0
y_test = classes_test_og

from keras.optimizers import Adam
class model_init:
    def __init__(self, classes, input_shape, num_dense_blocks, num_dense_units, num_conv_blocks, filters, kernel_size, pool_size, dropout_rate, padding, regularizer_strength, conv_activation, dense_activation):
        """
        Initialize the model with the given parameters
        """
        self.classes = classes
        self.input_shape = input_shape
        self.num_dense_blocks = num_dense_blocks
        self.num_dense_units = num_dense_units
        self.num_conv_blocks = num_conv_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.regularizer_strength = regularizer_strength
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation


        self.model = self.build_model()

    def conv_block(self, input, filters, kernel_size, pool_size):
        """
        Convolutional block with Conv2D, Activation and MaxPooling2D layers
        """
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=self.padding, activation=self.conv_activation)(input)
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=self.padding, activation=self.conv_activation)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size)(x)
        return x
    
    def dense_block(self, input, units, dropout):
        """
        Dense block with Dense, Activation and Dropout layers
        """
        x = tf.keras.layers.Dense(units, activation=self.dense_activation, kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_strength))(input)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x
    
    def build_model(self):
        """
        Build the model with the given parameters
        """
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = self.conv_block(input, self.filters, self.kernel_size, self.pool_size)
        for i in range(self.num_conv_blocks - 1):
            if x.shape[1] < 10 or x.shape[2] < 10:
                break
            x = self.conv_block(x, self.filters, self.kernel_size, self.pool_size)
        
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = self.conv_block(input, self.filters, self.kernel_size, self.pool_size)
        for i in range(self.num_conv_blocks - 1):
            x = self.conv_block(x, self.filters, self.kernel_size, self.pool_size)

        x = tf.keras.layers.Flatten()(x)
        
        for i in range(self.num_dense_blocks):
            x = self.dense_block(x, self.num_dense_units, self.dropout_rate)
        output = tf.keras.layers.Dense(self.classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=input, outputs=output)
        return model
    
opt_params = {
    "num_dense_blocks": 2,
    "num_dense_units": 448,
    "num_conv_blocks": 3,
    "filters": 160,
    "kernel_size": 4,
    "pool_size": 2,
    "dropout_rate": 0.5,
    "regularizer_strength": 0.06,
    "conv_activation": "elu",
    "dense_activation": "elu",
    "learning_rate": 0.00010964
}

from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='val_loss', patience=10)

final_model = model_init(6, input_shape=(X_train[0].shape[0], X_train[0].shape[1], 1), num_dense_blocks=opt_params["num_dense_blocks"], num_dense_units=opt_params["num_dense_units"], num_conv_blocks=opt_params["num_conv_blocks"], filters=opt_params["filters"], kernel_size=opt_params["kernel_size"], pool_size=opt_params["pool_size"], dropout_rate=opt_params["dropout_rate"], padding="same", regularizer_strength=opt_params["regularizer_strength"], conv_activation=opt_params["conv_activation"], dense_activation=opt_params["dense_activation"])

final_model = final_model.build_model()

print(final_model.summary())

# final_model.compile(optimizer=Adam(learning_rate=opt_params["learning_rate"]), loss="categorical_crossentropy", metrics=["accuracy", "AUC", "Precision"])

# history = final_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=16, callbacks=[callback])

# np.save('../data/history_final_model_olddata.npy', history.history)

# from sklearn.metrics import classification_report

# y_pred = final_model.predict(X_test)

# y_pred = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_test, axis=1)

# np.save('../data/y_pred_final_olddata.npy', y_pred)
# np.save('../data/y_true_final_olddata.npy', y_true)

# print(classification_report(y_true, y_pred))