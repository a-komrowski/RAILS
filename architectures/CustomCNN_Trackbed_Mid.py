import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras import Model

class CustomCNN_Trackbed_Mid:
    def __init__(self):
        self.n_layers = 1
        self.t_height = 224
        self.t_width = 224
        self.t_depth = 3

    def build(self, initialLearningRate, optimizer, lossFunction, label_map=[]):
        inputs = Input(shape=(self.t_height, self.t_width, self.t_depth))
        
        # Convolutional layers with decreasing filters
        x = Conv2D(256, (3, 3), activation="relu")(inputs)
        x = MaxPooling2D(2, 2)(x)
        
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = MaxPooling2D(2, 2)(x)
        
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = MaxPooling2D(2, 2)(x)
        
        # Classification layers
        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(5, activation="softmax")(x)  # Changed from sigmoid to softmax
        
        # Create the model
        model = Model(inputs, outputs)
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(initialLearningRate),
            decay_steps=10000,
            decay_rate=0.9
        )
        
        # Configure optimizer
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=lr_schedule)
        else:
            opt = Adam(learning_rate=lr_schedule)
        
        # Define metrics
        metric = tf.keras.metrics.CategoricalAccuracy(
            name='categorical_accuracy'
        )
        
        # Compile the model
        model.compile(optimizer=opt, loss=lossFunction, metrics=[metric])
        model.summary()
        
        return model