import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout, Input
from keras import Model

class MultiClassEfficientNetB0_Trackbed:
    def __init__(self):
        self.n_layers = 1
        self.t_height = 224
        self.t_width = 224
        self.t_depth = 3

    def build(self, initialLearningRate, optimizer, lossFunction, label_map=[]):
        input_layer = Input(shape=(self.t_height, self.t_width, self.t_depth))
        x = input_layer

        # Build base model without pretrained weights
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(self.t_height, self.t_width, self.t_depth),
            pooling='avg'
        )

        # Base model is trainable (no freezing)
        base_model.trainable = True

        # Connect input directly to the base model
        x = base_model(x)

        # Add classification layers
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(5, activation='softmax')(x)  # Changed from sigmoid to softmax

        # Create the model
        model = Model(inputs=input_layer, outputs=outputs)

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