import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import json
import pickle
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_dice_coef',   # or 'val_loss' if you prefer
    patience=5,
    restore_best_weights=True,
    mode='max'                 # 'max' because Dice should increase
)

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_dice_coef',
    save_best_only=True,
    mode='max'
)

# -----------------------
# Custom Metrics and Loss
# -----------------------

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# --------------------------
# UNetSegmentationPipeline
# --------------------------

class UNetSegmentationPipeline:
    def __init__(self, input_shape=(256, 256, 1), num_classes=1,
                 encoder_weights_path=None, freeze_encoder=False):
        """
        Initialize the U-Net segmentation pipeline.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder_weights_path = encoder_weights_path
        self.freeze_encoder = freeze_encoder
        self.loss = combined_loss
        self.metrics = [dice_coef, iou_metric]
        self.callbacks = [early_stop, checkpoint]
        self.model = self.build_model()

    def _build_encoder(self, inputs):
        skips = []

        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

        return x, skips

    def _build_decoder(self, x, skips):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skips[-1]])
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skips[-2]])
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

        output = layers.Conv2D(self.num_classes, 1, activation='sigmoid')(x)
        return output

    def build_model(self):
        """
        Build and compile the U-Net model.
        """
        inputs = layers.Input(shape=self.input_shape)
        encoder_output, skips = self._build_encoder(inputs)
        outputs = self._build_decoder(encoder_output, skips)
        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(),
                      loss=self.loss,
                      metrics=self.metrics)
        return model

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        return self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val) if X_val is not None else None,
            # callbacks=self.callbacks,
            **kwargs
        )

    def evaluate(self, X_test, Y_test, **kwargs):
        """
        Evaluate the model on test data.
        """
        return self.model.evaluate(X_test, Y_test, **kwargs)

    def predict(self, X, **kwargs):
        """
        Predict segmentation masks from input images.
        """
        return self.model.predict(X, verbose=0, **kwargs)

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        return self.model.summary()

    def save(self, filepath):
        """
        Save the entire model to a file.
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a saved model and return a UNetSegmentationPipeline instance.
        """
        model = models.load_model(filepath,
                                  custom_objects={
                                      'dice_coef': dice_coef,
                                      'iou_metric': iou_metric,
                                      'combined_loss': combined_loss,
                                      'dice_loss': dice_loss
                                  })
        instance = cls()
        instance.model = model
        return instance

    def save_training_history(self, history, filename, format='json'):
        """
        Save training history to a file.

        Args:
            history: Keras History object (from model.fit()).
            filename: File path without extension.
            format: 'json' (default) or 'pkl'
        """
        history_data = history.history
        if format == 'json':
            with open(f"{filename}.json", "w") as f:
                json.dump(history_data, f)
            print(f"Training history saved to {filename}.json")
        elif format == 'pkl':
            with open(f"{filename}.pkl", "wb") as f:
                pickle.dump(history_data, f)
            print(f"Training history saved to {filename}.pkl")
        else:
            raise ValueError("Format must be 'json' or 'pkl'")

    def load_training_history(self, filepath, format='json'):
        """
        Load saved training history from a file.

        Args:
            filepath: Full path to history file (without extension).
            format: 'json' or 'pkl'

        Returns:
            history_dict: Dictionary of training history
        """
        if format == 'json':
            with open(f"{filepath}.json", "r") as f:
                return json.load(f)
        elif format == 'pkl':
            with open(f"{filepath}.pkl", "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Format must be 'json' or 'pkl'")
