import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import json
import pickle
import numpy as np


early_stop = EarlyStopping(
    monitor="val_dice_coef",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1             # 'max' because Dice should increase
)

checkpoint = ModelCheckpoint(
    filepath="best_model_transfer_learning_25.keras",
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
    verbose=1
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

    def build_model(self):
        """
        Build and compile the U-Net model with input-level BatchNormalization and data augmentation.
        """
        inputs = layers.Input(shape=self.input_shape, name="input_layer")

        # Step 1: Normalize input
        x = layers.BatchNormalization(name="input_batch_norm")(inputs)

        # Step 3: Build encoder from normalized + augmented input
        encoder_output, skips = self._build_encoder(x)
        # self.encoder = models.Model(inputs, encoder_output, name="encoder")
        self.encoder = models.Model(x, encoder_output, name="encoder")

        # Step 4: Load weights (if provided)
        if self.encoder_weights_path:
            print(f"Loading encoder weights from: {self.encoder_weights_path}")
            pretrained_encoder = load_model(self.encoder_weights_path)
            for target_layer, source_layer in zip(self.encoder.layers, pretrained_encoder.layers):
                if target_layer.weights and source_layer.weights:
                    if all(t.shape == s.shape for t, s in zip(target_layer.get_weights(), source_layer.get_weights())):
                        target_layer.set_weights(source_layer.get_weights())

            # self.encoder.set_weights(pretrained_encoder.get_weights())

        # Step 5: Freeze encoder (if requested)
        if self.freeze_encoder:
            print("Freezing encoder layers.")
            for layer in self.encoder.layers:
                layer.trainable = False

        # Step 6: Build decoder and compile model
        outputs = self._build_decoder(encoder_output, skips)
        model = models.Model(inputs, outputs, name="unet")
        model.compile(optimizer=optimizers.Adam(),
                      loss=self.loss,
                      metrics=self.metrics)

        return model

    def _build_encoder(self, inputs):
        skips = []

        # Block 1
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

        # Block 2
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

        # Block 3
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

        # Block 4
        x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

        # Bottleneck
        x = layers.Conv2D(1024, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(1024, 3, activation='relu', padding='same')(x)

        return x, skips

    def _build_encoder_simple(self, inputs):
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
        # Decoder Block 1: 8x8 → 16x16
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up1")(x)
        x = layers.Concatenate(name="dec_concat1")([x, skips[-1]])
        x = layers.Dropout(0.3, name="dec_dropout1")(x)
        x = layers.Conv2D(512, 3, activation='relu', padding='same', name="dec1_conv1")(x)
        x = layers.Conv2D(512, 3, activation='relu', padding='same', name="dec1_conv2")(x)

        # Decoder Block 2: 16x16 → 32x32
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up2")(x)
        x = layers.Concatenate(name="dec_concat2")([x, skips[-2]])
        x = layers.Dropout(0.3, name="dec_dropout2")(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same', name="dec2_conv1")(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same', name="dec2_conv2")(x)

        # Decoder Block 3: 32x32 → 64x64
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up3")(x)
        x = layers.Concatenate(name="dec_concat3")([x, skips[-3]])
        x = layers.Dropout(0.3, name="dec_dropout3")(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same', name="dec3_conv1")(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same', name="dec3_conv2")(x)

        # Decoder Block 4: 64x64 → 128x128
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up4")(x)
        x = layers.Concatenate(name="dec_concat4")([x, skips[-4]])
        x = layers.Dropout(0.3, name="dec_dropout4")(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name="dec4_conv1")(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name="dec4_conv2")(x)

        # Decoder Block 5: 128x128 → 256x256 (no skip connection)
        # x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up5")(x)
        # x = layers.Conv2D(32, 3, activation='relu', padding='same', name="dec5_conv1")(x)
        # x = layers.Conv2D(32, 3, activation='relu', padding='same', name="dec5_conv2")(x)

        # Output Layer
        output = layers.Conv2D(self.num_classes, 1, activation='sigmoid', name="output_layer")(x)

        return output

    def _build_decoder_v2(self, x, skips):
        # Decoder Block 1
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up1")(x)
        x = layers.Concatenate(name="dec_concat1")([x, skips[-1]])
        x = layers.Dropout(0.3, name="dec_dropout1")(x)
        x = layers.Conv2D(512, 3, activation='relu', padding='same', name="dec1_conv1")(x)
        x = layers.Conv2D(512, 3, activation='relu', padding='same', name="dec1_conv2")(x)

        # Decoder Block 2
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up2")(x)
        x = layers.Concatenate(name="dec_concat2")([x, skips[-2]])
        x = layers.Dropout(0.3, name="dec_dropout2")(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same', name="dec2_conv1")(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same', name="dec2_conv2")(x)

        # Decoder Block 3
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up3")(x)
        x = layers.Concatenate(name="dec_concat3")([x, skips[-3]])
        x = layers.Dropout(0.3, name="dec_dropout3")(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same', name="dec3_conv1")(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same', name="dec3_conv2")(x)

        # Decoder Block 4
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up4")(x)
        x = layers.Concatenate(name="dec_concat4")([x, skips[-4]])
        x = layers.Dropout(0.3, name="dec_dropout4")(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name="dec4_conv1")(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name="dec4_conv2")(x)

        # Output Layer
        output = layers.Conv2D(self.num_classes, 1, activation='sigmoid', name="output_layer")(x)

        return output

    def _build_decoder_simple(self, x, skips):
        # Decoder Block 1
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up1")(x)
        x = layers.Concatenate(name="dec_concat1")([x, skips[-1]])
        x = layers.Dropout(0.3, name="dec_dropout1")(x)  # NEW
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name="dec1_conv1")(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name="dec1_conv2")(x)

        # Decoder Block 2
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec_up2")(x)
        x = layers.Concatenate(name="dec_concat2")([x, skips[-2]])
        x = layers.Dropout(0.3, name="dec_dropout2")(x)  # NEW
        x = layers.Conv2D(32, 3, activation='relu', padding='same', name="dec2_conv1")(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same', name="dec2_conv2")(x)

        # Output Layer
        output = layers.Conv2D(self.num_classes, 1, activation='sigmoid', name="output_layer")(x)

        return output

    def _build_decoder_v1(self, x, skips):
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

    def build_model_v1(self):
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

    def build_model_v2(self):
        """
        Build and compile the U-Net model.
        """
        inputs = layers.Input(shape=self.input_shape)

        # Build encoder and get intermediate skips
        encoder_output, skips = self._build_encoder(inputs)

        # Save encoder as its own model
        self.encoder = models.Model(inputs, encoder_output, name="encoder")

        # Build decoder from encoder output
        outputs = self._build_decoder(encoder_output, skips)

        # Full U-Net model
        model = models.Model(inputs, outputs, name="unet")
        model.compile(optimizer=optimizers.Adam(),
                      loss=self.loss,
                      metrics=self.metrics)

        return model

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        return self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val) if X_val is not None else None,
            callbacks=self.callbacks,
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

    def _build_encoder_with_loaded_weights(self, encoder_path, inputs):

        # Step 1: Build the encoder architecture exactly as in _build_encoder
        skips = []

        x = layers.Conv2D(32, 3, activation='relu', padding='same', name='enc_conv1')(inputs)
        x = layers.Conv2D(32, 3, activation='relu', padding='same', name='enc_conv2')(x)
        skips.append(x)
        x = layers.MaxPooling2D(name='enc_pool1')(x)

        x = layers.Conv2D(64, 3, activation='relu', padding='same', name='enc_conv3')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name='enc_conv4')(x)
        skips.append(x)
        x = layers.MaxPooling2D(name='enc_pool2')(x)

        x = layers.Conv2D(128, 3, activation='relu', padding='same', name='enc_conv5')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same', name='enc_conv6')(x)

        # Step 2: Create encoder model from inputs → output
        encoder_model = models.Model(inputs, x, name="encoder_with_skips")

        # Step 3: Load weights from MAE encoder (only shared layers will load)
        pretrained_encoder = load_model(encoder_path)
        encoder_model.set_weights(pretrained_encoder.get_weights())  # Must match structure

        return x, skips, encoder_model










#    def _build_encoder(self, inputs):
#        skips = []
#
#        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
#        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
#        skips.append(x)
#        x = layers.MaxPooling2D()(x)
#
#        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
#        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
#        skips.append(x)
#        x = layers.MaxPooling2D()(x)
#
#        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
#        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
#
#        return x, skips
#
#    def _build_decoder(self, x, skips):
#        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
#        x = layers.Concatenate()([x, skips[-1]])
#        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
#        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
#
#        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
#        x = layers.Concatenate()([x, skips[-2]])
#        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
#        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
#
#        output = layers.Conv2D(self.num_classes, 1, activation='sigmoid')(x)
#        return output
#
#    def build_model(self):
#        """
#        Build and compile the U-Net model.
#        """
#        inputs = layers.Input(shape=self.input_shape)
#        encoder_output, skips = self._build_encoder(inputs)
#        outputs = self._build_decoder(encoder_output, skips)
#        model = models.Model(inputs, outputs)
#        model.compile(optimizer=optimizers.Adam(),
#                      loss=self.loss,
#                      metrics=self.metrics)
#        return model
#
#    def fit(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
#        return self.model.fit(
#            X_train, Y_train,
#            validation_data=(X_val, Y_val) if X_val is not None else None,
#            # callbacks=self.callbacks,
#            **kwargs
#        )
#
#    def evaluate(self, X_test, Y_test, **kwargs):
#        """
#        Evaluate the model on test data.
#        """
#        return self.model.evaluate(X_test, Y_test, **kwargs)
#
#    def predict(self, X, **kwargs):
#        """
#        Predict segmentation masks from input images.
#        """
#        return self.model.predict(X, verbose=0, **kwargs)
#
#    def summary(self):
#        """
#        Print a summary of the model architecture.
#        """
#        return self.model.summary()
#
#    def save(self, filepath):
#        """
#        Save the entire model to a file.
#        """
#        self.model.save(filepath)
#        print(f"Model saved to {filepath}")
#
#    @classmethod
#    def load(cls, filepath):
#        """
#        Load a saved model and return a UNetSegmentationPipeline instance.
#        """
#        model = models.load_model(filepath,
#                                  custom_objects={
#                                      'dice_coef': dice_coef,
#                                      'iou_metric': iou_metric,
#                                      'combined_loss': combined_loss,
#                                      'dice_loss': dice_loss
#                                  })
#        instance = cls()
#        instance.model = model
#        return instance
#
#    def save_training_history(self, history, filename, format='json'):
#        """
#        Save training history to a file.
#
#        Args:
#            history: Keras History object (from model.fit()).
#            filename: File path without extension.
#            format: 'json' (default) or 'pkl'
#        """
#        history_data = history.history
#        if format == 'json':
#            with open(f"{filename}.json", "w") as f:
#                json.dump(history_data, f)
#            print(f"Training history saved to {filename}.json")
#        elif format == 'pkl':
#            with open(f"{filename}.pkl", "wb") as f:
#                pickle.dump(history_data, f)
#            print(f"Training history saved to {filename}.pkl")
#        else:
#            raise ValueError("Format must be 'json' or 'pkl'")
#
#    def load_training_history(self, filepath, format='json'):
#        """
#        Load saved training history from a file.
#
#        Args:
#            filepath: Full path to history file (without extension).
#            format: 'json' or 'pkl'
#
#        Returns:
#            history_dict: Dictionary of training history
#        """
#        if format == 'json':
#            with open(f"{filepath}.json", "r") as f:
#                return json.load(f)
#        elif format == 'pkl':
#            with open(f"{filepath}.pkl", "rb") as f:
#                return pickle.load(f)
#        else:
#            raise ValueError("Format must be 'json' or 'pkl'")
