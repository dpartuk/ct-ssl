import tensorflow as tf

print("TensorFlow version:", tf.__version__)

physical_devices = tf.config.list_physical_devices()
gpus = tf.config.list_physical_devices('GPU')

print("All devices:", physical_devices)
print("GPUs found:", gpus)

if gpus:
    print("✅ GPU acceleration is enabled!")
else:
    print("⚠️ No GPU found. Running on CPU.")
