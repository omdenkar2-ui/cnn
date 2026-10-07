import tensorflow as tf


def build_model(size=128, seed=42):
    """Reference-style Conv16/Conv32/Conv16 + Dense256 binary CNN."""
    layers = tf.keras.layers
    if size < 32:
        raise ValueError('Image size must be at least 32')
    model = tf.keras.Sequential([
        layers.Input((size,size,3)),
        layers.Rescaling(1./255),
        layers.RandomFlip('horizontal',seed=seed),
        layers.Conv2D(16,3,activation='relu',kernel_initializer='he_normal'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,activation='relu',kernel_initializer='he_normal'),
        layers.MaxPooling2D(),
        layers.Conv2D(16,3,activation='relu',kernel_initializer='he_normal'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256,activation='relu'),
        layers.Dropout(.3,seed=seed),
        layers.Dense(1,activation='sigmoid'),
    ],name='happy_sad_cnn')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
    return model
