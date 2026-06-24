

def build_model(
    num_classes: int,
    input_shape: tuple = (224, 224, 3),
    load_weights: bool = True,
):
    """
    Build a plant disease classifier using EfficientNetB0 transfer learning.

    Args:
        num_classes:  number of output classes (38 for full PlantVillage)
        input_shape:  image dimensions expected by the model
        load_weights: load ImageNet weights. Set False in tests/CI to skip download.

    Returns:
        Keras Model (not yet compiled — call model.compile() after)
    """
    import tensorflow as tf  # lazy import
    from tensorflow.keras import layers, Model

    weights = "imagenet" if load_weights else None

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base.trainable = False  # freeze backbone — train only the head first

    inputs = tf.keras.Input(shape=input_shape)
    # EfficientNet expects pixel values in [0, 255] — do NOT rescale to [0, 1]
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="plant_disease_efficientnet")
    return model


def unfreeze_top_layers(model, num_layers: int = 20):
    """
    Fine-tune: unfreeze the top N layers of EfficientNetB0 after initial training.
    Call after the head has converged (~epoch 10-15), then retrain with low LR (1e-5).
    """
    base = model.layers[1]  # EfficientNetB0 is the second layer
    base.trainable = True

    for layer in base.layers[:-num_layers]:
        layer.trainable = False

    return model


if __name__ == "__main__":
    m = build_model(num_classes=38, load_weights=False)
    m.summary()
