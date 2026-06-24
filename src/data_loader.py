import tensorflow as tf
from pathlib import Path


IMG_SIZE  = (224, 224)
BATCH_SIZE = 32
AUTOTUNE  = tf.data.AUTOTUNE


def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Data augmentation applied ONLY during training.
    Applied AFTER cache so each epoch sees different augmentations.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")


def load_datasets(data_dir: str, batch_size: int = BATCH_SIZE):
    """
    Load train and validation datasets from a directory structured as:
        data_dir/
            ClassName1/  image1.jpg ...
            ClassName2/  image1.jpg ...

    Args:
        data_dir:   path to dataset root (set via DATA_DIR env var in train.py)
        batch_size: images per batch

    Returns:
        train_ds, val_ds, class_names
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            f"Download PlantVillage from: https://www.kaggle.com/datasets/emmarex/plantdisease\n"
            f"Then set the DATA_DIR environment variable to its path."
        )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names

    augment = get_augmentation_layer()

    # FIX: cache RAW images first, THEN augment so each epoch gets fresh augmentations.
    # Old order was: augment → cache → shuffle (froze augmentations, defeating the purpose).
    train_ds = (
        train_ds
        .cache()                                                        # cache raw pixels
        .shuffle(1000, seed=42)
        .map(lambda x, y: (augment(x, training=True), y),             # augment after cache
             num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_ds = (
        val_ds
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    return train_ds, val_ds, class_names
