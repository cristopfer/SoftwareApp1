import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Configuración
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5


def load_dataset():
    """Carga el dataset de flores y devuelve info de clases."""
    ds, info = tfds.load('tf_flowers', split='train', with_info=True, as_supervised=True)
    class_names = info.features['label'].names
    return ds, class_names

def save_sample_images(num_images_per_class):
    """Guarda imágenes de ejemplo de cada clase en una carpeta."""
    ds, class_names = load_dataset()
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)

    # Limpiar carpeta de imágenes previas
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))

    # Guardar imágenes de cada clase
    saved_images = []
    for class_id, class_name in enumerate(class_names):
        class_ds = ds.filter(lambda image, label: label == class_id)
        for i, (image, label) in enumerate(class_ds.take(num_images_per_class)):
            image_path = f"{output_dir}/{class_name}_{i}.jpg"
            plt.imsave(image_path, image.numpy())
            saved_images.append({"class": class_name, "path": image_path})

    return saved_images

def load_and_preprocess_data():
    """Carga y preprocesa el dataset de flores con 100 imágenes por clase."""
    (ds_train, ds_test), info = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True
    )
    
    class_names = info.features['label'].names
    num_classes = len(class_names)
    IMAGES_PER_CLASS = 100  # Límite de imágenes por clase
    TRAIN_RATIO = 0.8  # 80% train, 20% test

    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    # Función para filtrar y limitar imágenes por clase
    def filter_and_limit(ds, class_id, limit):
        return ds.filter(lambda img, lbl: lbl == class_id).take(limit).map(preprocess)

    # Procesar train y test por clase
    train_datasets = []
    test_datasets = []
    
    for class_id in range(num_classes):
        train_limit = int(IMAGES_PER_CLASS * TRAIN_RATIO)  # 80
        test_limit = IMAGES_PER_CLASS - train_limit  # 20
        
        train_datasets.append(filter_and_limit(ds_train, class_id, train_limit))
        test_datasets.append(filter_and_limit(ds_test, class_id, test_limit))

    # Concatenar datasets (correctamente, de a 2)
    def concatenate_datasets(datasets):
        if len(datasets) == 1:
            return datasets[0]
        combined = datasets[0]
        for ds in datasets[1:]:
            combined = combined.concatenate(ds)
        return combined

    ds_train_final = concatenate_datasets(train_datasets) \
                     .shuffle(8 * IMAGES_PER_CLASS * num_classes) \
                     .batch(BATCH_SIZE) \
                     .prefetch(tf.data.AUTOTUNE)
    
    ds_test_final = concatenate_datasets(test_datasets) \
                    .batch(BATCH_SIZE) \
                    .prefetch(tf.data.AUTOTUNE)

    return ds_train_final, ds_test_final, class_names

def build_model():
    """Construye el modelo con Transfer Learning usando MobileNetV2."""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Congelar las capas base

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_model():
    """Entrena el modelo y lo guarda en formato .h5."""
    ds_train, ds_test, class_names = load_and_preprocess_data()
    model = build_model()

    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=10
    )

    model.save('static/model.h5')
    return class_names

def predict_flower(image_path):
    """Predice la clase de una imagen nueva."""
    model = tf.keras.models.load_model('static/model.h5')
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return predicted_class