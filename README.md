# Flower Classification Using CNN and Data Augmentation

## Overview

This project aims to classify images of flowers using a Convolutional Neural Network (CNN). We use data augmentation to address overfitting, thereby improving the model's generalization to new data.

## Dataset

The dataset contains images of five types of flowers:
- Roses
- Daisies
- Dandelions
- Sunflowers
- Tulips

The dataset is sourced from TensorFlow's official dataset repository.

## Dependencies

- matplotlib
- numpy
- cv2
- PIL
- tensorflow
- pathlib
- sklearn

## Data Loading and Preprocessing

### Loading Dataset

```python
import pathlib
import tensorflow as tf

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir)
```

### Organizing Images and Labels

```python
flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}
```

### Reading Images and Resizing

```python
import cv2
import numpy as np

X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (180, 180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)
```

### Splitting the Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
```

## Model Building and Training

### Initial Model (Without Data Augmentation)

```python
from tensorflow.keras import layers, Sequential

num_classes = 5

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(180, 180, 3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30)
```

### Model Evaluation

```python
model.evaluate(X_test_scaled, y_test)
```

## Addressing Overfitting with Data Augmentation

### Data Augmentation

```python
data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(180, 180, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
```

### Improved Model with Data Augmentation and Dropout

```python
model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30)
```

### Improved Model Evaluation

```python
model.evaluate(X_test_scaled, y_test)
```

## Results

- **Initial Model**: Training accuracy was 99%, but test accuracy was only 62.63%, indicating overfitting.
- **Improved Model**: After applying data augmentation and dropout, the test accuracy improved to 71.02%.

## Conclusion

Using data augmentation and dropout layers significantly improved the model's ability to generalize, reducing overfitting and increasing test accuracy.

## Credits

This project was based on TensorFlow's official [image classification tutorial](https://www.tensorflow.org/tutorials/images/classification).
