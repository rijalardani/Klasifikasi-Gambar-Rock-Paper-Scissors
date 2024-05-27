# Rock Paper Scissors Image Classification

This project aims to develop an image classification system that can differentiate between images of hands showing rock, paper, and scissors symbols. The main goal of this project is to train a machine learning model that can recognize and distinguish these images with high accuracy. This project involves building an image classification model using TensorFlow and Keras. The primary objective is to classify images into different categories using a convolutional neural network (CNN).

## Project Steps
The project follows these primary steps:

1. **Initial Setup and Library Import**:
    - Import necessary libraries such as TensorFlow, Keras, and others.
    
    ```python
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    ```

2. **Data Loading and Preparation**:
    - Download the dataset and explore its structure.
    - Split the dataset into training, validation, and testing sets.

    ```python
    _URL = 'https://dataset.url'
    path_to_zip = tf.keras.utils.get_file('dataset.zip', origin=_URL, extract=True)
    base_dir = os.path.join(os.path.dirname(path_to_zip), 'dataset')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    ```

3. **Data Preprocessing**:
    - Perform data augmentation to increase the diversity of the training data and reduce overfitting.
    - Normalize the data to improve model efficiency.

    ```python
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=40, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    ```

4. **Model Building**:
    - Define the model architecture using Keras, including convolutional, pooling, and dense layers.
    - Compile the model with an appropriate optimizer, loss function, and evaluation metrics.

    ```python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Additional layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```

5. **Model Training**:
    - Train the model using the training data and validate it using the validation data.
    - Use callbacks such as early stopping to prevent overfitting.

    ```python
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    ```

6. **Model Evaluation**:
    - Evaluate the model's performance using the test data.
    - Calculate evaluation metrics such as accuracy and display the confusion matrix.

    ```python
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {accuracy}")
    ```

7. **Model Saving**:
    - Save the trained model for future use.
    - Perform final testing and analysis of results.

    ```python
    model.save('my_model.h5')
    ```

## Tools and Libraries Used
- **TensorFlow**: An open-source deep learning framework developed by Google.
- **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.
- **NumPy**: A fundamental package for scientific computing with Python.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.
- **Pandas**: A fast, powerful, flexible, and easy-to-use open-source data analysis and data manipulation library.

## Conclusion
This project serves as an example of applying machine learning techniques to solve image classification problems using the rock-paper-scissors dataset as the source of training and testing data.
