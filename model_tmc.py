import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPooling2D,
                                     concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import ImageFile, Image
from keras import regularizers


# 相对路径不用/data/train.csv，而是应该data/train.csv即可
train_dataset = pd.read_csv('data/train.csv')
test_dataset = pd.read_csv('data/test.csv')

n = len(train_dataset)
train_n = int(0.8*n)
train_data = train_dataset[:train_n]
val_data = train_dataset[train_n:]
test_data = test_dataset[:100]
print(train_data.shape, val_data.shape, test_data.shape)


# 超参数
batch_size = 32
epochs = 15
# set image dimensions
img_width = 224
img_height = 224
# set text parameters
max_words = 10000
max_len = 200


# load and preprocess image data
def load_image(path):
    img = load_img(path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = img / 255.0
    return img


def preprocess_images(data):
    images = []
    for path in data['image_name']:
        path = os.path.join('data/Memes', path)
        img = load_image(path)
        images.append(img)
    images = np.array(images)
    return images


ImageFile.LOAD_TRUNCATED_IMAGES = True
train_images = preprocess_images(train_data)
val_images = preprocess_images(val_data)
test_images = preprocess_images(test_data)


# load and preprocess text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['Captions'])

train_sequences = tokenizer.texts_to_sequences(train_data['Captions'])
val_sequences = tokenizer.texts_to_sequences(val_data['Captions'])
test_sequences = tokenizer.texts_to_sequences(test_data['Captions'])

train_text = pad_sequences(train_sequences, maxlen=max_len)
val_text = pad_sequences(val_sequences, maxlen=max_len)
test_text = pad_sequences(test_sequences, maxlen=max_len)


# train labels
labels = train_data['Label']
train_labels = tf.keras.utils.to_categorical(labels, num_classes=3)

# test labels
v_labels = val_data['Label']
val_labels = tf.keras.utils.to_categorical(v_labels, num_classes=3)


# create model1
# Define the image input layer
input_img_x = Input(shape=(img_width, img_height, 3))  # shape=(224,224,3)
x = Conv2D(32, (5, 5), activation='relu')(input_img_x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (5, 5), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
# x = Conv2D(32, (5, 5), activation='relu')(x)
# x = MaxPooling2D((2, 2))(x)
x = GlobalMaxPooling2D()(x)

# Define the text input layer
input_text_y = Input(shape=(max_len,))  # shape=(200, )
y = Embedding(max_words, 128)(input_text_y)
y = Conv1D(32, 5, activation='relu')(y)
y = MaxPooling1D(5)(y)
y = Conv1D(32, 5, activation='relu')(y)
# y = MaxPooling1D(3)(y)
# y = Conv1D(32, 5, activation='relu')(y)
y = MaxPooling1D(5)(y)
y = GlobalMaxPooling1D()(y)


# create model2
# Define the image input layer
input_img_f = Input(shape=(img_width, img_height, 3))  # shape=(224,224,3)
conv1 = Conv2D(32, (3, 3), activation='relu')(input_img_f)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
# conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
# pool3 = MaxPooling2D((2, 2))(conv3)
flatten1 = Flatten()(pool2)

# Define the text input layer
input_text_l = Input(shape=(max_len,))  # shape=(200, )
# Define the embedding layer for the text input
embedding = Embedding(input_dim=10000, output_dim=50)(input_text_l)
lstm = LSTM(128)(embedding)


# Concatenate the output of the image and text inputs
# merged = concatenate([flatten1, lstm])
merged = concatenate([x, y])  # 效果不好啊
merged = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.05))(merged)
merged = Dropout(0.2)(merged)
merged = Dropout(0.5)(merged)
merged = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.05))(merged)
merged = Dropout(0.2)(merged)
merged = Dropout(0.5)(merged)
output = Dense(3, activation='softmax')(merged)

# Create the model
model = Model(inputs=[input_img_x, input_text_y], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
print(model.summary())


# Train the model with the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)
checkpoint = ModelCheckpoint('model_tmc.keras', save_best_only=True)
model_history = model.fit([train_images, train_text], train_labels, batch_size=batch_size, epochs=100,
                          validation_data=([val_images, val_text], val_labels), callbacks=[checkpoint, early_stopping])


# 画图
history_dict = model_history.history
print(history_dict.keys())
acc = history_dict['auc_14']
val_acc = history_dict['val_auc_14']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_x = range(1, len(acc)+1)


fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# 第一幅图：训练和验证损失
axs[0].plot(epoch_x, loss, 'r-', label='Training Loss')
axs[0].plot(epoch_x, val_loss, 'g', label='Validation Loss')
axs[0].set_title('Training and Validation loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# 第二幅图：训练和验证 AUC
axs[1].plot(epoch_x, acc, 'y', label='Training AUC')
axs[1].plot(epoch_x, val_acc, 'b', label='Validation AUC')
axs[1].set_title('Training and Validation AUC')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Area Under the Curve')
axs[1].legend()

fig.tight_layout()  # 调整子图之间的距离
plt.show()




