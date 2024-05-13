# %% [markdown]
# # Лабораторная работа №6
# Lidar. Tree Classification.

# %%
# Loading extension for reloading editable packages (pip install -e .)
# %load_ext autoreload

# %% [markdown]
"""
Вариант: 14 (чётный => данные h5.v**2**)
"""

# %%
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from tqdm import tqdm

# %matplotlib inline

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
"""
## Tree Classification with Lidar
### Загрузка набора данных
"""

# %% [markdown]
# Подготовка модулей и переменных для работы с файлами.

# %%
import os  # noqa
import shutil  # noqa
import urllib  # noqa
from pathlib import Path  # noqa

data_path = Path("data")
external_data_path = data_path / "external"
interim_data_path = data_path / "interim"
processed_data_path = data_path / "processed"

external_data_path.mkdir(exist_ok=True)
interim_data_path.mkdir(exist_ok=True)
processed_data_path.mkdir(exist_ok=True)

# %% [markdown]
# Подготовка модулей и переменных для загрузки данных.

# %%
url = "https://github.com/iu5git/Deep-learning/raw/main/datasets/lidar/v2.h5"
dataset_filename = "v2.h5"

external_data_path.mkdir(exist_ok=True)

file_path = external_data_path / dataset_filename

# %% [markdown]
# Загрузка набора данных.
# Каждая точка одного облака точек содержит информацию о 3 различных координатах,
# данный формат удобен для хранения и работы с таким большим объемом данных.

# %%
if not os.path.isfile(file_path):
    print(urllib.request.urlretrieve(url, file_path))


# %% [markdown]
# ### Чтение тренировочной и тестовой выборки

# %%
import h5py

h5f = h5py.File(file_path,'r') # файл по вашему варианту
X = h5f.get('dataset_X')[:]
Y = h5f.get('dataset_Y').asstr()[:]
h5f.close()

# %%
NUM_POINTS = 4096
BATCH_SIZE = 64

unique_classes = np.unique(Y)
CLASSES = {i: tree_class for i, tree_class in enumerate(unique_classes)}
CLASSES

# %%
Y = np.array([list(CLASSES.values()).index(y) for y in Y])
indexes = []
[indexes.append(y) for y in list(Y) if y not in indexes]
indexes.sort()
CLASS_MAP = {i: CLASSES[k] for (k, i) in (zip(indexes, range(len(indexes))))}

#кол-во классов по вашему варианту
NUM_CLASSES = len(CLASS_MAP)

for (k, i) in (zip(indexes, range(len(indexes)))):
  Y[Y == k] = i

# %%
points = X[50]

fig = plt.figure(figsize=(5, 5))
# 111 = size, size, number of cell
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
plt.show()

# %%
import pandas as pd

data = {'Кол-во деревьев': list(pd.value_counts(Y).sort_index()),
        'Деревья': list(CLASS_MAP.values())}
df = pd.DataFrame(data).set_index('Деревья')
ax = df.plot.bar()

# %% [markdown]
# ### Разбиение выборки на тренировочную и тестовую

# %%
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5).split(X, Y)

for train_index, test_index in skf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

X_augment = []
y_augment = []

for i in range(4):
    point_select = []
    for x in X_train:
        idx = np.random.choice(NUM_POINTS, size=NUM_POINTS, replace=True)
        point_select.append(x[idx])
    point_select = np.array(point_select)        
    point_select = point_select + np.random.normal(0, 0.005, point_select.shape)
    X_augment.append(point_select)
    y_augment.append(y_train)

X_augment = np.array(X_augment)
y_augment = np.array(y_augment)
X_augment = np.reshape(X_augment,(X_augment.shape[0] * X_augment.shape[1], NUM_POINTS, 3))
y_augment = np.reshape(y_augment,(-1))

# %%
import tensorflow as tf

tf.random.set_seed(42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_augment, y_augment))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(len(X_augment)).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(X_test)).batch(BATCH_SIZE)

# %% [markdown]
# ### Построение модели
# Каждый сверточный и полносвязный слой (не включая конечных слоев) состоит из Convolution / Dense -> Batch Normalization -> ReLU Activation.

# %%
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

# %% [markdown]
# PointNet состоит из двух основных компонентов:
# основная сеть MLP (многослойный перцептрон) и трансформаторная сеть T-net.

# %%
from tensorflow import keras

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

# %% [markdown]
# Определим общую функцию для построения слоев T-net.

# %%
def tnet(inputs, num_features):

    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

# %%
from tensorflow.keras import layers

inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

# %% [markdown]
# ### Обучение модели

# %%
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# %% [markdown]
# ### Визуализация

# %%
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

fig = plt.figure(figsize=(15, 15))
for i in range(8):
    ax = fig.add_subplot(4, 2, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()

# %%
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %%
from sklearn.metrics import confusion_matrix

data = test_dataset.take(1)
points, labels = list(data)[0]
points = points[:, ...]
labels = labels[:, ...]

preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

cm = confusion_matrix(y_true=labels, y_pred=preds)

# %%
plot_confusion_matrix(cm=cm, classes=CLASS_MAP.values(), title='Confusion Matrix')

