# %% [markdown]
# # Лабораторная работа №1

# %%
# Loading extension for reloading editable packages (pip install -e .)
# %load_ext autoreload

# %%
# Reloading editable packages.
# %autoreload
# from lab1.main import get_results

# %% [markdown]
"""
Вариант для задания №3:
1. Номер группы + 15 = 2 + 15 = 17
2. Номер варианта + 56 = 14 + 56 = 70
3. ИУ5 (Номер варианта + 21) = 14 + 21 = 35
"""

# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import ArrayLike
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

# %matplotlib inline

# %% [markdown]
"""
## Часть 1. Задача регрессии по теореме универсальной аппроксимации, ручное дифференцирование
Генерация выборки и инициализация параметров нейронной сети
"""

# %%
from operator import itemgetter  # noqa

X = (np.arange(100) / 100 - 0.5).repeat(5)
# Наша функция, которую мы пытаемся получить в ходе апроксимации.
# y = 1 / (1 + np.exp(-10 * X))  # Исходная.
y = np.sin(X * 10) / 2
yn = np.random.normal(scale=0.05, size=y.size) + y

plt.plot(X, yn)
plt.plot(X, y, linestyle="--", c="k")
################################################

HIDDEN_SIZE = 64


def np_to_tensor(arr: ArrayLike):
    return torch.Tensor(arr.reshape(-1, 1))


tensor_X = np_to_tensor(X)
tensor_y = np_to_tensor(yn)


# Инициализация весов MLP с одним скрытым слоём
def init_neural_network(hidden_size=HIDDEN_SIZE):
    weights1 = (torch.rand(1, hidden_size) - 0.5) / 10
    bias1 = torch.zeros(hidden_size)

    weights2 = (torch.rand(hidden_size, 1) - 0.5) / 10
    bias2 = torch.zeros(1)

    return {"weights1": weights1, "bias1": bias1, "weights2": weights2, "bias2": bias2}


weights1, bias1, weights2, bias2 = itemgetter("weights1", "bias1", "weights2", "bias2")(
    init_neural_network()
)

# %% [markdown]
# ### Обучение нейронной сети задачи регрессии


# %%
# Определяем функцию нелинйности
def relu(x: torch.Tensor):
    return torch.maximum(x, torch.Tensor([0]))


# Прямой проход
def forward(x: torch.Tensor) -> torch.Tensor:
    return (weights2.t() * relu((weights1 * x) + bias1)).sum(
        axis=-1, keepdims=True
    ) + bias2


def loss(y: torch.Tensor, y_: torch.Tensor) -> torch.Tensor:
    return ((y - y_) ** 2).sum(axis=-1)


# обратный проход
def backward(X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor):
    # производная функции потерь по y_pred
    dL = 2 * (y_pred - y)
    # значения нейронов скрытго слоя до применения активации
    Ax = (weights1 * X) + bias1
    # значения нейронов скрытого слоя после применения активации
    A = relu(Ax)
    # производная функции потерь по weight_2
    dW2 = torch.mm(A.t(), dL)
    # производная функции потерь по bias_2
    db2 = dL.sum(axis=0)
    # производная функции потерь по значениям скрытого слоя после активации
    dA = torch.mm(dL, weights2.t())
    # производная функции потерь по значениям скрытого слоя до активации
    dA[Ax <= 0] = 0
    # производная функции потерь по weight_1
    dW1 = torch.mm(X.t(), dA)
    # производная функции потерь по bias_1
    db1 = dA.sum(axis=0)
    # print(dW.shape, db.shape, dW2.shape, db2.shape)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def optimize(params, grads, lr=0.001):
    # градиентный спуск по всей обучающей выборке
    W1, b1, W2, b2 = params
    W1 -= lr * grads["dW1"]
    W2 -= lr * grads["dW2"]
    b1 -= lr * grads["db1"]
    b2 -= lr * grads["db2"]

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# 50 тысяч итераций градиентного спуска == 50 тысяч эпох
for i in range(50000):
    output = forward(tensor_X)
    cur_loss = loss(output, tensor_y)
    grads = backward(tensor_X, tensor_y, output)
    params = [weights1, bias1, weights2, bias2]
    optimized_params = optimize(params, grads, 1e-4)
    weights1, bias1, weights2, bias2 = itemgetter("W1", "b1", "W2", "b2")(
        optimized_params
    )

    if (i + 1) % 10000 == 0:
        plt.plot(X, output.numpy(), label=str(i + 1), alpha=0.5)

plt.plot(X, y, linestyle="--", c="k", label="real")
plt.legend()
plt.ylim(y.min(), y.max())
print(cur_loss.numpy().mean())

# %% [markdown]
"""
## Часть 2. Бинарная классификация с помощью автодиффиренцирования PyTorch
Генерация выборки и инициализация параметров нейронной сети
"""

# %%
from sklearn.datasets import make_circles, make_moons  # noqa

# Данные, которые стараемся классифицировать:
# Исходная функция:
# X = np.random.randint(2, size=(1000, 2))
# y = (X[:, 0] + X[:, 1]) % 2  # XOR
# X = X + np.random.normal(0, scale=0.1, size=X.shape)

# Кольца, вложенные друг в друга.
# X, y = make_circles(n_samples=1000, noise=0.025)

# Вложенные друг в друга месяцы.
X, y = make_moons(n_samples=1000, noise=0.025)

plt.scatter(X[:, 0], X[:, 1], c=y)
####################################################
tensor_X = torch.Tensor(X.reshape(-1, 2))
tensor_y = torch.Tensor(y.reshape(-1, 1))

HIDDEN_SIZE = 16
# Инициализация весов MLP с одним скрытым слоём
weights1 = ((torch.rand(2, HIDDEN_SIZE) - 0.5) / 10).detach().requires_grad_(True)
bias1 = torch.zeros(HIDDEN_SIZE, requires_grad=True)

weights2 = ((torch.rand(HIDDEN_SIZE, 1) - 0.5) / 10).detach().requires_grad_(True)
bias2 = torch.zeros(1, requires_grad=True)

# %% [markdown]
# ### Обучение нейронной сети задачи классификации


# %%
# Определяем функцию нелинейности
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# Прямой проход
def forward(x):
    hidden = torch.mm(x, weights1) + bias1
    hidden_nonlin = sigmoid(hidden)
    output = (weights2.t() * hidden_nonlin).sum(axis=-1, keepdims=True) + bias2

    return sigmoid(output)


# Logloss
def loss(y_true, y_pred):
    return (
        -1 * (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).sum()
    )


# задаём шаг обучения
lr = 1e-3
# задаём число итераций
iters = 10000
params = [weights1, bias1, weights2, bias2]
losses = []
for i in range(iters):
    output = forward(tensor_X)
    lossval = loss(tensor_y, output)
    lossval.backward()  # тут включается в работу autograd
    for w in params:
        with torch.no_grad():
            w -= w.grad * lr  # обновляем веса
        w.grad.zero_()  # зануляем градиенты, чтобы не накапливались за итерации
    losses.append(lossval.item())
# выводим историю функции потерь по итерациям
plt.plot(losses)


# %% [markdown]
# ### Проверка результатов обучения

# %%
X_diff = X.max() - X.min()
X_left = X.min() - 0.1 * X_diff
X_right = X.max() + 0.1 * X_diff
grid = np.arange(X_left, X_right, 0.01)
grid_width = grid.size
surface = []
# создаем точки по сетке
for x1 in grid:
    for x2 in grid:
        surface.append((x1, x2))
surface = np.array(surface)
# получаем предсказания для всех точек
with torch.no_grad():
    Z = forward(torch.Tensor(surface)).detach().numpy()
# меняем форму в виде двухмерного массива
Z = Z.reshape(grid_width, grid_width)
xx = surface[:, 0].reshape(grid_width, grid_width)
yy = surface[:, 1].reshape(grid_width, grid_width)
# рисуем разделяющие поверхности классов
plt.contourf(xx, yy, Z, alpha=0.5)
# рисуем обучающую выборку
plt.scatter(X[:, 0], X[:, 1], c=output.detach().numpy() > 0.5)
# задаём границы отображения графика
plt.xlim(X_left, X_right)
plt.ylim(X_left, X_right)

# %% [markdown]
"""
## Часть 3. Классификация изображений CIFAR100
### Загрузка и распаковка набора данных CIFAR100
"""


import shutil
# %%
import urllib
from pathlib import Path

url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
filename = "cifar-100-python.tar.gz"
data_path = Path("data")

data_path.mkdir(exist_ok=True)

file_path = data_path / filename

urllib.request.urlretrieve(url, file_path)
shutil.unpack_archive(file_path, extract_dir=data_path)
file_path.unlink()  # Remove archive after extracting it.


# %% [markdown]
# ### Чтение тренировочной и тестовой выборки


# %%
def stem_extensions(filename: Path):
    extensions = "".join(filename.suffixes)

    return str(filename).removesuffix(extensions)


# %%
dataset_path = Path(stem_extensions(file_path))

with open(dataset_path / "train", "rb") as f:
    data_train = pickle.load(f, encoding="latin1")
with open(dataset_path / "test", "rb") as f:
    data_test = pickle.load(f, encoding="latin1")

# Здесь указать ваши классы по варианту!!!
CLASSES = [0, 55, 58]

train_X = data_train["data"].reshape(-1, 3, 32, 32)
train_X = np.transpose(train_X, [0, 2, 3, 1])  # NCHW -> NHWC
train_y = np.array(data_train["fine_labels"])
mask = np.isin(train_y, CLASSES)
train_X = train_X[mask].copy()
train_y = train_y[mask].copy()
train_y = np.unique(train_y, return_inverse=1)[1]
del data_train

test_X = data_test["data"].reshape(-1, 3, 32, 32)
test_X = np.transpose(test_X, [0, 2, 3, 1])
test_y = np.array(data_test["fine_labels"])
mask = np.isin(test_y, CLASSES)
test_X = test_X[mask].copy()
test_y = test_y[mask].copy()
test_y = np.unique(test_y, return_inverse=1)[1]
del data_test
Image.fromarray(train_X[50]).resize((256, 256))

# %% [markdown]
# ### Создание Pytorch DataLoader'a

# %%
batch_size = 128
dataloader = {}
for (X, y), part in zip([(train_X, train_y), (test_X, test_y)], ["train", "test"]):
    tensor_x = torch.Tensor(X)
    tensor_y = (
        F.one_hot(torch.Tensor(y).to(torch.int64), num_classes=len(CLASSES)) / 1.0
    )
    dataset = TensorDataset(tensor_x, tensor_y)  # создание объекта датасета
    dataloader[part] = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )  # создание экземпляра класса DataLoader
dataloader

# %% [markdown]
# ### Создание Pytorch модели многослойного перцептрона с одним скрытым слоем


# %%
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input):
        x = input / 255.0
        x = x - self.mean
        x = x / self.std

        return torch.flatten(x, start_dim=1)  # nhwc -> nm


class Cifar100_MLP(nn.Module):
    def __init__(self, hidden_size=32, classes=100):
        super(Cifar100_MLP, self).__init__()
        # https://blog.jovian.ai/image-classification-of-cifar100-dataset-using-pytorch-8b7145242df1
        self.norm = Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025])
        self.seq = nn.Sequential(
            nn.Linear(32 * 32 * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, classes),
        )

    def forward(self, input):
        x = self.norm(input)

        return self.seq(x)


HIDDEN_SIZE = 10
model = Cifar100_MLP(hidden_size=HIDDEN_SIZE, classes=len(CLASSES))
model

# %% [markdown]
# ### Выбор функции потерь и оптимизатора градиентного спуска

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

# %% [markdown]
# ### Обучение модели по эпохам

# %%
EPOCHS = 250
steps_per_epoch = len(dataloader["train"])
steps_per_epoch_val = len(dataloader["test"])
for epoch in range(EPOCHS):  # проход по набору данных несколько раз
    running_loss = 0.0
    model.train()
    for i, batch in enumerate(dataloader["train"], 0):
        # получение одного минибатча; batch это двуэлементный список из [inputs, labels]
        inputs, labels = batch

        # очищение прошлых градиентов с прошлой итерации
        optimizer.zero_grad()

        # прямой + обратный проходы + оптимизация
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # для подсчёта статистик
        running_loss += loss.item()
    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / steps_per_epoch:.3f}")
    running_loss = 0.0
    model.eval()
    with torch.no_grad():  # отключение автоматического дифференцирования
        for i, data in enumerate(dataloader["test"], 0):
            inputs, labels = data

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    print(
        f"[{epoch + 1}, {i + 1:5d}] val loss: {running_loss / steps_per_epoch_val:.3f}"
    )
print("Обучение закончено")

# %% [markdown]
# ### Проверка качества модели по классам на обучающей и тестовой выборках

# %%
for part in ["train", "test"]:
    y_pred = []
    y_true = []
    with torch.no_grad():  # отключение автоматического дифференцирования
        for i, data in enumerate(dataloader[part], 0):
            inputs, labels = data

            outputs = model(inputs).detach().numpy()
            y_pred.append(outputs)
            y_true.append(labels.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        print(part)
        print(
            classification_report(
                y_true.argmax(axis=-1),
                y_pred.argmax(axis=-1),
                digits=4,
                target_names=list(map(str, CLASSES)),
            )
        )
        print("-" * 50)

# %% [markdown]
# ### Визуализация весов

# %%
weights = list(model.parameters())[0].detach().numpy()
print(weights.shape)
fig, ax = plt.subplots(1, weights.shape[0], figsize=(3 * weights.shape[0], 3))
for i, ω in enumerate(weights):
    ω = ω.reshape(32, 32, 3)
    ω -= np.percentile(ω, 1, axis=[0, 1])
    ω /= np.percentile(ω, 99, axis=[0, 1])
    ω = np.clip(ω, 0, 1)
    ax[i].imshow(ω)
