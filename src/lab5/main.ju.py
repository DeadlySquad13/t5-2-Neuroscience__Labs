# %% [markdown]
# # Лабораторная работа №5

# %%
# Loading extension for reloading editable packages (pip install -e .)
# %load_ext autoreload

# %% [markdown]
"""
Вариант:
1. Номер группы + 15 = 2 + 15 = 17
2. Номер варианта + 56 = 14 + 56 = 70
3. ИУ5 (Номер варианта + 21) = 14 + 21 = 35
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
## Часть 1. Построение автоэнкодера на основе изображений CIFAR100
### Загрузка и распаковка набора данных CIFAR100
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
# Подготовка модулей и переменных для загрузки CIFAR100.

# %%
url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
audio_filename = "cifar-100-python.tar.gz"

external_data_path.mkdir(exist_ok=True)

file_path = external_data_path / audio_filename

# %% [markdown]
# Загрузка CIFAR100.

# %%
# After extracting a new dir will be created so no need to create it manually.
if not os.path.isfile(file_path):
    urllib.request.urlretrieve(url, file_path)
    shutil.unpack_archive(file_path, extract_dir=external_data_path)
    file_path.unlink()  # Remove archive after extracting it.


# %% [markdown]
# ### Чтение тренировочной и тестовой выборки


# %%
def stem_extensions(filename: Path):
    """
    Stem extensions from file (including multi-part extensions like ".tar.gz").

    returns: filename without extensions and trimmed string with extensions (without leading dot).
    """
    extensions = "".join(filename.suffixes)

    return str(filename).removesuffix(extensions), extensions[1:]


# %%
dataset_path, _ = stem_extensions(file_path)
dataset_path = Path(dataset_path)

with open(dataset_path / "train", "rb") as f:
    data_train = pickle.load(f, encoding="latin1")
with open(dataset_path / "test", "rb") as f:
    data_test = pickle.load(f, encoding="latin1")

# Классы по варианту.
CLASSES = [17, 70, 35]

train_X_raw = data_train["data"].reshape(-1, 3, 32, 32)
train_X_raw = np.transpose(train_X_raw, [0, 2, 3, 1])  # NCHW -> NHWC
train_y_raw = np.array(data_train["fine_labels"])
mask = np.isin(train_y_raw, CLASSES)
train_X = train_X_raw[mask].copy()
train_y = train_y_raw[mask].copy()
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

# print(train_y_raw.tolist())


# %%
def createImage(data: ArrayLike):
    return Image.fromarray(data).resize((256, 256))


# %%
# Source: https://stackoverflow.com/a/47334314
def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    column = 0
    for i in range(len(list_of_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == no_of_columns + 1:
            fig = plt.figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, no_of_columns, column)
        plt.imshow(list_of_images[i])
        plt.axis("off")
        if len(list_of_titles) >= len(list_of_images):
            plt.title(list_of_titles[i])


# %%
# По 3 экземпляра класса из выборки.
number_of_images_per_class_to_show = 3

for class_id in CLASSES:
    print(f"{class_id = }:")
    i = number_of_images_per_class_to_show
    image_index_for_class = -1
    class_images = []
    image_indices = []

    while i > 0:
        image_index_for_class = train_y_raw.tolist().index(
            class_id, image_index_for_class + 1
        )
        image_indices.append(image_index_for_class)
        class_images.append(createImage(train_X_raw[image_index_for_class]))
        i -= 1
    grid_display(class_images, image_indices, number_of_images_per_class_to_show)
    plt.show()

# %% [markdown]
# ### Создание Cifar Dataset с аугментацией

# %%
from torch import Tensor  # noqa
from torch.utils.data import Dataset  # noqa


class CifarDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor, transform=None, p=0.0):
        assert X.size(0) == y.size(0)
        super(Dataset, self).__init__()
        self.X = X
        self.y = y
        self.transform = transform
        self.prob = p

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, index):
        X = self.X[index]
        if self.transform and np.random.random() < self.prob:
            X = self.transform(X.permute(2, 0, 1) / 255).permute(1, 2, 0) * 255

        y = self.y[index]

        return X, y


# %% [markdown]
"""
Попробуем применить аугментацию к одной из картинок.
"""

# %%
import torchvision.transforms as T  # noqa

transform = T.Compose(
    [
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.0),
        # shear - сдвиг.
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
    ]
)

Image.fromarray(
    (CifarDataset(Tensor(train_X), Tensor(train_y), transform=transform, p=1)[10])[0]
    .numpy()
    .astype(np.uint8)
).resize((256, 256))


# %% [markdown]
# ### Создание Pytorch DataLoader'a


# %%
def create_dataloader(batch_size=128):
    dataloader: dict[str, DataLoader] = {}
    for (X, y), part in zip([(train_X, train_y), (test_X, test_y)], ["train", "test"]):
        tensor_x = torch.Tensor(X)
        tensor_y = (
            F.one_hot(torch.Tensor(y).to(torch.int64), num_classes=len(CLASSES)) / 1.0
        )
        dataset = CifarDataset(
            tensor_x, tensor_y, transform=transform if part == "train" else None, p=0.5
        )  # создание объекта датасета
        dataloader[part] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            prefetch_factor=8 if part == "train" else 2,
            num_workers=2,
            persistent_workers=True,
        )  # создание экземпляра класса DataLoader

    return dataloader


# %% [markdown]
# ### Создание Pytorch модели-автоэнкодера


# %%
HIDDEN_SIZE = 512


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)

    def forward(self, input):
        x = input / 255.0
        x = x - self.mean
        x = x / self.std
        # NEW: instead of permute now flattenning. Othewise shape will be
        # different.
        return torch.flatten(x, start_dim=1)  # nhw -> nm


class Cifar100_AE(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, classes=len(CLASSES)):
        super(Cifar100_AE, self).__init__()
        # https://blog.jovian.ai/image-classification-of-cifar100-dataset-using-pytorch-8b7145242df1
        self.norm = Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025])
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, hidden_size // 8),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 8, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 32 * 32 * 3),
        )

    def forward(self, input):
        normed = self.norm(input)
        encoded = self.encoder(normed)
        out = self.decoder(encoded)
        return out, encoded, normed


model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
summary(model, input_size=(32, 32, 3))
model

# %% [markdown]
# ### Создание логов для Tensorboard

# %%
import time  # noqa

from torch.utils.tensorboard import SummaryWriter  # noqa

# %load_ext tensorboard

current_time = str(int(time.time()))

run_path = Path("runs/tensorboard")

train_run_path = run_path / "train" / current_time
test_run_path = run_path / "test" / current_time

train_summary_writer = SummaryWriter(log_dir=train_run_path)
test_summary_writer = SummaryWriter(log_dir=test_run_path)

# %% [markdown]
# ## Обучение


# %%
REDRAW_EVERY = 20
# New: Changed epochs.
EPOCHS = 200

from sklearn.metrics import r2_score


def train(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    dataloader: dict[str, DataLoader],
    scheduler: optim.lr_scheduler.LRScheduler = None,
    epochs=EPOCHS,
):
    steps_per_epoch = len(dataloader["train"])
    steps_per_epoch_val = len(dataloader["test"])

    pbar = tqdm(total=epochs * steps_per_epoch)
    losses = []
    losses_val = []
    passed = 0
    # Для создания чекпоинта
    best_acc = 0
    checkpoint_path = Path("cifar_autoencoder.pth")

    for epoch in range(epochs):  # проход по набору данных несколько раз
        tmp = []
        model.train()
        for i, batch in enumerate(dataloader["train"], 0):
            # получение одного минибатча; batch это двуэлементный список из [inputs, labels]
            inputs, _ = batch
            # NEW:
            inputs, _ = inputs.to(device), _.to(device)

            # очищение прошлых градиентов с прошлой итерации
            optimizer.zero_grad()

            # прямой + обратный проходы + оптимизация
            outputs = model(inputs)
            # NEW: outputs[0] (out) and outputs[2] (norm) instead of outputs and label.
            loss = criterion(outputs[0], outputs[2])
            loss.backward()
            optimizer.step()

            # для подсчёта статистик
            # NEW: r2_score instead of mean of max.
            accuracy = (
                r2_score(
                    outputs[2].detach().cpu().numpy(), outputs[0].detach().cpu().numpy()
                )
                * 100
            )
            tmp.append((loss.item(), accuracy.item()))
            pbar.update(1)

            with train_summary_writer as writer:
                writer.add_scalar("loss", tmp[-1][0], global_step=pbar.n)
                writer.add_scalar("accuracy", tmp[-1][1], global_step=pbar.n)

        losses.append(
            (
                np.mean(tmp, axis=0),
                np.percentile(tmp, 25, axis=0),
                np.percentile(tmp, 75, axis=0),
            )
        )

        if scheduler:
            scheduler.step()  # Обновляем learning_rate каждую эпоху.

        tmp = []
        model.eval()
        with torch.no_grad():  # отключение автоматического дифференцирования
            for i, data in enumerate(dataloader["test"], 0):
                inputs, _ = data
                # на GPU
                inputs, _ = inputs.to(device), _.to(device)

                outputs = model(inputs)
                # NEW: Taking certain indices of outputs instead of whole outputs and label tensors.
                loss = criterion(outputs[0], outputs[2])
                # NEW: r2_score instead of mean of max.
                accuracy = (
                    r2_score(
                        outputs[2].detach().cpu().numpy(),
                        outputs[0].detach().cpu().numpy(),
                    )
                    * 100
                )
                tmp.append((loss.item(), accuracy.item()))
        losses_val.append(
            (
                np.mean(tmp, axis=0),
                np.percentile(tmp, 25, axis=0),
                np.percentile(tmp, 75, axis=0),
            )
        )
        with test_summary_writer as writer:
            writer.add_scalar("loss", losses_val[-1][0][0], global_step=pbar.n)
            writer.add_scalar("accuracy", losses_val[-1][0][1], global_step=pbar.n)

        acc = losses_val[-1][0][1]
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)

        # Обновление графиков.
        if (epoch + 1) % REDRAW_EVERY != 0:
            continue
        clear_output(wait=False)
        print(
            "Эпоха: %s\n"
            "Лучшее значение метрики: %s\n"
            "Текущее значение метрики: %s" % (epoch + 1, best_acc, acc)
        )
        passed += pbar.format_dict["elapsed"]
        pbar = tqdm(total=EPOCHS * steps_per_epoch, miniters=5)
        pbar.update((epoch + 1) * steps_per_epoch)
        x_vals = np.arange(epoch + 1)
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        stats = np.array(losses)
        stats_val = np.array(losses_val)
        ax[1].set_ylim(stats_val[:, 0, 1].min() - 5, 100)
        ax[1].grid(axis="y")
        for i, title in enumerate(["MSE", "Accuracy"]):
            ax[i].plot(x_vals, stats[:, 0, i], label="train")
            ax[i].fill_between(x_vals, stats[:, 1, i], stats[:, 2, i], alpha=0.4)
            ax[i].plot(x_vals, stats_val[:, 0, i], label="val")
            ax[i].fill_between(
                x_vals, stats_val[:, 1, i], stats_val[:, 2, i], alpha=0.4
            )
            ax[i].legend()
            ax[i].set_title(title)
        plt.show()

    model.load_state_dict(torch.load(checkpoint_path))
    print("Обучение закончено за %s секунд" % passed)

    return dataloader


# %%
# Запуск tensorboard в Jupyter Notebook.
# %tensorboard --logdir runs/tensorboard

# %% [markdown]
# ### Выбор функции потерь и оптимизатора градиентного спуска


# %%
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# New weight_decay ruins metrics a lot.
WEIGHT_DECAY = 0


def train_autoencoder(
    model: nn.Module,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    # Регуляризация модели за счёт коэффициента, учитывающего сложность модели.
    # Норма параметров будет прибавлена к функции потерь. Чем больше
    # weight_decay, тем сильнее штраф за сложность.
    weight_decay=1e-5,
    # Постепенное уменьшение шага обучения каждые N эпох.
    scheduler_step_size=240,
):
    # NEW: MSELoss instead of CrossEntropyLoss because we are not doing
    # classification now. Also it doesn't have label smoothing.
    criterion = nn.MSELoss()
    # NEW: Adam instead of SGD. It doesn't have momentum param.
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    dataloader = create_dataloader(batch_size=batch_size)
    # Добавляем постепенное уменьшение шага обучения каждые step_size
    #   эпох.
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=scheduler_step_size, gamma=0.5
    )

    return train(
        model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=dataloader,
        epochs=epochs,
        # scheduler=scheduler,
    )


dataloader = train_autoencoder(model, weight_decay=WEIGHT_DECAY)

# %% [markdown]
# ### Проверка качества модели по классам на обучающей и тестовой выборках


# %%
def get_encoder_results(model: Cifar100_AE, dataloader: DataLoader):
    embeddings = []
    images = []
    reconstructs = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader["test"], 0):
            # получение одного минибатча; batch это двуэлементный список из [inputs, labels]
            inputs, labels = batch
            # на GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # очищение прошлых градиентов с прошлой итерации
            # optimizer.zero_grad()

            # прямой + обратный проходы + оптимизация
            out, embedding, norm = model(inputs)
            embeddings.append(embedding.detach().cpu().numpy())
            images.append(inputs.detach().cpu().numpy())
            reconstructs.append(out.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    images = np.concatenate(images, axis=0)
    reconstructs = np.concatenate(reconstructs, axis=0)
    reconstructs = (reconstructs - reconstructs.min()) / (
        reconstructs.max() - reconstructs.min()
    )

    return {
        "embeddings": embeddings,
        "images": images,
        "reconstructs": reconstructs,
    }


# %%
results = get_encoder_results(model, dataloader)

# %% [markdown]
# #### Визуализация эмбеддингов

# %%
# Тут мы применяем уменьшение размерности к многомерному
# эмбеддингу, полученному из бутылочного горлышка автоэнкодера.
# Можете попробовать сделать число нейронов равное 2
# и тогда данный этап можно пропустить и написать просто:
# projection = embeddings

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA


def visualize_images_embeddings(images: NDArray, embeddings: NDArray):
    projections = PCA(n_components=2).fit_transform(embeddings)

    def implot(x, y, image, ax, zoom=1):
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
        ax.add_artist(ab)
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    for img, x in zip(images, projections):
        img = img.reshape(32, 32, 3) / 255.0
        implot(x[0], x[1], img, ax=ax, zoom=0.5)


# %% [markdown]
# #### Визуализация восстановленной картинки

# %%
from ipywidgets import interact


def report_encoder_results(images: NDArray, reconstructs: NDArray, embeddings: NDArray):
    def draw_comparision(index: int):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(images[index].reshape(32, 32, 3) / 255.0)
        ax[1].imshow(reconstructs[index].reshape(32, 32, 3))
        ax[0].set_title("Оригинал")
        ax[1].set_title("Восстановленное изображение")

    interact(draw_comparision, index=(0, len(images) - 1))

    visualize_images_embeddings(images, embeddings)


# %% [markdown]
# Как видно, получившиеся embedding'и образуют некоторые кластеры.

# %%
report_encoder_results(**results)

# %% [markdown]
# ### Подберём гиперпараметры
# #### batch size и epochs
# Меняем их параллельно (одно уменьшили в два раза, следовательно другое
# увеличили в два раза)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(model, weight_decay=WEIGHT_DECAY)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model, weight_decay=WEIGHT_DECAY, batch_size=int(BATCH_SIZE / 2), epochs=EPOCHS * 2
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model, weight_decay=WEIGHT_DECAY, batch_size=int(BATCH_SIZE / 4), epochs=EPOCHS * 4
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE * 4, epochs=int(EPOCHS / 4)
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE * 2, epochs=int(EPOCHS / 2)
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %% [markdown]
# Как видно, при количестве эпох меньше 100 наблюдается недообучение. При
# количестве эпох больше 200 - переобучение. Лучший результат ~83%.
# Зафиксируем среднее - 200 эпох и соответствующий этому batch size:

# %%
OPTIMAL_EPOCHS = 200
OPTIMAL_BATCH_SIZE = int(BATCH_SIZE * (OPTIMAL_EPOCHS / EPOCHS))
OPTIMAL_BATCH_SIZE


# %% [markdown]
# #### learning rate
# learning rate также стоит менять вместе с epochs либо batch_size
# параллельно для сохранения количества итерация константным.
# Будем при увеличении learning_rate в n раз уменьшать epochs в n раз.

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE * 2,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=int(OPTIMAL_EPOCHS / 2),
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE * 4,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=int(OPTIMAL_EPOCHS / 4),
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=int(LEARNING_RATE / 4),
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS * 4,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=int(LEARNING_RATE / 2),
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS * 2,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %% [markdown]
# В итоге изменение learning_rate не принесло плодов: начальный был оптимальным.

# %%
OPTIMAL_LEARNING_RATE = LEARNING_RATE

# %% [markdown]
# #### hidden size

# %%
model = Cifar100_AE(hidden_size=HIDDEN_SIZE)
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=int(HIDDEN_SIZE * 2))
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=int(HIDDEN_SIZE * 4))
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=int(HIDDEN_SIZE * 4))
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY + 1e-5,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=int(HIDDEN_SIZE * 4))
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY + 2e-5,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=int(HIDDEN_SIZE / 2))
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=int(HIDDEN_SIZE / 4))
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %%
model = Cifar100_AE(hidden_size=int(HIDDEN_SIZE / 8))
model.to(device)
dataloader = train_autoencoder(
    model,
    weight_decay=WEIGHT_DECAY,
    learning_rate=OPTIMAL_LEARNING_RATE,
    batch_size=OPTIMAL_BATCH_SIZE,
    epochs=OPTIMAL_EPOCHS,
)
results = get_encoder_results(model, dataloader)
report_encoder_results(**results)

# %% [markdown]
# Вполне закономерно, увеличение hidden_size ведёт к улучшение модели.
# В итоге наилучшим HIDDEN_SIZE оказалось значение, увеличенное вдвое - 86,5%. При
# дальнейшем увеличении параметра, начинается переобучение. Поставив
# weight_decay = 1e-5, мы уменьшили переобучение. При weight_decay = 2e-5 мы
# полностью устранили переобучение, достигнув 82%. Этот результат хуже
# предыдущих вариантов, при этом модель обучается дольше в 1.5 раза.


# %% [markdown]
"""
### Анализ результатов обучения модели автоэнкодера
Автоэнкодер позволяет построить эмбеддинги, по которым, например, можно в будущем реализовывать кластеризацию.

По результатам наших экспериментов с гиперпараметрами видно, что автоэнкодер по сравнению с нейронными
сетями из предыдущих лабораторных работ начинает переобучаться довольно поздно. Из-за этого, например, мы почти не пользовались
weight_decay и совершенно отбросили scheduler.
"""

# %% [markdown]
# ## Часть 2. Очистка звука

# %% [markdown]
# Подготовка модулей и переменных для работы с аудио файлами.

# %%
import subprocess
from pathlib import Path

external_audio_data_path = external_data_path / "audio"
external_audio_data_path.mkdir(exist_ok=True)

external_noise_data_path = external_data_path / "noises"
external_noise_data_path.mkdir(exist_ok=True)

# %% [markdown]
# Подготовка модулей и переменных для работы с музыкой, с которой мы собираемся
# экспериментировать.

# %%
# Поменять на музыку, с которой хотите экспериментировать.
audio_name = "Arcanum"
audio_filename = f"{audio_name}.mp3"
audio_path = external_audio_data_path / audio_filename

start_time = "00:00:00"
finish_time = "00:00:15"

str(audio_path), str(stem_extensions(audio_path))

# %% [markdown]
# Аналогично с файлом шума.

# %%
# Поменять на файл с шумом, с которым хотите экспериментировать.
noise_name = "bee"
noise_filename = f"{noise_name}.mp3"
noise_path = external_noise_data_path / noise_filename

# %% [markdown]
# ### Подготовка файла музыки
# Создадим функцию, которая с помощью ffmpeg поможет нам обрезать звуковой
# файл.


# %%
def change_filename(file_path: Path, new_filename: str, preserve_extension=True):
    """
    Changes filename.
    params:
        preserve_extension: if True, only name will be changed: "test.mp3" -> "new.mp3".
            Otherwise, extension will be changed too: "test.mp3" -> "new"
            (you can change name and extension at the same time this way).
    """
    parent_path = file_path.parent

    if not preserve_extension:
        return parent_path / new_filename

    file_path_without_extensions, trimmed_extensions = stem_extensions(file_path)
    file_path_without_extensions = Path(file_path_without_extensions)

    return parent_path / f"{file_path_without_extensions.name}.{trimmed_extensions}"


def append_to_filename(file_path: Path, str_to_append: str):
    """
    Appends str_to_append to the file_path just before extension, for example:
        "test/audio.mp3" with str_to_append = "_processed" will become
        "test/audio_processed.mp3"
    """
    file_path_without_extensions, trimmed_extensions = stem_extensions(file_path)
    file_path_without_extensions = Path(file_path_without_extensions)

    return change_filename(
        file_path=file_path,
        new_filename=f"{file_path_without_extensions.name}{str_to_append}.{trimmed_extensions}",
        preserve_extension=False,
    )


# %%
def ffmpeg(
    audio_path: Path,
    options: list[str],
    output_path: Path = None,
    output_filename: str = None,
    output_file_path: Path = None,
):
    """
    returns (Path): file path of the resulting file
    """
    if not audio_path.exists():
        raise Exception(f"No file {audio_path} exists!")

    if output_file_path and output_path:
        raise Exception(
            f"If you've passed output_file_path, output_path will be ignored. Please, remove it from parameters"
        )

    if output_file_path and output_filename:
        raise Exception(
            f"If you've passed output_file_path, output_filename will be ignored. Please, remove it from parameters"
        )

    output_path = output_path or interim_data_path

    output_filename = output_filename or audio_path.name

    output_file_path = output_file_path or (output_path / output_filename)

    subprocess.run(
        [
            "ffmpeg",
            "-y",  # Allow overwriting file if it exists.
            "-i",
            str(audio_path),
            *options,
            output_file_path,
        ]
    )

    return output_file_path


# %%
def cut_audio(
    audio_path: Path,
    start_time: str = "00:00:00",
    finish_time: str = "00:00:15",
    output_path: Path = None,
    output_filename: str = None,
    output_file_path: Path = None,
) -> Path:
    """
    returns (Path): file path of the resulting file
    """
    output_filename = (
        output_filename
        or append_to_filename(file_path=audio_path, str_to_append="--cut").name
    )

    return ffmpeg(
        audio_path=audio_path,
        output_path=output_path,
        output_filename=output_filename,
        output_file_path=output_file_path,
        options=[
            "-ss",
            start_time,
            "-to",
            finish_time,
            "-c",
            "copy",
        ],
    )


# %% [markdown]
# Возьмём первые 15 секунд, всё остальное обрежем.

# %%
try:
    cut_audio_file_path = cut_audio(
        audio_path=audio_path, start_time=start_time, finish_time=finish_time
    )
except Exception as error:
    print(f"Caught this error while trying to cut_audio for audio file: {repr(error)}")

# %% [markdown]
# Аналогично с шумом.

# %%
try:
    cut_noise_file_path = cut_audio(
        audio_path=noise_path, start_time=start_time, finish_time=finish_time
    )
except Exception as error:
    print(f"Caught this error while trying to cut_audio for noise file: {repr(error)}")

# %% [markdown]
# Создадим функцию, которая с помощью ffmpeg конвертирует файл в формат,
# совместимый с автоэнкодером.


# %%
def convert_audio_for_autoencoder(
    audio_path: Path,
    output_path: Path = None,
    output_filename_without_extension: str = None,
    output_file_path: Path = None,
) -> Path:
    """
    returns (Path): file path of the resulting file
    """

    audio_path_without_extensions = Path(stem_extensions(audio_path)[0])
    output_filename = f"{output_filename_without_extension}.wav" or change_filename(
        file_path=audio_path,
        new_filename=f"{audio_path_without_extensions.name}.wav",
        preserve_extension=False
    )

    return ffmpeg(
        audio_path=audio_path,
        output_path=output_path or processed_data_path,
        output_filename=output_filename,
        output_file_path=output_file_path,
        options=[
            "-ac",
            "1",  # Audio channels.
            "-ar",
            "16000",  # Audio frame*r*ate.
        ],
    )


# %% [markdown]
# Конвертируем в wav файл с одним каналом и частотой дискретизации 16кГц

# %%
try:
    audio_for_autoencoder_path = convert_audio_for_autoencoder(audio_path=cut_audio_file_path, output_filename_without_extension=audio_name)
except Exception as error:
    print(f"Caught this error while trying to convert_audio_for_autoencoder for audio: {repr(error)}")

# %% [markdown]
# Аналогично для файла шума.

# %%
try:
    noise_for_autoencoder_path = convert_audio_for_autoencoder(audio_path=cut_noise_file_path, output_filename_without_extension=noise_name)
except Exception as error:
    print(f"Caught this error while trying to convert_audio_for_autoencoder for noise: {repr(error)}")

# %% [markdown]
# Добавим шум в оригинал.

# %%
from scipy.io import wavfile

# Считывание аудио файла.
fs, data = wavfile.read(audio_for_autoencoder_path)
data = data / (2**16-1)
data[0] = 1
# Добавление шума.
_, noise = wavfile.read(noise_for_autoencoder_path)
noise = noise / (2**16-1) 
noise = np.tile(noise, 1+data.size//noise.size)[:data.size].copy()
noise = (0.1*data.max()/noise.max())*noise
data_noised = data+noise

# %% [markdown]


# %%
start_sec = 0
finish_sec = 15

from IPython.display import Audio, display

print('Original')
display(Audio(data[fs*start_sec:fs*finish_sec], rate=fs))
print('Noised')
display(Audio(data_noised[fs*start_sec:fs*finish_sec], rate=fs))

# %%
from scipy import signal

_, _, Zxx = signal.stft(data_noised, fs=fs, nperseg=512)
_, xrec = signal.istft(Zxx, fs)
print('Reconstructed Noised')
display(Audio(xrec[fs*start_sec:fs*finish_sec], rate=fs))
_, _, Zxx0 = signal.stft(noise, fs=fs, nperseg=512)
_, xrec = signal.istft(Zxx0, fs)
print('Noise')
display(Audio(xrec[fs*start_sec:fs*finish_sec], rate=fs))

# %%
# https://github.com/digantamisra98/Mish
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        return x*self.tanh(self.softplus(x))

class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 256, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1),
            Mish(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.ConvTranspose1d(256, 2, kernel_size=3, stride=2, padding=1),
        )
            

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

net = DenoisingAE()
net.to(device)
print(Zxx.shape[0])
net(torch.rand(1, 2, Zxx.shape[0], device=device)).detach().cpu().numpy().shape

# %%
criterion = nn.MSELoss() #nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-4) #, weight_decay=1e-6)

# %% [markdown]
# ### Создадим DataLoader'ы, разделив набор данных на обучающую и тестовую выборки.

# %%
batch_size = 128

X = np.concatenate([np.real(Zxx).T[:,:,None],
                    np.imag(Zxx).T[:,:,None]], axis=-1)
y = np.concatenate([np.real(Zxx0).T[:,:,None],
                    np.imag(Zxx0).T[:,:,None]], axis=-1)

normalization = X.reshape(-1, 2).std()
X /= normalization
y /= normalization
tensor_x = torch.Tensor(np.transpose(X, [0, 2, 1])) # channels first
tensor_y = torch.Tensor(np.transpose(y, [0, 2, 1])) # channels first

train_dataset = TensorDataset(tensor_x[:tensor_x.shape[0]*9//10],
                              tensor_y[:tensor_x.shape[0]*9//10]) # create your datset
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True) # create your dataloader
test_dataset = TensorDataset(tensor_x[tensor_x.shape[0]*9//10:],
                              tensor_y[tensor_x.shape[0]*9//10:]) # create your datset
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size, shuffle=True) # create your dataloader

# %%
def eval_dataset(dataloader):
    net.eval()
    target = []
    pred = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        preds = net(inputs).detach().cpu().numpy()
        target.append(inputs.detach().cpu().numpy().reshape(inputs.shape[0], -1))
        pred.append(preds.reshape(inputs.shape[0], -1))
    net.train()
    target = np.concatenate(target)
    pred = np.concatenate(pred)
    return r2_score(target, pred)

eval_dataset(train_dataloader)

# %%
scores = []
scores_val = []
net.train()
for epoch in tqdm(range(15)):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, batch in tqdm(enumerate(train_dataloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(train_dataloader):.5f}')
    scores.append(eval_dataset(train_dataloader))
    scores_val.append(eval_dataset(test_dataloader))
    print('R2 score on train:', scores[-1])
    print('R2 score on val:', scores_val[-1])
print('Finished Training')
plt.plot(scores, label='training')
plt.plot(scores_val, label='validation')
plt.legend()

# %%
net.eval()
pred = []
batch_size = 128
for i in tqdm(range(tensor_x.shape[0]//batch_size+1)):
    preds = net(tensor_x[i*batch_size:(i+1)*batch_size].to(device))
    pred.append(np.transpose(preds.detach().cpu().numpy(),
                             [0, 2, 1])*normalization)
pred = np.concatenate(pred, axis=0) 
_, xrec = signal.istft((pred[:,:,0]+pred[:,:,1]*1j).T, fs)
xrec = data_noised - xrec[:data.size]
print(r2_score(data, data_noised))
print(r2_score(data, xrec[:data.size]))
print('Original')
display(Audio(data[fs*start_sec:fs*finish_sec], rate=fs))
print('Noised')
display(Audio(data_noised[fs*start_sec:fs*finish_sec], rate=fs))
print('Denoised')
display(Audio(xrec[fs*start_sec:fs*finish_sec], rate=fs))


# %% [markdown]
# ## Экспорт модели

# %%
models_path = Path("models")
model_filename = "cifar100_resnet.pt"

models_path.mkdir(exist_ok=True)

model_file_path = models_path / model_filename

torch.save(model, model_file_path)
# загрузка
new_model_2 = torch.load(model_file_path)
new_model_2.eval()

# %%
# входной тензор для модели
onnx_model_filename = "cifar100_resnet.onnx"
x = torch.randn(1, 32, 32, 3, requires_grad=True).to(device)
torch_out = model(x)

# экспорт модели
torch.onnx.export(
    model,  # модель
    x,  # входной тензор (или кортеж нескольких тензоров)
    models_path
    / onnx_model_filename,  # куда сохранить (либо путь к файлу либо fileObject)
    export_params=True,  # сохраняет веса обученных параметров внутри файла модели
    opset_version=9,  # версия ONNX
    do_constant_folding=True,  # следует ли выполнять укорачивание констант для оптимизации
    input_names=["input"],  # имя входного слоя
    output_names=["output"],  # имя выходного слоя
    dynamic_axes={
        "input": {
            0: "batch_size"
        },  # динамичные оси, в данном случае только размер пакета
        "output": {0: "batch_size"},
    },
)
