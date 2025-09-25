import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from torch.utils.data import TensorDataset, DataLoader, random_split
import GAN_models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd


import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageOps
import torchvision.ops as ops
import torchvision.transforms.functional as F
import joblib
import numpy as np

device='cuda'

# Пример объединения условного вектора
def create_condition_vector(crystal, stats, embedding_layer_crystal):
    # Преобразование Crystal в числовой вектор
    crystal_encoded = embedding_layer_crystal(crystal)  # Например, [batch_size, embedding_dim]
    
    # Преобразование Stats в столбец для правильного объединения с crystal_encoded
    stats = stats.unsqueeze(1)  # Преобразует [batch_size] в [batch_size, 1]
    #stats_encoded = embedding_layer_stats(stats)
    
    # Объединение для создания полного условного вектора
    #condition = torch.cat([crystal_encoded, stats], dim=1)
    condition = torch.cat([crystal_encoded, stats], dim=1)
    return condition

# Определяем embedding_layer
num_crystal_types = 30  # Укажите количество уникальных значений для Crystal
embedding_dim = 1100      # Размерность встраивания, можно менять в зависимости от задачи
num_stats_types = 6

#embedding_layer_crystal = nn.Embedding(num_crystal_types, embedding_dim)

#embedding_layer_crystal.to(device)

# Параметры
noise_dim = 1100  # Размерность шума (z)
condition_dim = 1100 + 1  # Размерность условного вектора (embedding_dim + stats), пример: 10 для Crystal + 1 для Stats
output_shape = (1, 250, 480)  # Размерность изображения (дифракционный паттерн)

class Loader:
    def __init__(self, device='cuda'):
        self.device = device
        self.ResNet18_base_model = None
        self.num_classes = None
        self.noise_dim = 1100  # Размерность шума (z)
        self.condition_dim = 1100 + 1  # Размерность условного вектора (embedding_dim + stats), пример: 10 для Crystal + 1 для Stats
        self.output_shape = (1, 250, 480)  # Размерность изображения (дифракционный паттерн)
        self.num_crystal_types = 30     # Укажите количество уникальных значений для Crystal
        self.embedding_dim = 1100   # Размерность встраивания, можно менять в зависимости от задачи

        self.GAN_generator = None
        self.GAN_discriminator = None
        self.embedding_layer_crystal = None
        self.df_segmented = None
        self.label_encoder = None
        self.Diff_train_tensor = None
        self.Diff_test_tensor = None
        self.Stats_train_tensor = None
        self.Stats_test_tensor = None
        self.Labels_train_tensor = None
        self.Labels_test_tensor = None
        self.mean  = torch.tensor(1.4643e+08)
        self.std = torch.tensor(1.7095e+08)
        self.train_loader = None
        self.test_loader = None

    # def load_GAN(self, GAN_generator_path, GAN_discriminator_path):
    #     """Загружает предобученную модель."""
    #     #self.GAN_generator= torch.load(GAN_generator_path)
    #     #self.GAN_generator.to(self.device)
    #     self.GAN_generator= GAN_models.ConvGenerator(self.noise_dim, self.condition_dim, self.output_shape).to(device)
    #     self.GAN_generator.load_state_dict(torch.load(GAN_generator_path))
    #     self.GAN_generator.eval()  # Перевод в режим инференса

    #     #self.GAN_discriminator= torch.load(GAN_discriminator_path)
    #     #self.GAN_discriminator.to(self.device)
    #     self.GAN_discriminator= GAN_models.ConvDiscriminator(self.condition_dim, self.output_shape).to(device)
    #     self.GAN_discriminator.load_state_dict(torch.load(GAN_discriminator_path))
    #     self.GAN_discriminator.eval()  # Перевод в режим инференса

    def load_GAN(self, model_checkpoint_path):

        checkpoint = torch.load(model_checkpoint_path)
        self.GAN_generator= GAN_models.ConvGenerator(self.noise_dim, self.condition_dim, self.output_shape).to(self.device)
        self.GAN_generator.load_state_dict(checkpoint['generator_state_dict'])
        self.GAN_generator.eval()  # Перевод в режим инференса

        self.embedding_layer_crystal = nn.Embedding(self.num_crystal_types, self.embedding_dim)
        self.embedding_layer_crystal.to(self.device)
        self.embedding_layer_crystal.load_state_dict(checkpoint['embedding_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_ResNet_base(self, num_classes):

        self.num_classes = num_classes

        # Загружаем предобученную ResNet18
        self.ResNet18_base_model = models.resnet18(pretrained=True)

        # Изменяем первый сверточный слой
        self.ResNet18_base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Инициализируем веса первого слоя
        nn.init.kaiming_normal_(self.ResNet18_base_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Изменяем последний полносвязный слой для вашего количества классов
        num_ftrs = self.ResNet18_base_model.fc.in_features
        self.ResNet18_base_model.fc = nn.Linear(num_ftrs, self.num_classes)

        for name, param in self.ResNet18_base_model.named_parameters():
            if "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.ResNet18_base_model.to(self.device)

    def load_all_data(self):

        with open(r'../datasets/seg_maps_dataset.pkl', 'rb') as f:
            self.df_segmented = pickle.load(f)

    def load_no_Seg_data(self):

        # Загрузка DataFrame из файла
        with open(r'../datasets/dataset.pkl', 'rb') as f:
            self.df_segmented = pickle.load(f)

        crystal = self.df_segmented['Crystal']
        Crystal = []

        for i in crystal:
            i = i[1:]
            foo, fooo = i.split(".")
            i = foo
            #print(i)
            Crystal.append(i)

        self.df_segmented['Crystal'] = Crystal
        self.df_segmented = self.df_segmented.loc[((self.df_segmented['Crystal'] != 'Ag') & (self.df_segmented['Crystal'] != 'Au') & (self.df_segmented['Crystal'] != 'B4C'))]
        self.df_segmented = self.df_segmented.reset_index()

        # Преобразуем список матриц в 3D массив (количество матриц, высота, ширина)
        matrices = np.array(self.df_segmented['Matrix'].tolist())

        # Вычисляем среднее и стандартное отклонение для каждого пикселя по всем матрицам
        X_mean = matrices.mean(axis=0, keepdims=True)
        X_std = matrices.std(axis=0, keepdims=True)

        # Избегаем деления на ноль
        X_std[X_std == 0] = 1

        # Нормализуем все матрицы
        matrices_normalized = (matrices - X_mean) / X_std

        # Возвращаем нормализованные матрицы обратно в DataFrame
        self.df_segmented['Matrix'] = list(matrices_normalized)

        
    def load_test_data(self):

        X = self.df_segmented.drop(['Crystal', 'Stats', 'Pulce duration'], axis=1)

        # Шаг 1: Преобразование строковых меток классов в числовые метки
        self.label_encoder = LabelEncoder()

        Stats = self.df_segmented['Stats']

        y = self.df_segmented['Crystal']

        y_encoded = self.label_encoder.fit_transform(y)

        Diff_train, Diff_test, Stats_train, Stats_test, y_train, y_test = train_test_split(X, Stats, y_encoded, test_size=0.2, random_state=42)



        self.Diff_test_tensor = torch.FloatTensor(np.array(Diff_test['Matrix'].tolist()))
        self.Diff_test_tensor = self.Diff_test_tensor.unsqueeze(1)
        self.Stats_test_tensor = torch.FloatTensor(np.array(Stats_test.tolist()))
        self.Stats_test_tensor = (self.Stats_test_tensor - self.mean) / self.std
        self.Labels_test_tensor = torch.LongTensor(y_test)


    def load_train_data(self, train_size):

        test_size = 1 - train_size

        X = self.df_segmented.drop(['Crystal', 'Stats', 'Pulce duration'], axis=1)

        # Шаг 1: Преобразование строковых меток классов в числовые метки
        self.label_encoder = LabelEncoder()

        Stats = self.df_segmented['Stats']

        y = self.df_segmented['Crystal']

        y_encoded = self.label_encoder.fit_transform(y)

        Diff_train, Diff_test, Stats_train, Stats_test, y_train, y_test = train_test_split(X, Stats, y_encoded, test_size=test_size, random_state=42)



        self.Diff_train_tensor = torch.FloatTensor(np.array(Diff_train['Matrix'].tolist()))
        self.Diff_train_tensor = self.Diff_train_tensor.unsqueeze(1)
        self.Stats_train_tensor = torch.FloatTensor(np.array(Stats_train.tolist()))
        self.Stats_train_tensor = (self.Stats_train_tensor - self.mean) / self.std
        self.Labels_train_tensor = torch.LongTensor(y_train)


    def load_filtered_data(self, num_classes):

        self.num_classes = num_classes
        # Шаг 0: Обучаем LabelEncoder на полном датасете, чтобы сохранить оригинальный мэппинг
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.df_segmented['Crystal'])

        # Получаем список уникальных классов из полного датасета
        unique_classes = self.df_segmented['Crystal'].unique()
        # Выбираем случайную подвыборку классов нужного размера
        selected_classes = np.random.choice(unique_classes, num_classes, replace=False)

        # Фильтрация с созданием копии
        df_filtered = self.df_segmented[self.df_segmented['Crystal'].isin(selected_classes)].copy()

        # Применяем ранее обученный LabelEncoder для преобразования строковых меток
        # Это гарантирует, что для выбранных классов метки будут соответствовать оригинальному мэппингу
        # Преобразование меток с использованием .loc для явного указания изменения копии
        df_filtered.loc[:, 'Crystal_encoded'] = self.label_encoder.transform(df_filtered['Crystal'])

        # Разделяем на признаки и целевую переменную
        X = df_filtered.drop(['Crystal', 'Stats', 'Pulce duration'], axis=1)
        y = df_filtered['Crystal_encoded']
        Stats = df_filtered['Stats']

        # Делим данные на обучающую и тестовую выборки
        Diff_train, Diff_test, Stats_train, Stats_test, y_train, y_test = train_test_split(
            X, Stats, y, test_size=0.5, random_state=42
        )

        self.Diff_train_tensor = torch.FloatTensor(np.array(Diff_train['Matrix'].tolist()))
        self.Diff_train_tensor = self.Diff_train_tensor.unsqueeze(1)
        self.Stats_train_tensor = torch.FloatTensor(np.array(Stats_train.tolist()))
        self.Stats_train_tensor = (self.Stats_train_tensor - self.mean) / self.std
        self.Labels_train_tensor = torch.LongTensor(y_train.values)

        self.Diff_test_tensor = torch.FloatTensor(np.array(Diff_test['Matrix'].tolist()))
        self.Diff_test_tensor = self.Diff_test_tensor.unsqueeze(1)
        self.Stats_test_tensor = torch.FloatTensor(np.array(Stats_test.tolist()))
        self.Stats_test_tensor = (self.Stats_test_tensor - self.mean) / self.std
        self.Labels_test_tensor = torch.LongTensor(y_test.values)



    def do_loaders(self):

        # Создание набора данных
        train_dataset = TensorDataset(self.Diff_train_tensor, self.Stats_train_tensor, self.Labels_train_tensor)

        test_dataset = TensorDataset(self.Diff_test_tensor, self.Stats_test_tensor, self.Labels_test_tensor)

        # Создание DataLoader для каждой выборки
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        self.test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


LR = Loader()
#LR.load_GAN('../models/GAN_diffraction_generator_weights.pth', '../models/GAN_diffraction_discriminator_weights.pth')
LR.load_GAN('../models/model_checkpoint.pth')

#print(LR.GAN_generator)
#print(LR.GAN_discriminator)

#LR.load_ResNet_base()

#print(LR.ResNet18_base_model)

#LR.load_all_data()
LR.load_no_Seg_data()

#print(LR.df_segmented)

#print('--------------------------------')

LR.load_test_data()

#print(LR.Diff_test_tensor.size())
#print(LR.Stats_test_tensor.size())
#print(LR.Labels_test_tensor.size())

#print('--------------------------------')

train_sizes = np.linspace(0.05, 0.8, 16)
#train_sizes = np.arange(0.05, 0.21, 0.01)

num_classes = 35

accuracy_db = []
precision_db = []
recall_db = []
f1_db = []

for i, train_size in enumerate(train_sizes):

    LR.load_train_data(train_size)
    LR.do_loaders()
    LR.load_ResNet_base(num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(LR.ResNet18_base_model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    epochs = 7

    for epoch in range(epochs):
        LR.ResNet18_base_model.train()  # Включаем режим обучения

        for Diffractions, stats, crystals in LR.train_loader:

            # 1. # загружаем батч данных (вытянутый в линию)
            Diffractions = Diffractions.to(device)
            stats = stats.to(device)
            crystals = crystals.to(device)

            #Создание тензора с условиями генерации
            #condition = create_condition_vector(crystals, stats, LR.embedding_layer_crystal)

            # Generate fake images
            #latent = torch.randn(Diffractions.size(0), noise_dim).to(device)
            #fake_images =  LR.GAN_generator(latent, condition)

            #Diffractions = torch.cat((Diffractions, abs(fake_images)), dim=0)
            #crystals = torch.cat((crystals, crystals), dim=0)

            # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
            logits = LR.ResNet18_base_model(Diffractions)
            #logits = LR.ResNet18_base_model(abs(fake_images))

            # 3. вычислеяем - функцию потерь (loss)
            loss = criterion(logits, crystals)
        
            # 4. вычисляем градиенты
            optimizer.zero_grad()
            loss.backward()

            # 5. шаг градиентного спуска
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}] split [{i + 1}/{len(train_sizes)}]')

    # Списки для хранения предсказаний и истинных значений
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for  Diffractions, stats, crystals in LR.test_loader:

            # 1. # загружаем батч данных (вытянутый в линию)
            Diffractions = Diffractions.to(device)
            stats = stats.to(device)
            crystals = crystals.to(device)

            # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
            outputs = LR.ResNet18_base_model(Diffractions)

            _, predicted = torch.max(outputs, 1)

            # Добавляем предсказания и целевые значения в списки
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(crystals.cpu().numpy())



    # Преобразуем списки в numpy-массивы
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Вычисляем метрики на всем датасете
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    accuracy_db.append(accuracy)
    precision_db.append(precision)
    recall_db.append(recall)
    f1_db.append(f1)


"""
classes_filtered = [5, 10, 15, 20, 25, 30]

accuracy_db = []
precision_db = []
recall_db = []
f1_db = []

for i, num_classes in enumerate(classes_filtered):

    LR.load_all_data()
    LR.load_filtered_data(num_classes)
    LR.do_loaders()
    LR.load_ResNet_base()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(LR.ResNet18_base_model.parameters(), lr=1e-5, betas=(0.9, 0.99))

    epochs = 15

    for epoch in range(epochs):
        LR.ResNet18_base_model.train()  # Включаем режим обучения

        for Diffractions, stats, crystals in LR.train_loader:

            # 1. # загружаем батч данных (вытянутый в линию)
            Diffractions = Diffractions.to(device)
            stats = stats.to(device)
            crystals = crystals.to(device)

            #Создание тензора с условиями генерации
            condition = create_condition_vector(crystals, stats, embedding_layer_crystal)

            # Generate fake images
            latent = torch.randn(Diffractions.size(0), noise_dim).to(device)
            fake_images =  LR.GAN_generator(latent, condition)

            Diffractions = torch.cat((Diffractions, fake_images), dim=0)
            crystals = torch.cat((crystals, crystals), dim=0)

            # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
            logits = LR.ResNet18_base_model(Diffractions)

            # 3. вычислеяем - функцию потерь (loss)
            loss = criterion(logits, crystals)
        
            # 4. вычисляем градиенты
            optimizer.zero_grad()
            loss.backward()

            # 5. шаг градиентного спуска
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}] split [{i + 1}/{len(classes_filtered)}]')

    # Списки для хранения предсказаний и истинных значений
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for  Diffractions, stats, crystals in LR.test_loader:

            # 1. # загружаем батч данных (вытянутый в линию)
            Diffractions = Diffractions.to(device)
            stats = stats.to(device)
            crystals = crystals.to(device)

            # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
            outputs = LR.ResNet18_base_model(Diffractions)

            _, predicted = torch.max(outputs, 1)

            # Добавляем предсказания и целевые значения в списки
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(crystals.cpu().numpy())



    # Преобразуем списки в numpy-массивы
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Вычисляем метрики на всем датасете
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    accuracy_db.append(accuracy)
    precision_db.append(precision)
    recall_db.append(recall)
    f1_db.append(f1)
"""

data = pd.DataFrame(columns = ['Train split size', 'accuracy', 'precision', 'recall', 'f1'])

data['Train split size'] = train_sizes
data['accuracy'] = accuracy_db
data['precision'] = precision_db
data['recall'] = recall_db
data['f1'] = f1_db

data.to_csv('ResNet-18_accuracy_distribution_no_seg_diffractions.csv', index=False)