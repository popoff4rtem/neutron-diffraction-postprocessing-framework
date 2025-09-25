import torch
import pandas as pd
import numpy as np
import pickle
import GAN_archs
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

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


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def psnr(img1, img2, data_range=1.0):
    """
    Вычисляет PSNR между двумя изображениями.
    
    img1, img2: тензоры размера [N, C, H, W] с диапазоном значений [0, data_range].
    data_range: максимальное возможное значение пикселя (например, 1.0, если изображение нормировано).
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse)


def gaussian(window_size, sigma):
    """
    Генерирует одномерное гауссово окно.
    """
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def create_window(window_size, channel, sigma):
    """
    Создаёт 2D гауссово окно для сверточного вычисления SSIM.
    """
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()  # матричное произведение для получения 2D окна
    _2D_window = _2D_window.float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size)
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0, size_average=True):
    """
    Вычисляет SSIM между двумя изображениями.
    
    img1, img2: тензоры размера [N, C, H, W] с диапазоном значений [0, data_range].
    window_size: размер гауссова окна (обычно 11).
    sigma: стандартное отклонение для гауссова окна (обычно 1.5).
    size_average: если True, возвращается среднее значение SSIM по всему изображению; 
                  иначе — карта SSIM.
    """
    # Определяем число каналов (для одноканальных изображений channel = 1)
    channel = img1.size(1)
    window = create_window(window_size, channel, sigma).to(img1.device)
    
    # Вычисляем локальные средние
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Вычисляем локальные дисперсии и ковариацию
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Константы для числителя и знаменателя (как в оригинальной статье)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

class GAN:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = None
        self.discriminator = None
        self.embedding_layer_crystal = None
        self.noise_dim = 1100  # Размерность шума (z)
        self.embedding_dim = 1100
        self.condition_dim = self.embedding_dim + 1  # Размерность условного вектора (embedding_dim + stats), пример: 10 для Crystal + 1 для Stats
        self.output_shape = (1, 250, 480)  # Размерность изображения (дифракционный паттерн)
        self.num_crystal_types = 30  # Укажите количество уникальных значений для Crystal
        self.criterion = torch.nn.BCELoss()
        self.lr = 0.0001  # Скорость обучения
        self.beta1 = 0.5  # Параметры для Adam, beta1 = 0.5 часто используется в GAN
        self.num_epochs = 40
        

    def init_embedding_layer(self):
        checkpoint = torch.load('../models/model_checkpoint.pth')
        self.embedding_layer_crystal = nn.Embedding(self.num_crystal_types, self.embedding_dim)
        self.embedding_layer_crystal.to(self.device)
        self.embedding_layer_crystal.load_state_dict(checkpoint['embedding_state_dict'])


    def init_generator(self):
        self.generator = GAN_archs.ConvGenerator(self.noise_dim, self.condition_dim, self.output_shape)
        self.generator.apply(weights_init)
        self.generator.to(self.device)


    def init_discriminator(self):
        self.discriminator = GAN_archs.ConvDiscriminator(self.condition_dim, self.output_shape)
        self.discriminator.apply(weights_init)
        self.discriminator.to(self.device)

    def train(self, train_loader):
        # Create optimizers
        optimizer = {
            "discriminator": torch.optim.Adam(self.discriminator.parameters(), 
                                                lr=self.lr, betas=(0.5, 0.999)),
            "generator": torch.optim.Adam(self.generator.parameters(),
                                            lr=self.lr, betas=(0.5, 0.999))
        }
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        for epoch in range(self.num_epochs):
            loss_d_per_epoch = []
            loss_g_per_epoch = []
            real_score_per_epoch = []
            fake_score_per_epoch = []
            for real_images, stats, crystals in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
                real_images = real_images.to(self.device)
                stats = stats.to(self.device)
                crystals = crystals.to(self.device)

                #Создание тензора с условиями генерации
                condition = create_condition_vector(crystals, stats, self.embedding_layer_crystal)

                # Обучение дискриминатора
                # Оччистка градиентов для дискриминатора
                optimizer["discriminator"].zero_grad()

                # Pass real images through discriminator
                real_preds = self.discriminator(real_images, condition)
                
                real_targets = torch.ones(real_images.size(0), 1, device=self.device)
                real_loss = self.criterion(real_preds, real_targets)
                cur_real_score = torch.mean(real_preds).item()
                
                # Generate fake images
                latent = torch.randn(real_images.size(0), self.noise_dim).to(self.device)
                fake_images =  self.generator(latent, condition)


                # Pass fake images through discriminator
                fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
                fake_preds = self.discriminator(fake_images, condition)

                fake_loss = self.criterion(fake_preds, fake_targets)
                cur_fake_score = torch.mean(fake_preds).item()

                real_score_per_epoch.append(cur_real_score)
                fake_score_per_epoch.append(cur_fake_score)

                # Update discriminator weights
                loss_d = real_loss + fake_loss
                loss_d.backward(retain_graph=True)
                #loss_d.backward()
                optimizer["discriminator"].step()
                loss_d_per_epoch.append(loss_d.item())


                # Train generator
                # Clear generator gradients
                optimizer["generator"].zero_grad()
                
                # Generate fake images
                latent = torch.randn(real_images.size(0), self.noise_dim).to(self.device)
                fake_images = self.generator(latent, condition)

                # Try to fool the discriminator
                preds = self.discriminator(fake_images, condition)
                targets = torch.ones(real_images.size(0), 1, device=self.device)
                loss_g = self.criterion(preds, targets)
                
                # Update generator weights
                loss_g.backward()
                optimizer["generator"].step()
                loss_g_per_epoch.append(loss_g.item())

            # Record losses & scores
            losses_g.append(np.mean(loss_g_per_epoch))
            losses_d.append(np.mean(loss_d_per_epoch))
            real_scores.append(np.mean(real_score_per_epoch))
            fake_scores.append(np.mean(fake_score_per_epoch))

            print(f'Epoch [{epoch}/{self.num_epochs}], d_loss: {losses_d[-1]}, g_loss: {losses_g[-1]},  real_score: {real_scores[-1]}, fake_score: {fake_scores[-1]}')


class Validation:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ResNet = None

    def init_ResNet(self):
        self.ResNet = torch.load('../models/ResNet-18_diffraction.pth', weights_only=False)
        self.ResNet.eval()
        self.ResNet.to(self.device)

    def get_metrics(self, train_loader, gan_instance):
        # Списки для хранения предсказаний и истинных значений
        all_preds = []
        all_targets = []
        PSNR_array = []
        SSIM_array = []

        with torch.no_grad():
            for real_images, stats, crystals in train_loader:
                # 1. # загружаем батч данных (вытянутый в линию)
                stats = stats.to(self.device)
                crystals = crystals.to(self.device)
                real_images = real_images.to(self.device)

                #Создание тензора с условиями генерации
                condition = create_condition_vector(crystals, stats, gan_instance.embedding_layer_crystal)

                # Generate fake images
                latent = torch.randn(real_images.size(0), gan_instance.noise_dim).to(self.device)
                fake_images =  gan_instance.generator(latent, condition)

                # Вычисляем PSNR
                psnr_value = psnr(real_images, abs(fake_images), data_range=1.0)
                #print("PSNR:", psnr_value.item())
                PSNR_array.append(psnr_value.item())

                # Вычисляем SSIM
                ssim_value = ssim(real_images, abs(fake_images), window_size=11, sigma=1.5, data_range=1.0, size_average=True)
                #print("SSIM:", ssim_value.item())
                SSIM_array.append(ssim_value.item())

                # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
                outputs = self.ResNet(abs(fake_images))

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
        PSNR_mean = sum(PSNR_array) / len(PSNR_array)
        SSIM_mean = sum(SSIM_array) / len(SSIM_array)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision (Weighted): {precision:.2f}")
        print(f"Recall (Weighted): {recall:.2f}")
        print(f"F1-Score (Weighted): {f1:.2f}")
        print(f'PSNR_mean {PSNR_mean}')
        print(f'SSIM_mean {SSIM_mean}')

        return accuracy, precision, recall, f1, PSNR_mean, SSIM_mean
    

# === 1. Загрузка данных
with open(r'./datasets/seg_maps_dataset.pkl', 'rb') as f:
    df = pickle.load(f)

results_df = pd.DataFrame(columns=[
    'Fold', 
    'Accuracy', 
    'Precision', 
    'Recall', 
    'F1', 
    'PSNR_mean', 
    'SSIM_mean'
])

batch_size = 5

# === 2. Подготовка признаков и меток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Crystal'])

all_matrices = np.array(df['Matrix'].tolist())  # shape: (N, H, W)
all_stats = np.array(df['Stats'].tolist())      # shape: (N, )

# Нормализация stats (глобально по всему датасету)
stats_mean = np.mean(all_stats)
stats_std = np.std(all_stats)
all_stats_norm = (all_stats - stats_mean) / stats_std

# === 3. K-Fold разбиение
k_folds = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

Gnn = GAN()
Vall = Validation()
Gnn.init_embedding_layer()

fold = 0
for train_idx, val_idx in skf.split(all_matrices, y_encoded):
    print(f"\nFold {fold + 1}/{k_folds}")

    # === 4. Делим данные
    X_train, X_val = all_matrices[train_idx], all_matrices[val_idx]
    Stats_train, Stats_val = all_stats_norm[train_idx], all_stats_norm[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # === 5. Преобразуем в тензоры
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    Stats_train_tensor = torch.FloatTensor(Stats_train)
    y_train_tensor = torch.LongTensor(y_train)

    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    Stats_val_tensor = torch.FloatTensor(Stats_val)
    y_val_tensor = torch.LongTensor(y_val)

    # === 6. Делаем DataLoader-ы
    train_dataset = TensorDataset(X_train_tensor, Stats_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Stats_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === 7. Здесь твой цикл обучения
    Gnn.init_generator()
    Gnn.init_discriminator()

    Gnn.train(train_loader)

    Vall.init_ResNet()
    accuracy, precision, recall, f1, PSNR_mean, SSIM_mean = Vall.get_metrics(val_loader, Gnn)

    # Добавляем строку
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Fold': fold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'PSNR_mean': PSNR_mean,
        'SSIM_mean': SSIM_mean
    }])], ignore_index=True)
    
    
    fold += 1

results_df.to_csv('cross_validation_GAN_results.csv', index=False)