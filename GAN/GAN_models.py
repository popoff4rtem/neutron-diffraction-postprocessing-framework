import torch
import torch.nn as nn

# Создаем обертку для паддинга
class Pad(nn.Module):
    def __init__(self, padding):
        super(Pad, self).__init__()
        self.padding = padding

    def forward(self, x):
        return nn.functional.pad(x, self.padding)

# Пример сверточного генератора с использованием Sequential и добавлением Pad
class ConvGenerator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_shape):
        super(ConvGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim + condition_dim, 512 * 7 * 15)
        
        # Определяем сверточные слои для генерации изображения в Sequential
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x15 -> 14x30
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            Pad((0, 0, 1, 0)),  # Паддинг (left, right, top, bottom)

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 15x30 -> 30x60
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            Pad((0, 0, 1, 0)),  # Паддинг

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 31x60 -> 62x120
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 62x120 -> 125x240
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            Pad((0, 0, 1, 0)),  # Паддинг

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # 125x240 -> 250x480
            nn.Tanh()
        )

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.fc1(x).view(-1, 512, 7, 15)  # Начальное преобразование в 7x15
        x = self.main(x)
        return x[:, :, :250, :480]  # Обрезка до требуемого размера
    
# Пример сверточного дискриминатора
class ConvDiscriminator(nn.Module):
    def __init__(self, condition_dim, input_shape):
        super(ConvDiscriminator, self).__init__()
        # Полносвязный слой для встраивания condition
        self.fc1 = nn.Linear(condition_dim, input_shape[0] * input_shape[1] * input_shape[2])

        # Сверточные слои для классификации
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),  # 250x480 -> 125x240
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 125x240 -> 62x120
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 62x120 -> 31x60
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),   # 31x60 -> 15x30
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Полносвязный слой для классификации реальное/сгенерированное
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(15 * 30, 1),
            nn.Sigmoid()
        )

    def forward(self, image, condition):
        # Преобразование condition в форму для конкатенации с изображением
        cond = self.fc1(condition).view(-1, 1, image.shape[2], image.shape[3])
        x = torch.cat([image, cond], dim=1)  # Конкатенация по канальному измерению
        #print(x.size())
        x = self.main(x)
        #print(x.size())
        x = self.classifier(x)  # Вычисление итоговой вероятности
        return x