import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置随机种子以便结果可复现
torch.manual_seed(42)

# 超参数
batch_size = 64
noise_dim = 100
num_epochs = 50
learning_rate = 0.0002

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),  # 输出28x28图像
            nn.Tanh()  # 输出范围在[-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出范围在[0, 1]
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
generator = Generator()
discriminator = Discriminator()

# 损失和优化器
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# 训练GAN
def train_gan():
    for epoch in range(num_epochs):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)  # 扁平化为784维

            # 真实标签和假标签
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # 训练判别器
            optimizer_D.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, noise_dim)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f'Epoch [{epoch+1}/{num_epochs}] Loss D: {d_loss_real.item() + d_loss_fake.item()}, Loss G: {g_loss.item()}')

# 显示生成的图像
def show_generated_images(generator, num_images):
    z = torch.randn(num_images, noise_dim)
    generated_images = generator(z).view(-1, 1, 28, 28).detach().numpy()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(generated_images[i][0], cmap='gray')
        axes[i].axis('off')
    plt.show()

# 开始训练
train_gan()

# 生成并显示一些图像
show_generated_images(generator, 10)
