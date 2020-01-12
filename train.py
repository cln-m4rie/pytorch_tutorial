import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim  # オプティマイザ用のライブラリ
from typing import Optional
SAVE_DIR = Path(__file__).resolve().parent / "trained_models"


class Net(nn.Module):
    # NNの各構成要素を定義
    def __init__(self):
        super(Net, self).__init__()

        # 畳み込み層とプーリング層の要素定義
        self.conv1 = nn.Conv2d(3, 6, 5)  # (入力, 出力, 畳み込みカーネル（5*5）)
        self.pool = nn.MaxPool2d(2, 2)  # (2*2)のプーリングカーネル
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全結合層の要素定義
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # (入力, 出力)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # クラス数が１０なので最終出力数は10

    # この順番でNNを構成
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1->relu->pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2->relu->pool
        x = x.view(-1, 16 * 5 * 5)  # データサイズの変更
        x = F.relu(self.fc1(x))  # fc1->relu
        x = F.relu(self.fc2(x))  # fc2->relu
        x = self.fc3(x)
        return x

def get_gpu_if_available(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e


def train(trainloader, save_dir: Path) -> Optional[Path]:
    save_dir.mkdir(exist_ok=True)
    net = Net()
    # これやると推論のときにうまく動かないのでコメントアウトしておく
    net = get_gpu_if_available(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    last_saved_model = None
    print('Start Training')
    for epoch in range(2):
        start = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # 訓練データから入力画像の行列とラベルを取り出す
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # 勾配パラメータを０にする
            optimizer.zero_grad()

            # 順伝播 → 逆伝播 → 勾配パラメータの最適化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 損失関数の変化を2000ミニバッチごとに表示
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                elapsed_time = time.time() - start
                print(f'[{epoch + 1}, {i + 1}]',
                      f'loss: {running_loss / 2000:.3f}',
                      f'time: {elapsed_time}')
                running_loss = 0.0
                start = time.time()
                last_saved_model = save_dir / f'epochs_{epoch + 1}_iter_{i + 1}.pth'
                torch.save(net.state_dict(), last_saved_model)
    print('Finished Training')

    return last_saved_model
