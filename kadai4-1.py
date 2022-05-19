#ライブラリの準備
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

"""from torch import optim
import torch.nn.functional as F

import math
import copy"""

#GPUチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, num_classes)    
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

#点描円
def circle():
    r = 2  # 半径
    n_points = 50  # 点の数+1
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = r * np.sin(theta)+2
    y = r * np.cos(theta)+2
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(x, y, ".")
    return x, y     #それぞれ円のx軸とy軸の値を持った数列

def dataload(x, y):
    #円の点のx座標とy座標の数値をリストに
    correct_x = []
    correct_y = []
    time = []
    for i in range(len(x)):
        correct_x.append(x[i])
        correct_y.append(y[i])
        time.append([i])

    #50個目の点の正解は0個目の点.それ以外の点は次の点が正解.x,y,時系列データをリストにまとめる.
    
    input_data= []
    output_data = []
    for i in range(len(correct_x)):
        if i == 49:
            input_data.append([correct_x[i], correct_y[i]])
            output_data.append([correct_x[0], correct_y[0]])
        else:
            input_data.append([correct_x[i], correct_y[i]])
            output_data.append([correct_x[i+1], correct_y[i+1]])
    input_data_time = []
    output_data_time = []
    for i in range(len(correct_x)):
        input_data_time.append([input_data[i]])
        output_data_time.append([output_data[i]])
    input_data_time = np.transpose(input_data_time, (1, 0, 2))
    output_data_time = np.transpose(output_data_time, (1, 0, 2))
    arr_2d = np.array(correct_x)
    arr_3d = np.array(input_data)
    arr_1d = np.array(input_data_time)
    print(arr_2d.shape)
    print(arr_3d.shape)
    print(arr_1d.shape)
    print(len(input_data_time))

    #バッチデータの準備儀式
    input_data_time = torch.FloatTensor(input_data_time) 
    output_data_time = torch.FloatTensor(output_data_time)
    dataset = TensorDataset(input_data_time, output_data_time)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    return input_data_time, output_data_time, train_loader
    print(input_data_time.shape)

def train(EPOCHS, model, train_loader, input_data_time, output_data_time):
    #最適化手法の定義
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    n_total_steps = len(train_loader)
    record_loss_train = []
    record_loss_test = []
    test_losses = []
    test_x = []
    test_y = []
    for epoch in range(EPOCHS):
        model.train()
        loss_train = 0
        for i, cic_train in enumerate(train_loader): 
            train_input = cic_train[0].to(device)   #ここ二行意味不
            train_output = cic_train[1].to(device)
            output = model(train_input)
            loss = criterion(model(train_input), train_output)
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        loss_train /= i+1
        record_loss_train.append(loss_train)

        loss_test = 0.0
        for j, cic_train in enumerate(train_loader):
            input_data_time.to(device) #.to(device)転送作業、CPU、GPUどっち使うか
            model = model.to(device)
            test = model(input_data_time).to(device) #open
            test_x.append(test[0][0][0])
            test_y.append(test[0][0][1])
            test_input = cic_train[0].to(device)
            test_output = cic_train[1].to(device)
            val_output = model(test_input)
            val_loss = criterion(model(test_input), test_output)
            loss_test += val_loss.item()
        loss_test /= j+1
        record_loss_test.append(loss_test)
        if epoch%10 == 0:
            print("epoch: {}, loss: {},  " \
            "val_epoch: {}, val_loss: {}".format(epoch, loss_train, epoch, loss_test))

    return test, test_x, test_y, output, val_output, record_loss_train, record_loss_test
"""
def open(model, input_data_time):
    model = model.to(device)
    input_data_0 = []
    open_xy = []
    input_data_0 = [[[input_data_time[0][0][0],input_data_time[0][0][1]]]]
    input_data_0 = np.transpose(input_data_0, (1, 0, 2))
    input_data_0 = torch.FloatTensor(input_data_0)
    arr_2d = np.array(input_data_0)
    print("input_data_0のsize = ", arr_2d.shape)
    print("input_data_0=", input_data_0)
    open_xy1 = model(input_data_0).to(device)
    open_xy.append(input_data_0)
    #open_xy0.append(model(input_data_0).to(device))
    print("open_xy1=", open_xy1)
    open_xy2 = model(open_xy1).to(device)
    print("open_xy2=", open_xy2)
    open_datax.append(input_data_time[0][0][0])
    print("X: {}",open_datax)#[tensor(2.)]
    open_datay.append(input_data_time[0][0][1])
    print("Y: {}",open_datay)#[tensor(4.)]
    for i in range(49):
        open_xy1 = model(open_xy1).to(device)
        open_xy.append(open_xy1)
    #open_xysize = np.array(open_xy)
    #print("open_xysize = ",open_xysize.shape)
    open_xy = torch.cat((open_xy), 1)
    return open_xy"""

def main():
    EPOCHS = 10000
    x, y = circle()
    # Hyper-parameters 
    input_size = 2
    hidden_size = 50 
    num_classes = 2
    num_epochs = 2
    #batch_size = 100
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)


    input_data_time, output_data_time, train_loader = dataload(x, y)
    #tensor([[x, y],[x, y],...,[x, y]])50個のx、yのリスト2*50
    test, test_x, test_y, output, val_output, record_loss_train, record_loss_test = train(EPOCHS, model, train_loader, input_data_time, output_data_time) 
    test = test.detach().numpy()
    """open_xy = open(model, input_data_time)
    print(input_data_time)
    print(open_xy)
    open_xy = open_xy.detach().numpy()"""
    #test_x = test_x.detach().numpy()
    #test_y = test_y.detach().numpy()
    val_output = val_output.detach().numpy()
    #print("test_x=",test_x)
    """output_x = []
    output_y = []
    for i in range(50):
        output_x.append([open_xy[0][i][0]])
        output_y.append([open_xy[0][i][1]])"""
    output_x = []
    output_y = []
    for i in range(50):
        output_x.append([val_output[0][i][0]])
        output_y.append([val_output[0][i][1]])
    plt.plot(output_x, output_y, linestyle="None", linewidth=0, marker='o')
    plt.show()
    plt.style.use('ggplot')
    plt.plot(record_loss_train, label='train loss')
    plt.plot(record_loss_test, label='validation loss')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()