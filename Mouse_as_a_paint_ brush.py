import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


batch_size=128
num_classes = 10

DATA_PATH='/home/rishi/Desktop/pytorch'
trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_dataset=torchvision.datasets.MNIST(root=DATA_PATH,train=True,transform=trans,download=False)
test_dataset=torchvision.datasets.MNIST(root=DATA_PATH,train=False,transform=trans,download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print(len(train_loader))
print(len(test_loader))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net=Net()
net = net.double()

print(net)

import torch.optim as optim
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)

loss_ls = []
acc_ls = []
for epoch in range(4):
    for i, data in enumerate(train_loader):
        images, labels = data
        outputs = net(images.double())
        loss = criterion(outputs, labels)
        loss_ls.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = labels.size(0)
        predicted = torch.max(outputs.data, 1)[1]
        correct = (predicted == labels).sum().item()
        accuracy = correct / total * 100
        acc_ls.append(accuracy)

        if (i + 1) % 100 == 0:
            print('[%d %d] Loss=%f Accuracy=%f' % (epoch + 1, i + 1, loss.item(), accuracy))

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        outputs = net(images.double())

        predictions = torch.max(outputs, 1)[1]
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total * 100
print("Test Acuracy= ", accuracy)

while(1):
    drawing = False  # true if mouse is pressed
    ix, iy = -1, -1


    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:

               cv2.circle(img, (x, y),1, (255, 255, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

    img = np.zeros((28, 28, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            cv2.imwrite('number.png', img)
        elif k == 27:
            break

    # cv2.destroyAllWindows()

    img = cv2.imread('number.png')
    resized = cv2.resize(img, (28, 28))
    #print(resized.shape)
    #cv2.imshow("Resized image", resized)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray image', gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(gray.shape)

    x = gray

    # print(x.dtype)
    # print(x)

    y = np.zeros((x.shape[0], x.shape[1]))
    y = y + x

    # print(y)
    y = torch.from_numpy(y)
    #print(y.size())
    y = y.unsqueeze(0)
    #print(y.size())
    y = y.unsqueeze(0)
    #print(y.size())
    output = net(y.double())
    prediction = torch.max(output, 1)[1]
    # print("My prediction: ", prediction.item())
    result = prediction.item()
    font = cv2.FONT_HERSHEY_SIMPLEX
    F_Result = np.ones((250, 250)) * 255
    text="My prediction : " + str(result)
    F_Result=cv2.putText(F_Result,text,(10,50),font,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow('Result', F_Result)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()