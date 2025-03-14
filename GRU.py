import torch
import torch.nn as nn
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
Data = yf.download('035420.KS', start='2022-03-01', end='2025-03-08') #Pandas 형식

#데이터 시각화
# Data[['Close', 'High', 'Low', 'Open']].plot(kind='line', figsize=(8, 4), title='Naver Stock Close, High, Low, Open')
# Data['Volume'].plot(kind='line', figsize=(8, 4), title='Naver Stock Volume')
# plt.show()

print(len(Data))

#Close 열 삭제
Delete_Close_Data = Data.drop('Close', axis=1)
Only_Close_Data = Data[['Close']]

StandardScaler = StandardScaler()
MinMaxScaler = MinMaxScaler()

X_StandardScaler = StandardScaler.fit_transform(Delete_Close_Data)
y_MinMaxScaler = MinMaxScaler.fit_transform(Only_Close_Data)

X_train = X_StandardScaler[:500, :]
X_test = X_StandardScaler[500:, :]

y_train = y_MinMaxScaler[:500, :]
y_test = y_MinMaxScaler[500:, :]

# print('Training Shape :', X_train.shape, y_train.shape)
# print('Testing Shape :', X_test.shape, y_test.shape)

X_train_tensors = torch.Tensor(X_train)
X_test_tensors = torch.Tensor(X_test)

y_train_tensors = torch.Tensor(y_train)
y_test_tensors = torch.Tensor(y_test)

X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])).to(device)
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])).to(device)

# print('Training Shape :', X_train.shape, y_train.shape)
# print('Testing Shape :', X_test.shape, y_test.shape)


class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, hn = self.gru(x, h_0)

        hn = hn[-1]

        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)

        return out


epochs = 10000
lr = 0.0001

input_size = 4
hidden_size = 10
num_layers = 1
num_classes = 1

model = GRU(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1])
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)


y_train_tensors = y_train_tensors.to(device)
for epoch in range(epochs):
    outputs = model(X_train_tensors_f)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors)
    loss.backward()

    optimizer.step()
    if epoch % 100 == 0 :
        print(f'Epoch : {epoch}, loss : {loss.item():1.5f}')

df_x_ss = StandardScaler.transform(Delete_Close_Data)
df_y_ms = MinMaxScaler.transform(Only_Close_Data)

df_x_ss = torch.tensor(df_x_ss, dtype=torch.float32)
df_y_ms = torch.tensor(df_y_ms, dtype=torch.float32)

df_x_ss = df_x_ss.unsqueeze(1)

model.eval()
with torch.no_grad():
    df_x_ss = df_x_ss.to(device)
    train_predict = model(df_x_ss)

    predicted = train_predict.detach().cpu().numpy()
    label_y =df_y_ms.detach().cpu().numpy()

    predicted = MinMaxScaler.inverse_transform(predicted)
    label_y = MinMaxScaler.inverse_transform(label_y)

    plt.figure(figsize=(10, 6))
    plt.axvline(x=datetime(2024, 1, 29), color='r', linestyle='--')

    Data['pred'] = predicted

    plt.plot(Data['Close'], label='Actual Data')
    plt.plot(Data['pred'], label='Predicted Data')

    plt.title('Time-series Prediction')
    plt.legend()
    plt.show()