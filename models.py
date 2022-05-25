################################################
#       　   Multi-Layer Perceptron     　 　   #
################################################

class MLP_model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(

### Fully-connected layer
      nn.Flatten(),
      nn.ReLU(),

      nn.Linear(1690, 256),
      nn.ReLU(),
      nn.Dropout(p=0.3),

      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(p=0.3),

      nn.Linear(128, 10),
      nn.Softmax()
    )

  def forward(self, x):
    return self.layers(x)

  
################################################
#        Convolutional Neural Network          #
################################################

class CNN_model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
### Convolutional layer
      
      nn.Conv2d(1,256,kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.Conv2d(256,256,kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.AvgPool2d(3, stride=2),
      nn.BatchNorm2d(256),
      nn.Conv2d(256,256,kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.AvgPool2d(3, stride=2),  
      nn.BatchNorm2d(256),
      nn.Conv2d(256,512,kernel_size=(4,4), padding=1),
      nn.ReLU(),
      nn.AvgPool2d(1, stride=2),
      nn.BatchNorm2d(512),

### Fully-connected layer
      nn.Flatten(),
      nn.ReLU(),

      nn.Linear(7680, 256),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(128, 10),
      nn.Softmax()
    )


  def forward(self, x):
    return self.layers(x)
  
  
################################################
#          Long Short-Term Memory     　       #
################################################

class LSTM_model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_prob = dropout_prob
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

################################################
# 　  Bidirection Long Short-Term Memory    　  #
################################################

class BiLSTM_model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_prob = dropout_prob
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
      
      
################################################
#       　   Gated Recurrent Unit      　 　   #
################################################

class GRU_model(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
       
        super(GRU_model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
       
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.to(device))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
      
################################################
#       　            END          　 　       #
################################################
