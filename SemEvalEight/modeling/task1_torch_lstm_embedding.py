import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score

from SemEvalEight.data_prep.loaders import load_subtask1_data
from SemEvalEight.data_prep import loaders
from SemEvalEight.data_prep import preprocessing
from SemEvalEight.utils import binary_classification_metrics
from keras.preprocessing.sequence import pad_sequences
from torch.nn import functional as F


class Task1TorchRNN(nn.Module):
    def __init__(self, embeddings, hidden_dim=50, n_lstm_layers=1, output_size=1):
        super(Task1TorchRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.output_size = output_size

        # Create our embedding layer to start
        self.embeddings_tensor = torch.from_numpy(embeddings)
        self.embeddings_layer = nn.Embedding(self.embeddings_tensor.size(0),  # Num words embedded
                                             self.embeddings_tensor.size(1))  # Embed dim

        self.lstm = nn.LSTM(self.embeddings_tensor.size(1),  # Inputs are embedding vectors
                            self.hidden_dim,                 # Must output to hidden dim size
                            self.n_lstm_layers, dropout=.5)

        # Create linear affinity map
        self.lin_decode = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Create Batch Norm Layer
        self.lin_decode_bn = nn.BatchNorm1d(self.hidden_dim)

        # Activation as exponential linear unit - get some gradient below zero
        self.nonlin = nn.ELU()

        # Additional feedforward hidden layer and BatchNorm Object
        self.lin_decode_out = nn.Linear(self.hidden_dim, self.output_size)
        self.lin_decode_out_bn = nn.BatchNorm1d(self.output_size)

        # Single output for binary classification through sigmoid [0, 1] for proba
        self.output_layer = nn.Sigmoid()

        # Init weights and other things
        self.init_hidden_params()

    def init_hidden_params(self):
        # Zero Bias terms
        # Glorot Init Parameters for better starting positions
        self.lin_decode.bias.data.fill_(0)
        torch.nn.init.xavier_normal(self.lin_decode.weight)

        self.lin_decode_out.bias.data.fill_(0)
        torch.nn.init.xavier_normal(self.lin_decode_out.weight)

        # Don't compute the gradient for the embeddings - this will indicate not to
        # train the parameters later on (and make it impossible to do so)
        self.embeddings_layer.weight.data.copy_(self.embeddings_tensor)
        self.embeddings_layer.weight.requires_grad = False # Set to True to tune embeddings

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.n_lstm_layers, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.n_lstm_layers, bsz, self.hidden_dim).zero_()))

    # Network forward pass graph computed here
    def forward(self, in_seq, hidden):
        # Run embeddings for the sequence through LSTM layers
        emb = self.embeddings_layer(in_seq)
        output, hidden = self.lstm(emb, hidden)

        # Output is in shape (sequence, batch, features)
        # Take the last output in the sequence, shape of (1, batch_size, hidden feature dim)
        last = F.dropout(output[-1, :, :], .5)

        # Affinity Map -> BatchNorm -> DropOut -> NonLinearity
        lin_out = self.nonlin(F.dropout(self.lin_decode_bn(self.lin_decode(last)), .5))

        nonlin_out = self.lin_decode_out_bn(self.lin_decode_out(lin_out))
        pred = self.output_layer(nonlin_out)

        return pred, hidden

def train(model, train_sequences, train_targets,
          cv_sequences=None, cv_targets=None,
          lr=0.00003, weight_decay=0.0039,
          n_epochs=200,
          batch_size=64,
          patience=5):

    if torch.cuda.is_available():
        # Data on device
        train_targets = train_targets.cuda()
        train_sequences = train_sequences.cuda()
        cv_sequences = cv_sequences.cuda()
        cv_targets = cv_targets.cuda()
        model.cuda()

    best_val_loss = 100.0
    best_val_acc  = 0.0
    impatience    = 0

    # Only optimize on parameters that have a gradient - this is to prevent training the word vectors.
    # May want to try tunning them by including in optimizer?
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Choose and optimizer - Adam does well
    #opt_func   = optim.SGD
    #opt_func   = optim.Adadelta
    opt_func   = optim.Adam
    optimizer  = opt_func(parameters, lr=lr,
                           weight_decay=weight_decay)

    # Turn on training aspects such as DropOut
    model.train()

    # Binary classification as one output -> Binary Cross-entropy Loss
    loss_func = nn.BCELoss()

    num_samples = train_sequences.size(0)
    for e in range(n_epochs):
        # If val loss doesn't keep decreasing, stop
        if impatience > patience:
            break

        cv_every     = 3
        n_cv_batches = 64

        total_loss   = 0
        total_acc    = 0
        cnt          = 0
        crnt_idx     = 0

        # Assuming data is already shuffled, iterate over batches
        while crnt_idx < (num_samples - batch_size):
            seq = train_sequences[crnt_idx:crnt_idx+batch_size]
            y = train_targets[crnt_idx:crnt_idx+batch_size]

            # Zero out accumulated values
            model.zero_grad()
            optimizer.zero_grad()

            # Run forward pass, getting the output and hidden state
            output, hidden = model(seq.view(-1, batch_size),
                                   model.init_hidden(batch_size))

            # Calc loss
            loss = loss_func(output.squeeze(), y)

            # Backprop loss
            loss.backward()

            # Prevent exploding gradient prevalent in LSTMs
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

            # Apply learning to params
            optimizer.step()

            # How good were these training outputs? Track accuracy and loss
            predictions = output.squeeze().round().data.cpu().numpy()
            actuals     = y.data.cpu().numpy()
            total_acc  += accuracy_score(actuals, predictions)

            total_loss += loss.data
            cnt        += 1
            crnt_idx   += batch_size

        epoch_train_loss = (total_loss[0]/float(cnt))
        epoch_train_acc = (total_acc/float(cnt))

        print("[%d/%d * %d] train loss: %.4f; train acc: %.3f"
              % ((e+1), n_epochs, cnt,
                 epoch_train_loss,
                 epoch_train_acc))

        # Every few iterations, do cross-validation check
        if (e%cv_every == 0) and cv_sequences is not None:
            # Exit training mode
            model.eval()

            val_acc = 0.
            val_loss = 0.

            for idx in np.random.random_integers(0, len(cv_sequences) - 128, n_cv_batches):
                # Run model and track performance
                output, _ = model(cv_sequences[idx:idx+128].view(-1, 128), model.init_hidden(128))
                val_loss += loss_func(output.squeeze(), cv_targets[idx:idx+128]).data.mean()
                predictions = output.squeeze().round().data.cpu().numpy()
                actuals = cv_targets[idx:idx+128].data.cpu().numpy()
                val_acc += accuracy_score(actuals, predictions)

            # return to training mode
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc  = val_acc
                impatience    = 0
            else:
                impatience += 1
            print("[%d] VAL LOSS: %.5f, VAL ACC: %.3f" %(impatience,
                                                         val_loss/n_cv_batches,
                                                         val_acc/n_cv_batches))

    return best_val_loss/n_cv_batches, best_val_acc/n_cv_batches

def run():
    np.random.seed(42)
    file_ixs = list(range(39))
    np.random.shuffle(file_ixs)

    X, Y = load_subtask1_data(file_ixs[:25])
    X_test, Y_test = load_subtask1_data(file_ixs[25:35])

    tokenizer, sequences = preprocessing.tokenize_texts(X, nb_words=1000)

    test_sequences = tokenizer.texts_to_sequences(X_test)
    # TODO: replace with pad_packed_sequence in torch?
    test_sequences = pad_sequences(test_sequences, maxlen=100)

    np_embeddings = loaders.load_glove_wiki_embedding(tokenizer.word_index)

    ######
    print("Torch Time")
    torch_Y = torch.from_numpy(np.array(Y).astype('float32'))
    torch_Y = Variable(torch_Y)

    torch_sequences = torch.LongTensor(sequences.astype('long'))
    var_torch_sequences = Variable(torch_sequences)


    ###
    torch_Y_test = torch.from_numpy(np.array(Y_test).astype('float32'))
    torch_Y_test = Variable(torch_Y_test)

    torch_test_sequences = torch.LongTensor(test_sequences.astype('long'))
    var_torch_test_sequences = Variable(torch_test_sequences)


    # Try various learning rates
    res = {lr: train(Task1TorchRNN(np_embeddings, n_lstm_layers=2, hidden_dim=30),
                     var_torch_sequences, torch_Y,
                     var_torch_test_sequences, torch_Y_test,
                     lr=lr)
            for lr in [0.0025, 0.003, 0.005, 0.01, 0.001] }

    print(res)

if __name__ == """__main__""":
    run()
