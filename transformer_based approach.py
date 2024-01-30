# Haowei Qiu
# Student ID: 260762269
from datasets import load_dataset
import pandas as pd
import torch as torch
import time
from sklearn.model_selection import train_test_split
from torchtext.data import TabularDataset
from torchtext import data, datasets

start = time.time()

class Embedding(torch.nn.Module):
  def __init__(self, vocabulary_size, max_length, embedding_dimensions, dp_rate=0.1):
    super(Embedding, self).__init__()
    self.word_embedding = torch.nn.Embedding(vocabulary_size, embedding_dimensions)
    self.position_embedding = torch.nn.Embedding(max_length, embedding_dimensions)
    self.dp_rate = torch.nn.Dropout(dp_rate)

  def forward(self, x):
    batch_size, len_sequence = x.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_pos = torch.arange(0, len_sequence).expand(batch_size, len_sequence).to(device)
    embedding = self.word_embedding(x) + self.position_embedding(word_pos)
    return self.dp_rate(embedding)

class MHA(torch.nn.Module):
    def __init__(self, embedding_dimensions, attention_count):
        super(MHA, self).__init__()
        self.embedding_dimensions = embedding_dimensions
        self.attention_count = attention_count
        self.attention_dimensions = embedding_dimensions // attention_count
        assert (self.attention_count * self.attention_dimensions == self.embedding_dimensions), \
            'embedding does not match attentions count'
        self.queries_linear = torch.nn.Linear(self.embedding_dimensions, self.embedding_dimensions, bias = False)
        self.keys_linear = torch.nn.Linear(self.embedding_dimensions, self.embedding_dimensions, bias = False)
        self.values_linear = torch.nn.Linear(self.embedding_dimensions, self.embedding_dimensions, bias = False)
        self.fully_connected_output = torch.nn.Linear(self.attention_dimensions * self.attention_count, self.embedding_dimensions)

    def forward(self, x):
        batch_size = x.shape[0]
        sentence_len = x.shape[1]
        Q = self.queries_linear(x).reshape( batch_size, sentence_len, self.attention_count, self.attention_dimensions).permute(0, 2, 1, 3)
        K = self.keys_linear(x).reshape(batch_size, sentence_len, self.attention_count, self.attention_dimensions).permute(0, 2, 3, 1)
        V = self.values_linear(x).reshape(batch_size, sentence_len, self.attention_count, self.attention_dimensions).permute(0, 2, 1, 3)
        score_att = torch.einsum('bijk,bikl->bijl', Q, K)
        distribution_att = torch.softmax(score_att /(self.embedding_dimensions ** (1 / 2)), dim = -1)
        attention_out = torch.einsum('bijk,bikl->bijl', distribution_att, V)
        return attention_out.permute(0, 2, 1, 3).reshape(batch_size, sentence_len, self.embedding_dimensions)

class Encoder(torch.nn.Module):
    def __init__(self, embedding_dimensions, attention_count, expan_factor, dp_rate=0.1):
        super(Encoder, self).__init__()
        self.attention = MHA(embedding_dimensions, attention_count)
        self.attention_norm_layer = torch.nn.LayerNorm(embedding_dimensions)
        self.ffn_norm_layer = torch.nn.LayerNorm(embedding_dimensions)
        self.ff_layer = torch.nn.Sequential(torch.nn.Linear(embedding_dimensions, expan_factor * embedding_dimensions),torch.nn.ReLU(),
            torch.nn.Linear(expan_factor * embedding_dimensions, embedding_dimensions)
        )
        self.dp_rate = torch.nn.Dropout(dp_rate)

    def forward(self, input):
        attention_out = self.dp_rate(self.attention(input))
        input = self.attention_norm_layer(input + attention_out)

        forward_out = self.dp_rate(self.ff_layer(input))
        encoder_output = self.ffn_norm_layer(input + forward_out)
        return encoder_output

class Classifier(torch.nn.Module):
  def __init__(self, vocabulary_size, max_length, embedding_dimensions, attention_count, expan_factor):
      super(Classifier, self).__init__()
      self.embedding = Embedding(vocabulary_size, max_length, embedding_dimensions)
      self.encoder = Encoder(embedding_dimensions, attention_count, expan_factor)
      self.class_fully_connected_output = torch.nn.Linear(embedding_dimensions, 5)  # change output size to 5 for star ratings

  def forward(self, x):
    embedding = self.embedding(x)
    encoding = self.encoder(embedding)
    compact_encoding = encoding.max(dim = 1)[0]
    out = self.class_fully_connected_output(compact_encoding)
    return out

def calc_accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def model_training(model, criterion, iterator, model_optim):
    loss_in_train, temp_accuracy_train = 0, 0
    model.train()
    for batch in iterator:
        model_optim.zero_grad()
        input = batch.review
        if input.shape[1] > maximum_input_length:
            input = input[:, :maximum_input_length]
        predictions = model(input)
        target = (batch.star - 1).clamp(min = 0)
        temp_loss = criterion(predictions, target.long())
        temp_accuracy = calc_accuracy(predictions, target.long())
        temp_loss.backward()
        model_optim.step()
        loss_in_train += temp_loss.item()
        temp_accuracy_train += temp_accuracy.item()

    return loss_in_train / len(iterator), temp_accuracy_train / len(iterator)

def model_evaluation(model, criterion, iterator):
    loss_in_eval, temp_accuracy_eval = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            input = batch.review
            if input.shape[1] > maximum_input_length:
                input = input[:, :maximum_input_length]
            predictions = model(input)
            target = (batch.star - 1).clamp(min = 0)
            temp_loss = criterion(predictions, target.long())
            temp_accuracy = calc_accuracy(predictions, target.long())
            loss_in_eval += temp_loss.item()
            temp_accuracy_eval += temp_accuracy.item()

    return loss_in_eval / len(iterator), temp_accuracy_eval / len(iterator)


TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm', batch_first = True)
LABEL = data.LabelField(dtype=torch.float)

dataset = load_dataset("app_reviews")
df = dataset['train'].to_pandas()
print(df.shape)
df = df[['review', 'star']]
# df = df.head(10000)
train_df, test_df = train_test_split(df, test_size = 0.25, random_state = 42)
fields = [('review', TEXT), ('star', LABEL)]
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
train_data, test_data = TabularDataset.splits(path='', train = 'train.csv', test = 'test.csv', format='csv', fields = fields)
train_data.sort_key = lambda x: len(x.review)
test_data.sort_key = lambda x: len(x.review)
maximum_vocabulary_size = 1000
TEXT.build_vocab(train_data, max_size = maximum_vocabulary_size)
LABEL.build_vocab(train_data)
param_batch_size = 128
embedding_dimensions = 4
attention_count = 1
param_expan_factor = 2
maximum_input_length = 64
epoch_count = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data),batch_size = param_batch_size,device = device)
param_vocabulary_size = len(TEXT.vocab)
classifier = Classifier(param_vocabulary_size, maximum_input_length, embedding_dimensions, attention_count, param_expan_factor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)

model_optim = torch.optim.SGD(classifier.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device);
for epoch in range(epoch_count):
    cross_entropy_loss_train, accuracy_train = model_training(classifier, criterion, train_iterator, model_optim)
    cross_entropy_loss_test, accuracy_test = model_evaluation(classifier, criterion, test_iterator)
    print(f'Cross Entropy Loss: {cross_entropy_loss_test}')
    print(f'Accuracy: {accuracy_test}')

stop = time.time()
print(f"Training time: {stop - start}s")