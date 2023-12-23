import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import librosa
import librosa.display
import IPython.display as ipd

import jax
from jax import numpy as jnp, jit, grad
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

metadata = pd.read_csv('musicnet_metadata.csv')
train_data_files = glob('musicnet/musicnet/train_data/*.wav')
test_data_files = glob('musicnet/musicnet/test_data/*.wav')

def wav_to_mel_spec(path):
  y, sr = librosa.load(path)
  spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512)
  return np.abs(librosa.amplitude_to_db(spec, ref=np.max))

train_data = [wav_to_mel_spec(path) for path in train_data_files]
test_data = [wav_to_mel_spec(path) for path in test_data_files]

train_data_ids = [int(path[-8:-4]) for path in train_data_files]
test_data_ids = [int(path[-8:-4]) for path in test_data_files]

train_labels = [metadata[metadata['id'] == i]['ensemble'].values[0] for i in train_data_ids]
test_labels = [metadata[metadata['id'] == i]['ensemble'].values[0] for i in test_data_ids]

labels_to_nums = {label: i for i, label in enumerate(set(train_labels))}
nums_to_labels = {i: label for label, i in labels_to_nums.items()}

x_train = np.array([x[:, :1024].reshape(512, 1024, 1) for x in train_data], np.float32)
x_test = np.array([x[:, :1024].reshape(512, 1024, 1) for x in test_data], np.float32)
y_train = np.array([labels_to_nums[label] for label in train_labels], np.int32)
y_test = np.array([labels_to_nums[label] for label in test_labels], np.int32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

batch_size = 10
train_ds = train_ds.shuffle(buffer_size=len(x_train)).batch(batch_size)
test_ds = test_ds.batch(batch_size)

class CNN(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=8, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1))
    
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)
    x = nn.Dense(features=21)(x)
    return x
  
def create_train_state(model, rng, learning_rate, momentum):
  params = model.init(rng, jnp.ones((1, *x_train.shape[1:])))['params']
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def train_step(state, batch):
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch[0])
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
  
  grad_fn = grad(loss_fn)
  return state.apply_gradients(grads=grad_fn(state.params))

@jit
def compute_metrics(state, batch):
  logits = state.apply_fn({'params': state.params}, batch[0])
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
  preds = jnp.argmax(logits, axis=1)
  acc = jnp.mean(preds == batch[1])
  return loss, acc

flax_model = CNN()
state = create_train_state(flax_model, jax.random.PRNGKey(0), learning_rate=0.01, momentum=0.9)

num_epochs = 10

for epoch in range(num_epochs):
  for batch in train_ds.as_numpy_iterator():
    state = train_step(state, batch)
  
  train_loss_list = []
  train_acc_list = []
  test_loss_list = []
  test_acc_list = []

  for batch in train_ds.as_numpy_iterator():
    loss, acc = compute_metrics(state, batch)
    train_loss_list.append(loss)
    train_acc_list.append(acc)

  for batch in test_ds.as_numpy_iterator():
    loss, acc = compute_metrics(state, batch)
    test_loss_list.append(loss)
    test_acc_list.append(acc)
  
  train_loss = sum(train_loss_list) / len(train_loss_list)
  train_acc = sum(train_acc_list) / len(train_acc_list)
  test_loss = sum(test_loss_list) / len(test_loss_list)
  test_acc = sum(test_acc_list) / len(test_acc_list)

  print(f'[epoch {epoch + 1}] train loss: {train_loss}, train acc: {train_acc}, test loss: {test_loss}, test acc: {test_acc}')