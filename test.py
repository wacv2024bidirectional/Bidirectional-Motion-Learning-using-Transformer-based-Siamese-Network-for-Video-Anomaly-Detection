
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np
import os
import PIL

import datetime
import time

import math

import sys

from sklearn.metrics import roc_curve, auc

import imageio

from tqdm import tqdm

"""# Config"""

class CONFIG:
  DATASET = "ucsdped1"
  # save location for checkpoint and images
  MODEL_PATH = "./model_sg_ped1"
  CHECKPOINT_FILE = "./model_sg_ped1/model_sg_ckpt_ucsdped1.pt"
  # Number of training folders
  FOLDERS_LIMIT = 34
  # list of gpus available
  GPU_LIST = [0,1,2,3,4,5]

DS_CONFIG = {
    "ucsdped1": {
        "extension": "tif",
        "train_path": "./UCSDped1/Train/",
        "test_path": "./UCSDped1/Test",
        "gt_path": "./Ground_Truth/UCSD_GT/ped1",
        "chans": 1,
    },
    "ucsdped2": {
        "extension": "tif",
        "train_path": "./UCSDped2/Train",
        "test_path": "./UCSDped2/Test",
        "gt_path": "./Ground_Truth/UCSD_GT/ped2",
        "chans": 1,
    },
    "cuhkavenue": {
        "extension": "jpg",
        "train_path": "./Avenue_Frames/Train",
        "test_path": "./Avenue_Frames/Test",
        "gt_path": "./Ground_Truth/Avenue_GT",
        "chans": 1,
    },
    "shanghaitech": {
        "extension": "jpg",
        "train_path": "./ShanghaiTech/Train",
        "test_path": "./ShanghaiTech/Test",
        "gt_path": "./Ground_Truth/ShanghaiTech_GT",
        "chans": 1,
    },
}

class TF_CONFIG:
  BATCH_SIZE = 16
  EPOCHS = 200

  INPUT_SHAPE = (10, 256, 256, 1)
  PATCH_SIZE = (4, 16, 16)

  LEARNING_RATE = 2e-4
  WEIGHT_DECAY = 1e-5
  BETAS = (0.5, 0.999)

  LAYER_NORM_EPS = 1e-6

  EMBED_DIM = 384
  NUM_HEADS = 12
  NUM_LAYERS = 12
  
  IN_CHANS = 1

"""# Dataset"""

def get_testing_dataset(test_path):
  imgs = []
  for img_file in sorted(os.listdir(test_path)):
    img_path = os.path.join(test_path, img_file)
    if img_path[-3:] == DS_CONFIG[CONFIG.DATASET]["extension"]:
      if DS_CONFIG[CONFIG.DATASET]["chans"] == 1:
        img = PIL.Image.open(img_path).resize((256, 256))
        img = np.array(img, dtype=np.float32) / 256.0
        imgs.append(img)
      elif DS_CONFIG[CONFIG.DATASET]["chans"] == 3:
        img = PIL.Image.open(img_path)
        img = np.array(img, dtype=np.float32) / 256.0
        img = img.resize((256, 256, 3))
        img = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
        imgs.append(img)
  return np.array(imgs)

def get_test_sw_dataset(raw_test_dataset, sequence_size=10):
  ds = []
  tg = []
  for i in range(len(raw_test_dataset) - sequence_size):
    ds.append(raw_test_dataset[i : i + sequence_size])
    tg.append(raw_test_dataset[i + sequence_size])
  return ds, tg

class UCSDDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, training_dataset, targets):
    print("training_dataset:", type(training_dataset), training_dataset.shape)
    self.training_dataset, self.targets = training_dataset, targets

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, idx):
    print("idx:", idx)
    print("training_dataset:", type(training_dataset), training_dataset.shape)
    return self.training_dataset[idx], self.targets[idx]

"""# Utils"""

def load_ckpt(model=None, optimizer=None, epoch=50):
  ckpt_file = CONFIG.CHECKPOINT_FILE.format(CONFIG.DATASET, epoch)
  if os.path.exists(ckpt_file):
    ckpt_state = torch.load(ckpt_file)
    if model:
      model.load_state_dict(ckpt_state['model'])
    if optimizer:
      optimizer.load_state_dict(ckpt_state['optimizer'])
    print("Checkpoint loaded at epoch:", ckpt_state['epoch'])
    return ckpt_state['epoch']
  else:
    print("Model not saved")
    return 1

def load_gt_data(test_num):
  gt_file_name = os.path.join(DS_CONFIG[CONFIG.DATASET]["gt_path"], "Test_gt_" + test_num + ".npy")
  data = np.load(gt_file_name)
  return data

"""# Layers"""

class TubeletEmbedding(nn.Module):
  def __init__(self, seq_size=(10, 256, 256), patch_size=(4, 16, 16), in_chans=1, embed_dim=128):
    super().__init__()
    self.num_patches = (seq_size[0] // patch_size[0]) * (seq_size[1] // patch_size[1]) * (seq_size[2] // patch_size[2])
    self.patch_size = patch_size
    self.seq_size = seq_size
    self.embed_dim = embed_dim
    self.projection = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding="valid")

  def forward(self, x):
    x = x.permute(0, 4, 1, 2, 3)
    x = self.projection(x)
    x = x.permute(0, 2, 3, 4, 1)
    x = x.reshape((x.shape[0], -1, self.embed_dim))
    return x

class PositionalEncoder(nn.Module):
  def __init__(self, embed_dim=128):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_tokens = -1

  def forward(self, encoded_tokens):
    if self.num_tokens == -1:
      self.num_tokens = encoded_tokens.shape[1]
      self.position_embedding = nn.Embedding(num_embeddings=self.num_tokens, embedding_dim=self.embed_dim)
      self.positions = torch.arange(start=0, end=self.num_tokens, step=1)
    encoded_positions = self.position_embedding(self.positions).cuda()
    encoded_tokens = encoded_tokens + encoded_positions
    return encoded_tokens

class Dense(nn.Module):
  def __init__(self, in_features, out_features=None, act_layer=None):
    super().__init__()
    out_features = out_features or in_features
    self.fc = nn.Linear(in_features, out_features)
    if act_layer:
      self.act = act_layer()
    else:
      self.act = nn.Identity()
  
  def forward(self, x):
    x = self.fc(x)
    x = self.act(x)
    return x

class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act1 = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.act2 = act_layer()

  def forward(self, x):
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.act2(x)
    return x

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim=128, num_heads=8, mlp_ratio=4, act_layer=nn.GELU):
    super().__init__()
    self.embed_dim = embed_dim
    self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
    self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
    self.layer_norm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
    self.mlp = Mlp(in_features=embed_dim, hidden_features=embed_dim*mlp_ratio, act_layer=act_layer)

  def forward(self, x):
    x1 = self.layer_norm1(x)
    attention_output, _ = self.attn(x1, x1, x1)
    x2 = attention_output + x
    x3 = self.layer_norm2(x2)
    x3 = self.mlp(x3)
    return x2 + x3

"""# Video Vision Transformer (ViViT)"""

class VideoVisionTransformer(nn.Module):
  def __init__(self, input_shape=(10, 256, 256, 1), patch_size=(4, 16, 16), in_chans=1, num_layers=8, num_heads=8, embed_dim=128, layer_norm_eps=1e-6):
    super().__init__()
    self.input_shape = input_shape
    self.tubelet_embedding = TubeletEmbedding(seq_size=input_shape, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
    self.positional_encoder = PositionalEncoder(embed_dim=embed_dim)
    self.transformer = nn.ModuleList([
        TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)
    ])
    self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=layer_norm_eps)
    self.dense = Dense(in_features=1, out_features=10)
    self.conv2dt1 = nn.ConvTranspose2d(in_channels=48, out_channels=96, kernel_size=(3, 3), stride=2, padding=1)
    self.relu = nn.ReLU()
    self.conv2dt2 = nn.ConvTranspose2d(in_channels=96, out_channels=1, kernel_size=(3, 3), stride=2, padding=1)

  def forward(self, x):
    patches = self.tubelet_embedding(x)
    encoded_patches = self.positional_encoder(patches)
    for blk in self.transformer:
      encoded_patches = blk(encoded_patches)
    representation = self.layer_norm1(encoded_patches)
    y = representation[:, :, :, None]
    y = self.dense(y)
    y = y.reshape((y.shape[0], self.input_shape[0], 64, 64, 48))
    y = y.permute(0, 1, 4, 2, 3)
    y = y.reshape(-1, 48, 64, 64)
    y = self.conv2dt1(y, output_size=(-1, 96, 128, 128))
    y = self.relu(y)
    y = self.conv2dt2(y, output_size=(-1, 1, 256, 256))
    y = y.permute(0, 2, 3, 1)
    y = y.reshape((-1, *self.input_shape))
    return y

"""# Siamese Network"""

class SiameseNet(nn.Module):
  def __init__(self,
    embed_dim=TF_CONFIG.EMBED_DIM,
    patch_size=TF_CONFIG.PATCH_SIZE,
    input_shape=TF_CONFIG.INPUT_SHAPE,
    num_layers=TF_CONFIG.NUM_LAYERS,
    num_heads=TF_CONFIG.NUM_HEADS,
    layer_norm_eps=TF_CONFIG.LAYER_NORM_EPS,
    **kwargs
  ):
    super().__init__()
    self.vivit = VideoVisionTransformer(
      input_shape=input_shape,
      num_layers=num_layers,
      num_heads=num_heads,
      embed_dim=embed_dim,
      layer_norm_eps=layer_norm_eps
    )
  
  def forward(self, x):
    y1 = self.vivit(x)
    x2 = torch.flip(y1, (0,))
    y2 = self.vivit(x2)
    y2 = torch.flip(y2, (0,))
    return y2, y1

"""# Testing"""

def eval(model, test_path):
  try:
    raw_test_dataset = get_testing_dataset(test_path)
  except OSError:
    print("[ERROR] OSError")
    return 0

  raw_test_dataset = raw_test_dataset[:, :, :, None]
  print("raw_test_dataset:", raw_test_dataset.shape)
  tds, ttg = get_test_sw_dataset(raw_test_dataset)
  td = UCSDDataset(tds, ttg)
  print("Number of images:", len(td))

  test_loader = DataLoader(td, batch_size=TF_CONFIG.BATCH_SIZE)
  print("Number of test batches:", len(test_loader))

  sr = []
  pred_imgs = []

  for it, (n_imgs, n_plus_1_imgs) in enumerate(test_loader):
    n_imgs = n_imgs.cuda()
    _, next_n_pred_imgs = model(n_imgs)
    n_plus_1_pred_imgs = next_n_pred_imgs[:, 0]
    n_plus_1_pred_imgs = n_plus_1_pred_imgs.detach().clone().cpu().numpy()
    pred_imgs.extend(n_plus_1_pred_imgs)
    n_plus_1_imgs = n_plus_1_imgs.cpu().numpy()
    frame_prediction_cost = np.array([np.linalg.norm(np.subtract(n_plus_1_pred_imgs[i], n_plus_1_imgs[i])) for i in range(len(n_plus_1_imgs))])
    sr.extend(frame_prediction_cost)

  sr = (sr - np.min(sr)) / np.max(sr)
  sr = 1.0 - sr

  sr = (sr - np.min(sr)) / (np.max(sr) - np.min(sr))

  test_end_time = time.time()

  print("Num frames:", len(sr))
  frame_num = [i + 11 for i in range(len(sr))]

  # plot the regularity scores
  fig = plt.figure(figsize=(10, 8))
  plt.plot(frame_num, sr)
  plt.ylabel('regularity score Sr(t)')
  plt.xlabel('frame t')
  # plt.show()
  fig.savefig(os.path.join(CONFIG.MODEL_PATH, f"regularity_score_{test_path[-3:]}.png"))
  plt.close(fig)

  N = 5
  mean_kernel = np.ones(N) / N
  filtered_sig = np.convolve(sr, mean_kernel, mode='valid')

  # plot the regularity scores
  fig = plt.figure(figsize=(10, 8))
  plt.plot(frame_num[2:-2], filtered_sig)
  plt.ylabel('smoothened regularity score Sr(t)')
  plt.xlabel('frame t')
  # plt.show()
  fig.savefig(os.path.join(CONFIG.MODEL_PATH, f"smoothened_regularity_score_{test_path[-3:]}.png"))
  plt.close(fig)

  gt_data = load_gt_data(test_path[-3:])
  fpr, tpr, threshold = roc_curve(gt_data[10:], sr)
  auc_val = auc(fpr, tpr)

  print("Area under the curve:", auc_val)

  fig = plt.figure(figsize=(10, 8))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr, tpr, label='AUC (area={:.3f})'.format(auc_val))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  # plt.show()
  fig.savefig(os.path.join(CONFIG.MODEL_PATH, f"roc_{test_path[-3:]}.png"))
  plt.close(fig)

  # Visualization

  test_save_folder = os.path.join(CONFIG.MODEL_PATH, test_path[-3:])
  os.makedirs(test_save_folder, exist_ok=True)

  for it, i in enumerate(pred_imgs):
    i = i[:, :, 0]*256
    i = i.astype(np.uint8)
    imageio.imwrite(os.path.join(test_save_folder, f"{it + 11}.jpg"), i)

  return test_end_time, auc_val

model = SiameseNet()
model = nn.DataParallel(model, device_ids=CONFIG.GPU_LIST)
model = model.cuda()

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("==> Generator (SiameseNet) has been created.")
print(f"-------> The model has {n_params} parameters.")

load_ckpt(model=model, epoch=100)
model.eval()

auc_values = []
time_taken = []

all_tests_start_time = time.time()

for test in tqdm(sorted(os.listdir(DS_CONFIG[CONFIG.DATASET]["test_path"]))):
  test_path = os.path.join(DS_CONFIG[CONFIG.DATASET]["test_path"], test)
  # print(test_path)
  if os.path.isdir(test_path) and test_path[-7:-3] == "Test":
    print("=" * 20)
    print("Test number:", test_path[-3:])
    test_start_time = time.time()
    test_end_time, auc_val = eval(model, test_path)
    test_time = test_end_time - test_start_time
    print("Took {}".format(str(datetime.timedelta(seconds=int(test_end_time - test_start_time)))))
    auc_values.append(auc_val)
    time_taken.append(test_time)
    

print("Took {} avg per test".format(str(datetime.timedelta(seconds=int(np.mean(time_taken))))))
print("Took {} in total for testing".format(str(datetime.timedelta(seconds=int(time.time() - all_tests_start_time)))))

fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(auc_values) + 1), auc_values)
plt.ylabel("AUC")
plt.xlabel("Test number")
fig.savefig(os.path.join(CONFIG.MODEL_PATH, "auc.png"))
plt.close(fig)

print("Average AUC value:", np.mean(auc_values))