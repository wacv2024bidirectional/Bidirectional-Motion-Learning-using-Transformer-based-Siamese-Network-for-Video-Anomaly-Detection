
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

from tqdm import tqdm

"""# Config"""

class CONFIG:
  DATASET = "ucsdped1"
  # save location for checkpoint and images
  MODEL_PATH = "./model_sg_ucsdped1"
  CHECKPOINT_FILE = "./model_sg_ucsdped1/model_sg_ckpt_{}_{}.pt"
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
  EPOCHS = 100

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

def get_raw_dataset(fl = 20 ):
  vids = []
  folders_count = 1
  folders_limit = fl
  for item in sorted(os.listdir(DS_CONFIG[CONFIG.DATASET]["train_path"])):
    item_path = os.path.join(DS_CONFIG[CONFIG.DATASET]["train_path"], item)
    if os.path.isdir(item_path):
      imgs = []
      print("Fetching dir no:", folders_count)
      for img_file in sorted(os.listdir(item_path)):
        img_path = os.path.join(item_path, img_file)
        if img_path[-3:] == DS_CONFIG[CONFIG.DATASET]["extension"]:
          try:
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
          except OSError:
            print("[ERROR]", img_path)
            imgs.append(imgs[-1])
      vids.append(imgs)
      if folders_count == folders_limit:
        break
      folders_count += 1
  return np.array(vids)

def get_sliding_window_dataset(raw_test_dataset, sequence_size = 10):
  test_dataset = []
  test_targets = []
  for i in range(len(raw_test_dataset) - 2 * sequence_size + 1):
    test_dataset.append(raw_test_dataset[i : i + sequence_size])
    test_targets.append(raw_test_dataset[i + sequence_size: i + 2 * sequence_size])
  return test_dataset, test_targets

def get_recons_dataset(raw_test_dataset, sequence_size = 10):
  ds = []
  tg = []
  for i in range(0, len(raw_test_dataset) - 2 * sequence_size + 1, sequence_size):
    ds.append(raw_test_dataset[i : i + sequence_size])
    tg.append(raw_test_dataset[i + sequence_size: i + 2 * sequence_size])
  return ds, tg

class UCSDDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, training_dataset, targets):
    # print("training_dataset:", type(training_dataset), training_dataset.shape)
    self.training_dataset, self.targets = training_dataset, targets

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, idx):
    # print("idx:", idx)
    # print("training_dataset:", type(training_dataset), training_dataset.shape)
    return self.training_dataset[idx], self.targets[idx]

"""# Util"""

"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import math
import warnings
import torch
import torch.distributed as dist

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def save_ckpt(epoch, model, optimizer):
  torch.save({
      'epoch': epoch,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
  }, CONFIG.CHECKPOINT_FILE.format(CONFIG.DATASET, epoch))
  print("Checkpoint saved at epoch:", epoch)

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

"""# SoftDTW"""

import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
def jacobean_product_squared_euclidean(X, Y, Bt):
    '''
    jacobean_product_squared_euclidean(X, Y, Bt):
    
    Jacobean product of squared Euclidean distance matrix and alignment matrix.
    See equations 2 and 2.5 of https://arxiv.org/abs/1703.01541
    '''
    # print(X.shape, Y.shape, Bt.shape)
    
    ones = torch.ones(Y.shape).to('cuda' if Bt.is_cuda else 'cpu')
    return 2 * (ones.matmul(Bt) * X - Y.matmul(Bt))

class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, X, Y, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, X, Y, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, X, Y, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        G = jacobean_product_squared_euclidean(X.transpose(1,2), Y.transpose(1,2), E.transpose(1,2)).transpose(1,2)

        return grad_output.view(-1, 1, 1).expand_as(G) * G, None, None, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()

        assert use_cuda, "Only the CUDA version is supported."

        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(X, Y, D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(X, Y, D_xy, self.gamma, self.bandwidth)

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
    in_chans=TF_CONFIG.IN_CHANS
  ):
    super().__init__()
    self.vivit = VideoVisionTransformer(
      input_shape=input_shape,
      patch_size=patch_size,
      in_chans=in_chans,
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

"""# Discriminator"""

class Discriminator(nn.Module):
  def __init__(self, input_shape=TF_CONFIG.INPUT_SHAPE):
    super().__init__()
    self.input_shape = input_shape

    self.conv2d1 = nn.Conv2d(1, 128, (11, 11), stride=4, padding=4)
    self.batchnorm1 = nn.BatchNorm2d(128)
    self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2)

    self.conv2d2 = nn.Conv2d(128, 64, (5, 5), stride=2, padding=2)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)

    self.conv2d3 = nn.Conv2d(64, 32, (3, 3), stride=2, padding=1)
    self.batchnorm3 = nn.BatchNorm2d(32)

    self.dense = Dense(in_features=10*16*16*32, out_features=1, act_layer=nn.Sigmoid)
  
  def forward(self, x):
    x = x.reshape(-1, self.input_shape[1], self.input_shape[2], self.input_shape[3])
    x = x.permute(0, 3, 1, 2)

    y = self.conv2d1(x)
    y = self.batchnorm1(y)
    y = self.leakyrelu1(y)

    y = self.conv2d2(y)
    y = self.batchnorm2(y)
    y = self.leakyrelu2(y)

    y = self.conv2d3(y)
    y = self.batchnorm3(y)

    y = y.reshape((-1, 10*16*16*32))
    y = self.dense(y)
    return y

"""# GAN"""

class GAN(nn.Module):
  def __init__(self, g, d):
    super().__init__()
    self.generator = g
    self.discriminator = d
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  def forward(self, n_imgs, next_n_imgs):
    print("going for generator")
    n_recons_imgs, next_n_recons_imgs = self.generator(n_imgs)
    print("going for discriminator")
    d_real = self.discriminator(next_n_imgs)
    d_fake = self.discriminator(next_n_recons_imgs)
    print("return vals")
    return n_recons_imgs, next_n_recons_imgs, d_real, d_fake

"""# Train one epoch"""


def train_one_epoch(generator, discriminator, loader, g_optimizer, d_optimizer):
  generator.train()
  discriminator.train()

  for it, (n_imgs, next_n_imgs) in tqdm(enumerate(loader)):
    n_imgs, next_n_imgs = n_imgs.cuda(), next_n_imgs.cuda()

    d_optimizer.zero_grad()

    d_real = discriminator(next_n_imgs).view(-1)
    d_loss_real = bce_loss_fn(d_real, real_labels)
    if not math.isfinite(d_loss_real.item()):
      print(f"{it + 1}: d_loss_real is {d_loss_real.item()}")
      print("Stopping training")
      sys.exit(1)
    d_loss_real.backward()
    
    n_recons_imgs, next_n_pred_imgs = generator(n_imgs)
    d_fake = discriminator(next_n_pred_imgs.detach()).view(-1)
    d_loss_fake = bce_loss_fn(d_fake, fake_labels)
    if not math.isfinite(d_loss_fake.item()):
      print(f"{it + 1}: d_loss_fake is {d_loss_fake.item()}")
      print("Stopping training")
      sys.exit(1)
    d_loss_fake.backward()

    d_loss = d_loss_real + d_loss_fake

    d_optimizer.step()

    g_optimizer.zero_grad()

    d_updated_fake = discriminator(next_n_pred_imgs.detach()).view(-1)
    g_loss_fake = bce_loss_fn(d_updated_fake, real_labels)
    if not math.isfinite(g_loss_fake.item()):
      print(f"{it + 1}: g_loss_fake is {g_loss_fake.item()}")
      print("Stopping training")
      sys.exit(1)
    g_loss_fake.backward()
    g_loss_mse = mse_loss_fn(n_imgs, n_recons_imgs)
    if not math.isfinite(g_loss_mse.item()):
      print(f"{it + 1}: g_loss_mse is {g_loss_mse.item()}")
      print("Stopping training")
      sys.exit(1)
    g_loss_mse.backward()
    next_n_imgs = next_n_imgs.detach().clone().requires_grad_(True).reshape((-1, TF_CONFIG.INPUT_SHAPE[0], TF_CONFIG.INPUT_SHAPE[1]*TF_CONFIG.INPUT_SHAPE[2]*TF_CONFIG.INPUT_SHAPE[3]))
    next_n_pred_imgs = next_n_pred_imgs.detach().clone().requires_grad_(True).reshape((-1, TF_CONFIG.INPUT_SHAPE[0], TF_CONFIG.INPUT_SHAPE[1]*TF_CONFIG.INPUT_SHAPE[2]*TF_CONFIG.INPUT_SHAPE[3]))
    g_loss_sdtw = sdtw_loss_fn(next_n_imgs, next_n_pred_imgs).mean() / max_sdtw_loss
    if not math.isfinite(g_loss_sdtw.item()):
      print(f"{it + 1}: g_loss_sdtw is {g_loss_sdtw.item()}")
      print("Stopping training")
      sys.exit(1)
    g_loss_sdtw.backward()

    g_loss = g_loss_fake + g_loss_mse + g_loss_sdtw
    print(it + 1, ": g_loss :", g_loss)

    g_optimizer.step()

    torch.cuda.synchronize()

  print("Generator training loss:", g_loss.item())
  print("Discriminator training loss:", d_loss.item())

  return g_loss.item(), d_loss.item()

"""# Training"""

# # Individual training

# torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

raw_dataset = get_raw_dataset(CONFIG.FOLDERS_LIMIT)
print("raw_dataset:", raw_dataset.shape)

training_dataset = []
training_targets = []

for i in raw_dataset:
  # td, tt = get_sliding_window_dataset(raw_dataset[i: i + CONFIG.IMAGES_LIMIT])
  td, tt = get_recons_dataset(i)
  training_dataset.extend(td)
  training_targets.extend(tt)

del raw_dataset

training_dataset = np.array(training_dataset)[:, :, :, :, np.newaxis]
training_targets = np.array(training_targets)[:, :, :, :, np.newaxis]
print("training_dataset:", training_dataset.shape)
print("training_targets:", training_targets.shape)

dataset = UCSDDataset(training_dataset, training_targets)

train_loader = DataLoader(dataset, batch_size=TF_CONFIG.BATCH_SIZE, drop_last=True)

print("Number of training batches:", len(train_loader))

generator = SiameseNet()
n_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)

print("Generator has been created")
print(f"=====> Model has {n_params} parameters")

discriminator = Discriminator()
n_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

print("Discriminator has been created")
print(f"=====> Model has {n_params} parameters")

generator = nn.DataParallel(generator, device_ids=CONFIG.GPU_LIST)
generator = generator.cuda()
discriminator = nn.DataParallel(discriminator, device_ids=CONFIG.GPU_LIST)
discriminator = discriminator.cuda()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=TF_CONFIG.LEARNING_RATE, weight_decay=TF_CONFIG.WEIGHT_DECAY)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=TF_CONFIG.LEARNING_RATE, weight_decay=TF_CONFIG.WEIGHT_DECAY)

mse_loss_fn = nn.MSELoss()
bce_loss_fn = nn.BCELoss()
sdtw_loss_fn = SoftDTW(use_cuda=True, gamma=0.1)

real_labels = torch.ones(TF_CONFIG.BATCH_SIZE).cuda()
fake_labels = torch.zeros(TF_CONFIG.BATCH_SIZE).cuda()

g_losses = []
d_losses = []

x = torch.zeros(TF_CONFIG.BATCH_SIZE, TF_CONFIG.INPUT_SHAPE[0], TF_CONFIG.INPUT_SHAPE[1]*TF_CONFIG.INPUT_SHAPE[2]*TF_CONFIG.INPUT_SHAPE[3]).cuda()
y = torch.ones(TF_CONFIG.BATCH_SIZE, TF_CONFIG.INPUT_SHAPE[0], TF_CONFIG.INPUT_SHAPE[1]*TF_CONFIG.INPUT_SHAPE[2]*TF_CONFIG.INPUT_SHAPE[3]).cuda()
max_sdtw_loss = sdtw_loss_fn(x, y).mean()
print("max_sdtw_loss:", type(max_sdtw_loss), max_sdtw_loss)

start_epoch = 1
start_time = time.time()

print(f"==> Starting training from epoch {start_epoch}")

for epoch in range(start_epoch, TF_CONFIG.EPOCHS + 1):
  epoch_start_time = time.time()
  print("Epoch:", epoch, "/", TF_CONFIG.EPOCHS)
  g_loss, d_loss = train_one_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer)
  print("Took {}".format(str(datetime.timedelta(seconds=int(time.time() - epoch_start_time)))))
  print("=" * 20)
  if epoch % 50 == 0:
    save_ckpt(epoch, generator, g_optimizer)
  g_losses.append(g_loss)
  d_losses.append(d_loss)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

if TF_CONFIG.EPOCHS % 50:
  save_ckpt(TF_CONFIG.EPOCHS, generator, g_optimizer)

fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
fig.savefig(os.path.join(CONFIG.MODEL_PATH, "g_loss.png"))
plt.close(fig)

fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
fig.savefig(os.path.join(CONFIG.MODEL_PATH, "d_loss.png"))
plt.close(fig)

fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator loss")
plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
fig.savefig(os.path.join(CONFIG.MODEL_PATH, "loss.png"))
plt.close(fig)
