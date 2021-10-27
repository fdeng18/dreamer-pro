import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class RSSM(common.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1, warm_up=1, num_prototypes=2500, proto=30,
      temperature=0.1, sinkhorn_eps=0.05, sinkhorn_iters=3):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    ################################################################################
    self._warm_up = warm_up
    self._num_prototypes = num_prototypes
    self._proto = proto
    self._temperature = temperature
    self._sinkhorn_eps = sinkhorn_eps
    self._sinkhorn_iters = sinkhorn_iters
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    self._prototypes = tf.Variable(
        initializer(shape=(num_prototypes, proto), dtype=tf.float32))
    ################################################################################
    self._cell = GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed, action = swap(embed), swap(action)
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action, sample)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden, self._act)(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('img_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  ################################################################################
  # Sinkhorn-Knopp
  @tf.function
  def sinkhorn(self, scores):
    shape = scores.shape
    K = shape[0]
    scores = tf.reshape(scores, [-1])
    log_Q = tf.nn.log_softmax(scores / self._sinkhorn_eps, axis=0)
    log_Q = tf.reshape(log_Q, [K, -1])
    N = log_Q.shape[1]
    for _ in range(self._sinkhorn_iters):
      log_row_sums = tf.math.reduce_logsumexp(log_Q, axis=1, keepdims=True)
      log_Q = log_Q - log_row_sums - math.log(K)
      log_col_sums = tf.math.reduce_logsumexp(log_Q, axis=0, keepdims=True)
      log_Q = log_Q - log_col_sums - math.log(N)
    log_Q = log_Q + math.log(N)
    Q = tf.math.exp(log_Q)
    return tf.reshape(Q, shape)

  @tf.function
  def proto_loss(self, post, embed, ema_proj):
    prototypes = tf.math.l2_normalize(self._prototypes, axis=-1)
    self._prototypes.assign(prototypes)

    obs_proj = self.get('obs_proj', tfkl.Dense, self._proto, None)(embed)
    obs_proj = tf.cast(obs_proj, tf.float32)
    obs_norm = tf.norm(obs_proj, axis=-1)
    obs_proj = tf.math.l2_normalize(obs_proj, axis=-1)

    B, T = obs_proj.shape[:2]
    obs_proj = tf.reshape(obs_proj, [B*T, self._proto])
    obs_scores = tf.linalg.matmul(self._prototypes, obs_proj, transpose_b=True)
    obs_scores = tf.reshape(obs_scores, [self._num_prototypes, B, T])
    obs_scores = obs_scores[:, :, self._warm_up:]
    obs_logits = tf.nn.log_softmax(obs_scores / self._temperature, axis=0)
    obs_logits_1, obs_logits_2 = tf.split(obs_logits, 2, axis=1)

    ema_proj = tf.reshape(ema_proj, [B*T, self._proto])
    ema_scores = tf.linalg.matmul(self._prototypes, ema_proj, transpose_b=True)
    ema_scores = tf.reshape(ema_scores, [self._num_prototypes, B, T])
    ema_scores = ema_scores[:, :, self._warm_up:]
    ema_scores_1, ema_scores_2 = tf.split(ema_scores, 2, axis=1)
    ema_targets_1 = tf.stop_gradient(self.sinkhorn(ema_scores_1))
    ema_targets_2 = tf.stop_gradient(self.sinkhorn(ema_scores_2))
    ema_targets = tf.concat([ema_targets_1, ema_targets_2], axis=1)

    feat = self.get_feat(post)
    feat_proj = self.get('feat_proj', tfkl.Dense, self._proto, None)(feat)
    feat_proj = tf.cast(feat_proj, tf.float32)
    feat_norm = tf.norm(feat_proj, axis=-1)
    feat_proj = tf.math.l2_normalize(feat_proj, axis=-1)

    feat_proj = tf.reshape(feat_proj, [B*T, self._proto])
    feat_scores = tf.linalg.matmul(self._prototypes, feat_proj, transpose_b=True)
    feat_scores = tf.reshape(feat_scores, [self._num_prototypes, B, T])
    feat_scores = feat_scores[:, :, self._warm_up:]
    feat_logits = tf.nn.log_softmax(feat_scores / self._temperature, axis=0)

    swav_loss = (
        -0.5 * tf.math.reduce_mean(
            tf.math.reduce_sum(ema_targets_2 * obs_logits_1, axis=0))
        -0.5 * tf.math.reduce_mean(
            tf.math.reduce_sum(ema_targets_1 * obs_logits_2, axis=0)))
    temp_loss = (
        -tf.math.reduce_mean(
            tf.math.reduce_sum(ema_targets * feat_logits, axis=0)))
    norm_loss = (
        +1.0 * tf.math.reduce_mean(
            tf.math.square(obs_norm - 1))
        +1.0 * tf.math.reduce_mean(
            tf.math.square(feat_norm - 1)))

    losses = {
        'swav': swav_loss,
        'temp': temp_loss,
        'norm': norm_loss,
    }

    return losses
  ################################################################################

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class ConvEncoder(common.Module):

  def __init__(
      self, depth=32, act=tf.nn.elu, kernels=(4, 4, 4, 4), keys=['image']):
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._depth = depth
    self._kernels = kernels
    self._keys = keys

  @tf.function
  def __call__(self, obs):
    if tuple(self._keys) == ('image',):
      x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
      for i, kernel in enumerate(self._kernels):
        depth = 2 ** i * self._depth
        x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
      x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
      shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
      return tf.reshape(x, shape)
    else:
      dtype = prec.global_policy().compute_dtype
      features = []
      for key in self._keys:
        value = tf.convert_to_tensor(obs[key])
        if value.dtype.is_integer:
          value = tf.cast(value, dtype)
          semilog = tf.sign(value) * tf.math.log(1 + tf.abs(value))
          features.append(semilog[..., None])
        elif len(obs[key].shape) >= 4:
          x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
          for i, kernel in enumerate(self._kernels):
            depth = 2 ** i * self._depth
            x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
          x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
          shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
          features.append(tf.reshape(x, shape))
        else:
          raise NotImplementedError((key, value.dtype, value.shape))
      return tf.concat(features, -1)


class MLP(common.Module):

  def __init__(self, shape, layers, units, act=tf.nn.elu, **out):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._out = out

  def __call__(self, features):
    x = tf.cast(features, prec.global_policy().compute_dtype)
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    return self.get('out', DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class DistLayer(common.Module):

  def __init__(self, shape, dist='mse', min_std=0.1, init_std=0.0):
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std

  def __call__(self, inputs):
    out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
      std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
      std = tf.cast(std, tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * tf.tanh(out / 5)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, common.TanhBijector())
      dist = tfd.Independent(dist, len(self._shape))
      return common.SampleDist(dist)
    if self._dist == 'trunc_normal':
      std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return common.OneHotDist(out)
    NotImplementedError(self._dist)
