import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import samplers
from tensorflow.keras.mixed_precision import experimental as prec

import common


class RSSM(common.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1, warm_up=1, perturb_std=0.2,
      perturb_tpc=True, perturb_csc=False, perturb_spc=True):
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
    self._perturb_std = perturb_std
    self._perturb_tpc = perturb_tpc
    self._perturb_csc = perturb_csc
    self._perturb_spc = perturb_spc
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
    # assert not self._discrete
    prior = self.img_step(prev_state, prev_action, sample)
    ################################################################################
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(embed)
    mean = self.get('obs_dist', tfkl.Dense, self._stoch, None)(x)
    stats = {'mean': mean, 'std': tf.stop_gradient(prior['std'])}
    ################################################################################
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
  def get_perturbed_stoch(self, stoch):
    # assert not self._discrete
    noise = samplers.normal(shape=stoch.shape, mean=0., stddev=1., dtype=stoch.dtype)
    return noise * self._perturb_std + stoch

  @tf.function
  def compute_losses(self, post, prior):
    # assert not self._discrete
    stoch = tf.cast(post['mean'], tf.float32)
    perturbed_stoch = self.get_perturbed_stoch(stoch)

    tpc_stoch = perturbed_stoch if self._perturb_tpc else stoch
    csc_stoch = perturbed_stoch if self._perturb_csc else stoch
    spc_stoch = perturbed_stoch if self._perturb_spc else stoch

    B, T = stoch.shape[:2]
    dist = self.get_dist(prior)

    tpc_stoch = tf.expand_dims(tpc_stoch, axis=1)
    tpc_scores = dist.log_prob(tpc_stoch)
    tpc_scores = tf.reshape(tf.transpose(tpc_scores, [0, 2, 1]), [B*T, B])
    tpc_targets = tf.repeat(tf.range(B), repeats=T)
    tpc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tpc_targets, logits=tpc_scores)
    tpc_loss = tf.math.reduce_mean(tf.reshape(tpc_loss, [B, T])[:, self._warm_up:])

    csc_loss = -tf.math.reduce_mean(dist.log_prob(csc_stoch)[:, self._warm_up:])

    spc_dist = tfd.MultivariateNormalDiag(stoch, tf.fill(stoch.shape, self._perturb_std))
    spc_stoch = tf.expand_dims(spc_stoch, axis=1)
    spc_scores = spc_dist.log_prob(spc_stoch)
    spc_scores = tf.reshape(tf.transpose(spc_scores, [0, 2, 1]), [B*T, B])
    spc_targets = tf.repeat(tf.range(B), repeats=T)
    spc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=spc_targets, logits=spc_scores)
    spc_loss = tf.math.reduce_mean(spc_loss)

    losses = {
        'tpc': tpc_loss,
        'csc': csc_loss,
        'spc': spc_loss,
    }

    return losses
  ################################################################################


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
