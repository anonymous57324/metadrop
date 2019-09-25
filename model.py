from layers import *

class MAML:
  def __init__(self, args):
    self.dataset = args.dataset
    if self.dataset == 'omniglot':
      self.xdim, self.input_channel = 28, 1
      self.n_channel = 64 # channel dim of conv layers
    elif self.dataset == 'mimgnet':
      self.xdim, self.input_channel = 84, 3
      self.n_channel = 32

    self.numclass = args.way # num of classes per each episode
    self.n_steps = args.n_steps # num of inner gradient steps
    self.metabatch = args.metabatch # metabatch size
    self.inner_lr = args.inner_lr
    self.n_test_mc_samp = args.n_test_mc_samp

    xshape = [self.metabatch, None, self.xdim*self.xdim*self.input_channel]
    yshape = [self.metabatch, None, self.numclass]
    # 's': support, 'q': query
    self.episodes = {
        'xs': tf.placeholder(tf.float32, xshape, name='xs'),
        'ys': tf.placeholder(tf.float32, yshape, name='ys'),
        'xq': tf.placeholder(tf.float32, xshape, name='xq'),
        'yq': tf.placeholder(tf.float32, yshape, name='yq')}

    # param initializers
    self.conv_init = tf.truncated_normal_initializer(stddev=0.02)
    self.fc_init = tf.random_normal_initializer(stddev=0.02)
    self.zero_init = tf.zeros_initializer()

  # main model param
  def get_theta(self, reuse=None):
    with tf.variable_scope('theta', reuse=reuse):
      theta = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        theta['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=self.conv_init)
        theta['conv%d_b'%l] = tf.get_variable('conv%d_b'%l,
            [self.n_channel], initializer=self.zero_init)
      factor = 5*5 if self.dataset == 'mimgnet' else 1
      theta['dense_w'] = tf.get_variable('dense_w',
          [factor*self.n_channel, self.numclass], initializer=self.fc_init)
      theta['dense_b'] = tf.get_variable('dense_b',
          [self.numclass], initializer=self.zero_init)
      return theta

  def get_phi(self, reuse=None):
    with tf.variable_scope('phi', reuse=reuse):
      phi = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        phi['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=self.conv_init)
        phi['conv%d_b'%l] = tf.get_variable('conb%d_b'%l,
            [self.n_channel], initializer=self.zero_init)
      factor = 5*5 if self.dataset == 'mimgnet' else 1
      single_w = tf.get_variable('dense_w', [factor*self.n_channel, 1],
          initializer=self.fc_init)
      single_b = tf.get_variable('dense_b', [1], initializer=self.zero_init)
      phi['dense_w'] = tf.tile(single_w, [1, self.numclass])
      phi['dense_b'] = tf.tile(single_b, [self.numclass])
      return phi

  def forward(self, x, theta, phi, sample=False):
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])
    for l in [1,2,3,4]:
      wt, bt = theta['conv%d_w'%l], theta['conv%d_b'%l]
      wp, bp = phi['conv%d_w'%l], phi['conv%d_b'%l]
      x = conv_block(x, wt, bt, wp, bp, sample=sample, bn_scope='conv%d_bn'%l)
    wt, bt = theta['dense_w'], theta['dense_b']
    wp, bp = phi['dense_w'], phi['dense_b']
    x = dense_block(x, wt, bt, wp, bp, sample=sample)
    return x

  def get_loss_single(self, inputs, training, reuse=None):
    xs, ys, xq, yq = inputs
    theta = self.get_theta(reuse=reuse)
    phi = self.get_phi(reuse=reuse)

    for i in range(self.n_steps):
      inner_loss = []
      for j in range(1 if training else self.n_test_mc_samp):
        inner_logits = self.forward(xs, theta, phi, sample=True)
        inner_loss.append(cross_entropy(inner_logits, ys))
      inner_loss = tf.reduce_mean(inner_loss)

      grads = tf.gradients(inner_loss, theta.values())
      gradients = dict(zip(theta.keys(), grads))
      theta = dict(zip(theta.keys(),
        [theta[key] - self.inner_lr * gradients[key] for key in theta.keys()]))

    os = self.forward(xs, theta, phi, sample=False)
    oq = self.forward(xq, theta, phi, sample=False)
    cents, accs = cross_entropy(os, ys), accuracy(os, ys)
    centq, accq = cross_entropy(oq, yq), accuracy(oq, yq)
    return cents, accs, centq, accq

  def get_loss_multiple(self, training):
    xs, ys = self.episodes['xs'], self.episodes['ys']
    xq, yq = self.episodes['xq'], self.episodes['yq']

    get_single_train = lambda inputs: self.get_loss_single(inputs, True, reuse=False)
    get_single_test = lambda inputs: self.get_loss_single(inputs, False, reuse=True)
    get_single = get_single_train if training else get_single_test

    # map_fn: enables parallization
    cents, accs, centq, accq \
        = tf.map_fn(get_single,
            elems=(xs, ys, xq, yq),
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32),
            parallel_iterations=self.metabatch)

    net = {}
    net['cents'] = tf.reduce_mean(cents)
    net['accs'] = tf.reduce_mean(accs)
    net['centq'] = tf.reduce_mean(centq)
    net['accq'] = accq
    net['weights'] = tf.trainable_variables()
    return net
