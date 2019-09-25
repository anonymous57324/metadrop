import numpy as np

class Data:
  def __init__(self, args):
    if args.dataset == 'omniglot':
      self.N = 20 # total num instances per class
      self.Ktr = 4800 # total num train classes
      self.Kte = 1692 # total num test classes

      # you need to prepare data
      xtr = np.load('./data/omniglot/train.npy')
      xte = np.load('./data/omniglot/test.npy')
      self.xtr = np.reshape(xtr, [4800,20,28*28*1])
      self.xte = np.reshape(xte, [1692,20,28*28*1])

    elif args.dataset == 'mimgnet':
      self.N = 600 # total num instances per class
      self.Ktr = 64 # total num train classes
      self.Kva = 16 # total num val classes
      self.Kte = 20 # total num test classes

      # you need to prepare data
      xtr = np.load('./data/mimgnet/train.npy')
      xva = np.load('./data/mimgnet/val.npy')
      xte = np.load('./data/mimgnet/test.npy')
      self.xtr = np.reshape(xtr, [64,600,84*84*3])
      self.xva = np.reshape(xva, [64,600,84*84*3])
      self.xte = np.reshape(xte, [20,600,84*84*3])
    else:
      raise ValueError('No such dataset %s' % args.dataset)

  def generate_episode(self, args, training=True, n_episodes=1):
    generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
    n_way, n_shot, n_query = args.way, args.shot, args.query
    (K,x) = (self.Ktr, self.xtr) if training else (self.Kte, self.xte)

    xs, ys, xq, yq = [], [], [], []
    for t in range(n_episodes):
      # sample WAY classes
      classes = np.random.choice(range(K), size=n_way, replace=False)

      support_set = []
      query_set = []
      for k in list(classes):
        # sample SHOT and QUERY instances
        idx = np.random.choice(range(self.N), size=n_shot+n_query, replace=False)
        x_k = x[k][idx]
        support_set.append(x_k[:n_shot])
        query_set.append(x_k[n_shot:])

      xs_k = np.concatenate(support_set, 0)
      xq_k = np.concatenate(query_set, 0)
      ys_k = generate_label(n_way, n_shot)
      yq_k = generate_label(n_way, n_query)

      xs.append(xs_k)
      xq.append(xq_k)
      ys.append(ys_k)
      yq.append(yq_k)

    xs, ys = np.stack(xs, 0), np.stack(ys, 0)
    xq, yq = np.stack(xq, 0), np.stack(yq, 0)
    return [xs, ys, xq, yq]
