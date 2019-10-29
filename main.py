from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import time
import os

from model import MAML
from data import Data
from accumulator import Accumulator
from layers import get_train_op

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--save_freq', type=int, default=1000)

parser.add_argument('--n_train_iters', type=int, default=60000)
parser.add_argument('--n_test_iters', type=int, default=1000)

parser.add_argument('--dataset', type=str, default='omniglot')
parser.add_argument('--way', type=int, default=20)
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--query', type=int, default=5)

parser.add_argument('--metabatch', type=int, default=16)
parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--inner_lr', type=float, default=0.1)
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--n_test_mc_samp', type=int, default=10)

args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

savedir = './results/run' \
    if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
  os.makedirs(savedir)

# data loader
data = Data(args)
model = MAML(args)
net = model.get_loss_multiple(True)
tnet = model.get_loss_multiple(False)

def train():
  global_step = tf.train.get_or_create_global_step()
  optim = tf.train.AdamOptimizer(args.meta_lr)
  train_op = get_train_op(optim, net['centq'], clip=[-3., 3.], global_step=global_step)

  saver = tf.train.Saver(tf.trainable_variables())
  logfile = open(os.path.join(savedir, 'train.log'), 'w')

  argdict = vars(args)
  print(argdict)
  for k, v in argdict.iteritems():
      logfile.write(k + ': ' + str(v) + '\n')
  logfile.write('\n')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  train_logger = Accumulator('cents', 'centq', 'accs', 'accq')
  train_to_run = [train_op, net['cents'], net['centq'], net['accs'],
      tf.reduce_mean(net['accq'])]

  test_logger = Accumulator('cents', 'centq', 'accs', 'accq')
  test_to_run = [tnet['cents'], tnet['centq'], tnet['accs'],
      tf.reduce_mean(tnet['accq'])]

  start = time.time()
  for i in range(args.n_train_iters+1):
    epi = model.episodes
    placeholders = [epi['xs'], epi['ys'], epi['xq'], epi['yq']]
    episode = data.generate_episode(args, training=True, n_episodes=args.metabatch)
    fdtr = dict(zip(placeholders, episode))
    train_logger.accum(sess.run(train_to_run, feed_dict=fdtr))

    if i % 5 == 0:
      line = 'Iter %d start, learning rate %f' % (i, args.meta_lr)
      print('\n' + line)
      logfile.write('\n' + line + '\n')
      train_logger.print_(header='train', episode=i*args.metabatch,
          time=time.time()-start, logfile=logfile)
      train_logger.clear()

    if i % 100 == 0:
      for j in range(10):
        epi = model.episodes
        placeholders = [epi['xs'], epi['ys'], epi['xq'], epi['yq']]
        episode = data.generate_episode(args, training=False, n_episodes=args.metabatch)
        fdte= dict(zip(placeholders, episode))
        test_logger.accum(sess.run(test_to_run, feed_dict=fdte))

      test_logger.print_(header='test ', episode=i*args.metabatch,
          time=time.time()-start, logfile=logfile)
      test_logger.clear()

    if i % args.save_freq == 0:
      saver.save(sess, os.path.join(savedir, 'model'))

  logfile.close()

def test():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  saver = tf.train.Saver(tnet['weights'])
  saver.restore(sess, os.path.join(savedir, 'model'))

  f = open(os.path.join(savedir, 'test.log'), 'w', 0)

  start = time.time()
  acc = []
  for j in range(args.n_test_iters//args.metabatch):
    if j % 10 == 0:
      print('(%.3f secs) test iter %d start'%(time.time()-start,j*args.metabatch))
    epi = model.episodes
    placeholders = [epi['xs'], epi['ys'], epi['xq'], epi['yq']]
    episode = data.generate_episode(args, training=False,
        n_episodes=args.metabatch)
    fdte= dict(zip(placeholders, episode))
    acc.append(sess.run(tnet['accq'], feed_dict=fdte))

  acc = 100.*np.concatenate(acc, axis=0)

  acc_mean = np.mean(acc)
  acc_95conf = 1.96*np.std(acc)/float(np.sqrt(args.n_test_iters))

  result = 'accuracy : %f +- %f'%(acc_mean, acc_95conf)
  print(result)
  f.write(result)
  f.close()

if __name__=='__main__':
  if args.mode == 'train':
    train()
  elif args.mode == 'test':
    test()
  else:
    raise ValueError('Invalid mode %s' % args.mode)
