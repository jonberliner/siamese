import tensorflow as tf
import numpy as np
rng = np.random
from modules import mlp, linear, static_size, ph, batch_norm
from util import Logger, Batcher
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pdb
from tensorflow.contrib.distributions import kl, Normal, Bernoulli

from scipy.spatial.distance import cdist
import sys

tsne = TSNE()

DATASET = 'mnist'
BN = True
VAE = False
DISTANCE_NETWORK = False

if DATASET == 'mnist':
    from tensorflow.examples.tutorials.mnist import input_data
    dataset = input_data.read_data_sets('MNIST_data', one_hot=False)
    legend_labels = np.arange(10)
elif DATASET == 'cifar10':
    DATA_DIR = '/data/CIFAR/cifar-10-batches-py/'
    from cifar10 import load_cifar_10
    dataset = load_cifar_10(DATA_DIR)
    legend_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
else:
    raise ValueError('DATASET must be "mnist" or "cifar10"')
DX = dataset.train.images.shape[1]
DY = 10

logit = lambda x: -tf.log((1./x) - 1.)

# pdb.set_trace()
def gumbel_bernoulli(p, temp):
    # logit_p = logit(p)
    gumbel_noise_p = -tf.log(-tf.log(tf.random_uniform(tf.shape(p))))
    gumbel_noise_np = -tf.log(-tf.log(tf.random_uniform(tf.shape(p))))

    w_p = tf.exp((tf.log(p) + gumbel_noise_p) / temp)
    w_np = tf.exp((tf.log(1. - p) + gumbel_noise_np) / temp)

    return w_p / (w_p + w_np)


with tf.name_scope('feature') as scope:
    x1, x2 = [ph((None, DX)) for _ in xrange(2)]
    y1, y2 = [ph((None), tf.uint8) for _ in xrange(2)]
    train_flag = ph(None, dtype=tf.bool)
    bn_flag = train_flag if BN else None

    match = tf.equal(y1, y2)

    DZ = 64
    DZV = 128
    HID_SIZE = [256]*3
    # match number of params
    if not DISTANCE_NETWORK:
        HID_SIZE = [h*2 for h in HID_SIZE]
    D_D_EMBEDDING = HID_SIZE[0]

    if VAE:
        with tf.name_scope('vae') as scope:
            z1_net = mlp([DX] + HID_SIZE + [ DZV*2], bn=bn_flag)
            z2_net = z1_net

            px_net = mlp([DZ] + HID_SIZE + [DX], bn=bn_flag)

            qz1 = z1_net(x1)
            qmu1, qlv1 = tf.unstack(tf.reshape(qz1, [-1, DZV, 2]), axis=2)
            qv1 = tf.nn.softplus(qlv1)
            z1 = qmu1 + tf.random_normal(tf.shape(qv1)) * qv1

            qz2 = z2_net(x2)
            qmu2, qlv2 = tf.unstack(tf.reshape(qz2, [-1, DZV, 2]), axis=2)
            qv2 = tf.nn.softplus(qlv2)
            z2 = qmu2 + tf.random_normal(tf.shape(qv2)) * qv2

            rx1 = px_net(z1)
            rx2 = px_net(z2)

        f1_net = mlp([DZV] + HID_SIZE + [DZ], bn=bn_flag)
        f2_net = f1_net

        f1 = f1_net(tf.stop_gradient(z1))
        f2 = f2_net(tf.stop_gradient(z2))
    else:
        f1_net = mlp([DX] + HID_SIZE + [DZ], bn=bn_flag)
        f2_net = f1_net

        f1 = f1_net(x1)
        f2 = f2_net(x2)


with tf.name_scope('distance') as scope:
    if DISTANCE_NETWORK:
        d_net_embedding_net = linear(DZ, D_D_EMBEDDING)
        d_net = mlp([D_D_EMBEDDING] + HID_SIZE[1:] + [1], bn=bn_flag)

        d_embedding1 = d_net_embedding_net(f1)
        d_embedding2 = d_net_embedding_net(f2)
        d_embedding = tf.nn.relu(
                        batch_norm(D_D_EMBEDDING, 2, bn_flag)\
                            (d_embedding1 + d_embedding2))
        d_logit = d_net(d_embedding)
        d = tf.nn.softplus(d_logit)

    else:
        d = tf.sqrt(tf.reduce_sum((f1 - f2)**2., 1))


with tf.name_scope('classify') as scope:
    # classifier_net = mlp([DZ, 1024, 1024, 1024, DY], bn=bn_flag)
    classifier_net = linear(DZ, DY)
    class_hat1 = classifier_net(tf.stop_gradient(f1))
    class_hat2 = classifier_net(tf.stop_gradient(f2))

    correct1 = tf.equal(tf.argmax(class_hat1, 1), tf.cast(y1, tf.int64))


loss_match = d**2.
loss_nomatch = tf.maximum(0., 1. - d)**2.

losses = tf.where(match, loss_match, loss_nomatch)
d_loss = tf.reduce_sum(losses)

c_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=class_hat1, labels=tf.cast(y1, tf.int32))
c_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=class_hat2, labels=tf.cast(y2, tf.int32))

if VAE:
    # lx1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rx1, labels=x1), 1)
    # lx2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rx2, labels=x2), 1)
    lx1 = tf.reduce_sum(kl(Bernoulli(p=x1), Bernoulli(logits=rx1), 1))
    lx2 = tf.reduce_sum(kl(Bernoulli(p=x2), Bernoulli(logits=rx2), 1))
    lz1 = tf.reduce_sum(kl(Normal(qmu1, qv1), Normal(0., 1.)), 1)
    lz2 = tf.reduce_sum(kl(Normal(qmu2, qv2), Normal(0., 1.)), 1)
    loss = tf.reduce_sum(d_loss + c_loss1 + c_loss2 + lx1 + lx2 + lz1 + lz2)
else:
    loss = tf.reduce_sum(d_loss + c_loss1 + c_loss2)

for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance'):
    loss += tf.reduce_sum(v**2.)*1e-3
    loss += tf.reduce_sum(tf.abs(v))*1e-3
# for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classify'):
#     loss += tf.reduce_sum(v**2.)*1e-3
#     loss += tf.reduce_sum(tf.abs(v))*1e-3

trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)

BS = 256
b1 = Batcher(dataset.train.num_examples, BS)
b2 = Batcher(dataset.train.num_examples, BS)
tb1 = Batcher(dataset.test.num_examples, dataset.test.num_examples)
tb2 = Batcher(dataset.test.num_examples, dataset.test.num_examples)
def prep_fd(dat, b1, b2, training):
    i1 = b1() if b1 else np.arange(dat.num_examples)
    x10 = dat.images[i1]
    y10 = dat.labels[i1]
    i2 = b2() if b2 else np.arange(dat.num_examples)
    x20 = dat.images[i2]
    y20 = dat.labels[i2]
    return {x1: x10, y1: y10, x2: x20, y2: y20, train_flag: training}

test_i = []
for l0 in np.unique(dataset.test.labels):
    pool = np.where(dataset.test.labels == l0)[0]
    test_i.append(np.random.choice(pool, 100))
test_i = np.concatenate(test_i).ravel()
test_x = dataset.test.images[test_i]
test_y = dataset.test.labels[test_i]

fig, ax = plt.subplots()

n_test = len(test_i)
get_dist = lambda i,j: n_test*j - j*(j+1)/2 + i - 1 - j

N_STEP = int(1e5)
TEST_EVERY = int(5e2)
print ' ',
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i_step in xrange(N_STEP):
        if i_step % TEST_EVERY == 0:
            plt.ion()
            features = f1.eval(feed_dict={x1: test_x, train_flag: False})
            train_c_out = correct1.eval(feed_dict=prep_fd(dataset.train, None, None, False))
            train_l_out = c_loss1.eval(feed_dict=prep_fd(dataset.train, None, None, False))
            c_out = correct1.eval(feed_dict=prep_fd(dataset.test, None, None, False))
            l_out = c_loss1.eval(feed_dict=prep_fd(dataset.test, None, None, False))
            print 'train acc step %d: %03f' % (i_step, train_c_out.sum() / float(train_c_out.shape[0]))
            print 'train loss step %d: %03f' % (i_step, train_l_out.mean())
            print 'test acc step %d: %03f' % (i_step, c_out.sum() / float(c_out.shape[0]))
            print 'test loss step %d: %03f' % (i_step, l_out.mean())
            print 'fitting tsne...'
            tsnefeatures = tsne.fit_transform(features)
            # tnse_dists = squareform(pdist(tsnefeatures))
            print 'fitting done'
            ax.clear()
            artists = []
            
            xmin, ymin = tsnefeatures.min(0)
            xmax, ymax = tsnefeatures.max(0)
            extent = xmin, xmax, ymin, ymax
            xsize = (xmax - xmin) * 0.02
            ysize = xsize  #(ymax - ymin) * 0.05

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('auto')

            cmap = iter(plt.cm.Set1(np.linspace(0, 1, DY)))
            in_d, out_d = 0., 0.
            n_in, n_out = 0, 0
            for li, l0 in enumerate(np.unique(test_y)):
                i0 = test_y==l0
                i_in = np.nonzero(i0)[0]

                # get within and between cat dists
                i_out = np.nonzero(~i0)[0]
                f_in = tsnefeatures[i_in]
                f_out = tsnefeatures[i_out]

                n_in += len(i_in)
                n_out += len(i_out)

                in_d0 = cdist(f_in, f_in)
                out_d0 = cdist(f_in, f_out)
                in_d += in_d0.sum()
                out_d += out_d0.sum()

                # plot
                f0 = tsnefeatures[i_in]
                c0 = next(cmap)
                ax.plot(f0[:,0], f0[:,1],
                        '.', 
                        color=c0, 
                        alpha=1.0, 
                        label=str(legend_labels[l0]), 
                        zorder=1)
                
                # plot example image
                for i00 in rng.choice(np.nonzero(i0)[0], 2, replace=False):
                    if DATASET == 'mnist':
                        im00 = test_x[i00].reshape([28, 28])
                    elif DATASET == 'cifar':
                        im00 = test_x[i00].reshape([32, 32, 3])
                    x00, y00 = tsnefeatures[i00]
                    extent00 = x00 - xsize, x00 + xsize, y00 - ysize, y00 + ysize
                    ax.imshow(im00, 
                              cmap=plt.cm.gray,
                              alpha=1.0,
                              extent=extent00, 
                              zorder=2)
                    
            in_d /= n_in
            out_d /= n_out
            print 'test same dist step %d: %03f' % (i_step, in_d)
            print 'test diff dist step %d: %03f' % (i_step, out_d)

            l = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)
            # l.set_zorder(0)
            plt.show()
            plt.pause(0.01)
            plt.ioff()
            # print sess.run(loss, feed_dict=prep_fd(dataset.test, tb1, tb2))
        l0, _  = sess.run([loss, trainer], feed_dict=prep_fd(dataset.train, b1, b2, True))
        print '\r%d' % (i_step),
        sys.stdout.flush()

