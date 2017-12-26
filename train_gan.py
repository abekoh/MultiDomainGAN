import os
import shutil
from glob import glob
import json
import time
from subprocess import Popen, PIPE

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from models import Generator, Discriminator
from ops import average_gradients
from utils import concat_imgs

FLAGS = tf.app.flags.FLAGS


class TrainingMultiDomainGAN():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._save_flags()
        self._load_dataset()
        self._prepare_training()

    def _setup_dirs(self):
        '''
        Setup output directories

        If destinations are not existed, make directories.
        '''
        if not os.path.exists(FLAGS.gan_dir):
            os.makedirs(FLAGS.gan_dir)
        self.dst_log = os.path.join(FLAGS.gan_dir, 'log')
        self.dst_samples = os.path.join(FLAGS.gan_dir, 'sample')
        if not os.path.exists(self.dst_log):
            os.mkdir(self.dst_log)
        self.dst_log_keep = os.path.join(self.dst_log, 'keep')
        if not os.path.exists(self.dst_log_keep):
            os.mkdir(self.dst_log_keep)
        if not os.path.exists(self.dst_samples):
            os.mkdir(self.dst_samples)

    def _save_flags(self):
        '''
        Save FLAGS as JSON

        Write FLAGS paramaters as 'log/flsgs.json'.
        '''
        with open(os.path.join(self.dst_log, 'flags.json'), 'w') as f:
            json.dump(FLAGS.__dict__['__flags'], f, indent=4)

    def _load_dataset(self):
        '''
        Load dataset

        Set up dataset. All of data is for training, and they are shuffled.
        '''
        self.dataset = Dataset(FLAGS.real_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim, True)
        self.dataset.set_load_data()
        self.dataset.shuffle()

    def _prepare_training(self):
        '''
        Prepare Training

        Make tensorflow's graph.
        To support Multi-GPU, divide mini-batch.
        And this program has resume function.
        If there is checkpoint file in FLAGS.gan_dir/log, load checkpoint file and restart training.
        '''
        assert FLAGS.batch_size >= FLAGS.style_presets_n, 'batch_size must be greater equal than style_presets_n'
        self.gpu_n = len(FLAGS.gpu_ids.split(','))
        self.domains = self.dataset.get_labels()
        self.domain_z_size = len(self.domains)
        self.z_size = FLAGS.style_z_size + self.domain_z_size

        with tf.device('/cpu:0'):
            # Set embeddings from uniform distribution
            style_presets_np = np.random.uniform(-1, 1, (FLAGS.style_presets_n, FLAGS.style_z_size)).astype(np.float32)
            with tf.variable_scope('embeddings'):
                self.style_presets = tf.Variable(style_presets_np, name='style_presets')

            self.style_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='style_ids')
            self.domain_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='domain_ids')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.real_imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='real_imgs')

            d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0., beta2=0.9)
            g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0., beta2=0.9)

        # Initialize lists for multi gpu
        fake_imgs = [0 for i in range(self.gpu_n)]
        d_loss = [0 for i in range(self.gpu_n)]
        g_loss = [0 for i in range(self.gpu_n)]

        d_grads = [0 for i in range(self.gpu_n)]
        g_grads = [0 for i in range(self.gpu_n)]

        divided_batch_size = FLAGS.batch_size // self.gpu_n
        is_not_first = False

        # Build graph
        for i in range(self.gpu_n):
            batch_start = i * divided_batch_size
            batch_end = (i + 1) * divided_batch_size
            with tf.device('/gpu:{}'.format(i)):
                generator = Generator(k_size=3, smallest_unit_n=64)
                discriminator = Discriminator(k_size=3, smallest_unit_n=64)

                # If sum of (font/char)_ids is less than -1, z is generated from uniform distribution
                style_z = tf.cond(tf.reduce_all(tf.equal(self.style_ids[batch_start:batch_end], -1)),
                                  lambda: tf.random_uniform((divided_batch_size, FLAGS.style_z_size), -1, 1),
                                  lambda: tf.nn.embedding_lookup(self.style_presets, self.style_ids[batch_start:batch_end]))
                domain_z = tf.one_hot(self.domain_ids[batch_start:batch_end], self.domain_z_size)
                z = tf.concat([style_z, domain_z], axis=1)

                # Generate fake images
                fake_imgs[i] = generator(z, is_reuse=is_not_first, is_train=self.is_train)

                # Calculate loss
                d_real = discriminator(self.real_imgs[batch_start:batch_end], is_reuse=is_not_first, is_train=self.is_train)
                d_fake = discriminator(fake_imgs[i], is_reuse=True, is_train=self.is_train)
                d_loss[i] = - (tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
                g_loss[i] = - tf.reduce_mean(d_fake)

                # Calculate gradient Penalty
                epsilon = tf.random_uniform((divided_batch_size, 1, 1, 1), minval=0., maxval=1.)
                interp = self.real_imgs[batch_start:batch_end] + epsilon * (fake_imgs[i] - self.real_imgs[batch_start:batch_end])
                d_interp = discriminator(interp, is_reuse=True, is_train=self.is_train)
                grads = tf.gradients(d_interp, [interp])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[-1]))
                grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                d_loss[i] += 10 * grad_penalty

                # Get trainable variables
                d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
                g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

                d_grads[i] = d_opt.compute_gradients(d_loss[i], var_list=d_vars)
                g_grads[i] = g_opt.compute_gradients(g_loss[i], var_list=g_vars)

            is_not_first = True

        with tf.device('/cpu:0'):
            self.fake_imgs = tf.concat(fake_imgs, axis=0)
            avg_d_grads = average_gradients(d_grads)
            avg_g_grads = average_gradients(g_grads)
            self.d_train = d_opt.apply_gradients(avg_d_grads)
            self.g_train = g_opt.apply_gradients(avg_g_grads)

        # Calculate summary for tensorboard
        tf.summary.scalar('d_loss', -(sum(d_loss) / len(d_loss)))
        tf.summary.scalar('g_loss', -(sum(g_loss) / len(g_loss)))
        self.summary = tf.summary.merge_all()

        # Setup session
        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver()

        # If checkpoint is found, restart training
        checkpoint = tf.train.get_checkpoint_state(self.dst_log)
        if checkpoint:
            saver_resume = tf.train.Saver()
            saver_resume.restore(self.sess, checkpoint.model_checkpoint_path)
            self.epoch_start = int(checkpoint.model_checkpoint_path.split('-')[-1])
            print('restore ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())
            self.epoch_start = 0

        # Setup writer for tensorboard
        self.writer = tf.summary.FileWriter(self.dst_log)

    def _get_ids(self, domain_selector=''):
        '''
        Get embedding ids
        '''
        # All ids are -1 -> z is generated from uniform distribution when calculate graph
        style_ids = np.ones(FLAGS.batch_size) * -1
        if type(domain_selector) == str and len(domain_selector) == 1:
            domain_ids = np.repeat(self.dataset.get_ids_from_labels(domain_selector)[0], FLAGS.batch_size).astype(np.int32)
        else:
            domain_ids = np.random.randint(0, self.domain_z_size, (FLAGS.batch_size), dtype=np.int32)
        return style_ids, domain_ids

    def train(self):
        '''
        Train GAN
        '''
        # Start tensorboard
        if FLAGS.run_tensorboard:
            self._run_tensorboard()

        for epoch_i in tqdm(range(self.epoch_start, FLAGS.gan_epoch_n), initial=self.epoch_start, total=FLAGS.gan_epoch_n):
            for domain in self.domains:
                # Approximate wasserstein distance
                for critic_i in range(FLAGS.critic_n):
                    real_imgs = self.dataset.get_random_by_labels(FLAGS.batch_size, [domain])
                    style_ids, domain_ids = self._get_ids(domain)
                    self.sess.run(self.d_train, feed_dict={self.style_ids: style_ids,
                                                           self.domain_ids: domain_ids,
                                                           self.real_imgs: real_imgs,
                                                           self.is_train: True})

                # Minimize wasserstein distance
                style_ids, domain_ids = self._get_ids(domain)
                self.sess.run(self.g_train, feed_dict={self.style_ids: style_ids,
                                                       self.domain_ids: domain_ids,
                                                       self.is_train: True})

                # Maximize character likelihood

            # Calculate losses for tensorboard
            real_imgs = self.dataset.get_random(FLAGS.batch_size)
            style_ids, domain_ids = self._get_ids()
            summary = self.sess.run(self.summary, feed_dict={self.style_ids: style_ids,
                                                             self.domain_ids: domain_ids,
                                                             self.real_imgs: real_imgs,
                                                             self.is_train: True})

            self.writer.add_summary(summary, epoch_i)

            # Save model weights
            dst_model_path = os.path.join(self.dst_log, 'result.ckpt')
            global_step = epoch_i + 1
            self.saver.save(self.sess, dst_model_path, global_step=global_step)
            if global_step % FLAGS.keep_ckpt_interval == 0:
                for f in glob(dst_model_path + '-' + str(global_step) + '.*'):
                    shutil.copy(f, self.dst_log_keep)

            # Save sample images
            if (epoch_i + 1) % FLAGS.sample_imgs_interval == 0:
                self._save_sample_imgs(epoch_i + 1)

    def _run_tensorboard(self):
        '''
        Run tensorboard
        '''
        Popen(['tensorboard', '--logdir', '{}'.format(os.path.realpath(self.dst_log)), '--port', '{}'.format(FLAGS.tensorboard_port)], stdout=PIPE)
        time.sleep(1)

    def _generate_img(self, style_ids, domain_ids, row_n, col_n):
        '''
        Generate image
        '''
        feed = {self.style_ids: style_ids, self.domain_ids: domain_ids, self.is_train: False}
        generated_imgs = self.sess.run(self.fake_imgs, feed_dict=feed)
        combined_img = concat_imgs(generated_imgs, row_n, col_n)
        combined_img = (combined_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height))
        else:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height, FLAGS.img_dim))
        return Image.fromarray(np.uint8(combined_img))

    def _init_sample_imgs_inputs(self):
        '''
        Initialize inputs for generating sample images
        '''
        self.sample_row_n = FLAGS.batch_size // FLAGS.sample_col_n
        self.sample_style_ids = np.repeat(np.arange(0, FLAGS.style_presets_n), self.domain_z_size)[:FLAGS.batch_size]
        self.sample_domain_ids = np.tile(np.arange(0, self.domain_z_size), FLAGS.style_presets_n)[:FLAGS.batch_size]

    def _save_sample_imgs(self, epoch_i):
        '''
        Save sample images
        '''
        if not hasattr(self, 'sample_style_ids'):
            self._init_sample_imgs_inputs()
        concated_img = self._generate_img(self.sample_style_ids, self.sample_domain_ids,
                                          self.sample_row_n, FLAGS.sample_col_n)
        concated_img.save(os.path.join(self.dst_samples, '{}.png'.format(epoch_i)))
