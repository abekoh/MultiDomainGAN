import tensorflow as tf
from datetime import datetime
import subprocess

FLAGS = tf.app.flags.FLAGS


def get_gpu_n():
    result = subprocess.run('nvidia-smi -L | wc -l', shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        return 0
    return int(result.stdout)


def define_flags():
    now_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # Mode
    tf.app.flags.DEFINE_boolean('png2dataset', False, 'make dataset')
    tf.app.flags.DEFINE_boolean('train_g', False, 'train GAN')

    # Common
    tf.app.flags.DEFINE_string('gpu_ids', ', '.join([str(i) for i in range(get_gpu_n())]), 'using GPU ids')
    tf.app.flags.DEFINE_string('real_h5', '', 'path of real images hdf5')
    tf.app.flags.DEFINE_integer('img_width', 64, 'image\'s width')
    tf.app.flags.DEFINE_integer('img_height', 64, 'image\'\'s height')
    tf.app.flags.DEFINE_integer('img_dim', 3, 'image\'s dimention')
    tf.app.flags.DEFINE_integer('style_presets_n', 128, 'num of font embedding ids')
    tf.app.flags.DEFINE_integer('style_z_size', 100, 'z size')
    tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')

    # Common Directories
    gan_dir = 'result/gan/' + now_str
    tf.app.flags.DEFINE_string('gan_dir', gan_dir, 'path of result\'s destination')
    tf.app.flags.DEFINE_string('real_pngs', '', 'path of real images\' directory')

    # Train GAN
    tf.app.flags.DEFINE_integer('gan_epoch_n', 10000, 'num of epoch for training GAN')
    tf.app.flags.DEFINE_integer('critic_n', 5, 'num of critics to approximate wasserstein distance')
    tf.app.flags.DEFINE_integer('sample_imgs_interval', 1, 'interval epochs of saving images')
    tf.app.flags.DEFINE_integer('sample_col_n', 40, 'sample images\' column num')
    tf.app.flags.DEFINE_integer('keep_ckpt_interval', 250, 'interval of keeping ckpts')
    tf.app.flags.DEFINE_boolean('run_tensorboard', True, 'run tensorboard or not')
    tf.app.flags.DEFINE_integer('tensorboard_port', 6006, 'port of tensorboard')


def main(argv=None):
    if FLAGS.png2dataset:
        assert FLAGS.real_pngs != '', 'have to set --real_pngs'
        assert FLAGS.real_h5 != '', 'have to set --real_h5'
        from dataset import Dataset
        dataset = Dataset(FLAGS.real_h5, 'w', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim, True)
        dataset.load_imgs(FLAGS.real_pngs)
        del dataset
    if FLAGS.train_g:
        assert FLAGS.real_h5 != '', 'have to set --font_h5'
        from train_gan import TrainingMultiDomainGAN
        gan = TrainingMultiDomainGAN()
        gan.train()
        del gan


if __name__ == '__main__':
    define_flags()
    tf.app.run()
