# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time
import losses
import utils
import os

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('naming', 'mosaic', '')
tf.app.flags.DEFINE_string('model_name', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_integer('batch_size', 4, 'batch size to train.')
tf.app.flags.DEFINE_string("content_layers", "vgg_16/conv3/conv3_3",
                           "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("style_layers", "vgg_16/conv1/conv1_2,vgg_16/conv2/conv2_2,vgg_16/conv3/conv3_3,vgg_16/conv4/conv4_3",
                           "Which layers to extract style from")
tf.app.flags.DEFINE_string("pretrained_path", "pretrained/vgg_16.ckpt", "pretrained path")
tf.app.flags.DEFINE_string("style_image", "img/mosaic.jpg", "Style to train")
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'vgg_16/fc',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 1, "Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 20, "Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 0.0, "Weight for total variation loss")
tf.app.flags.DEFINE_integer("EPOCH", 2, "EPOCH")
tf.app.flags.DEFINE_string("model_path", "models", "Path to read/write trained models")

FLAGS = tf.app.flags.FLAGS


def main(_):
    FLAGS.style_layers = FLAGS.style_layers.split(',')
    FLAGS.content_layers = FLAGS.content_layers.split(',')
    style_features_t = losses.get_style_features(FLAGS)
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """最后可以试下这个sess是不是可以放到后面去"""
            """Build Network"""
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=1,
                is_training=False)

            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.model_name,
                is_training=False)
            processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 'train2014/', image_preprocessing_fn)
            generated = model.net(processed_images, training=True)
            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unpack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.pack(processed_generated)
            _, endpoints_dict = network_fn(tf.concat(0, [processed_generated, processed_images]), spatial_squeeze=False)

            """Build Losses"""
            content_loss = losses.content_loss(endpoints_dict, FLAGS.content_layers)
            style_loss, style_loss_summary = losses.style_loss(endpoints_dict, style_features_t, FLAGS.style_layers)
            tv_loss = losses.total_variation_loss(generated)  # use the unprocessed image

            loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.CONTENT_WEIGHT * content_loss + FLAGS.TV_WEIGHT * tv_loss

            """Add Summary"""
            tf.scalar_summary('losses/content loss', content_loss)
            tf.scalar_summary('losses/style loss', style_loss)
            tf.scalar_summary('losses/regularizer loss', tv_loss)

            tf.scalar_summary('weighted_losses/weighted content loss', content_loss * FLAGS.CONTENT_WEIGHT)
            tf.scalar_summary('weighted_losses/weighted style loss', style_loss * FLAGS.STYLE_WEIGHT)
            tf.scalar_summary('weighted_losses/weighted regularizer loss', tv_loss * FLAGS.TV_WEIGHT)
            tf.scalar_summary('total loss', loss)
            for layer in FLAGS.style_layers:
                tf.scalar_summary('style_losses/' + layer, style_loss_summary[layer])
            tf.image_summary('generated', generated)
            # tf.image_summary('processed_generated', processed_generated)  # May be better?
            tf.image_summary('origin', tf.pack([
                image_unprocessing_fn(image) for image in tf.unpack(processed_images, axis=0, num=FLAGS.batch_size)
            ]))
            summary = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter(training_path)

            """Prepare to Train"""
            global_step = tf.Variable(0, name="global_step", trainable=False)
            variable_to_train = []
            for variable in tf.trainable_variables():
                if not(variable.name.startswith(FLAGS.model_name)):
                    variable_to_train.append(variable)

            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            variables_to_restore = []
            for v in tf.all_variables():
                if not(v.name.startswith(FLAGS.model_name)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore)
            sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            """Start Training"""
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    if step % 10 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path + 'fast-style-model.ckpt'), global_step=step)
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
