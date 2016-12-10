# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('model_name', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer('batch_size', 4, 'batch size to train.')
tf.app.flags.DEFINE_string("content_layers", "vgg_16/conv3/conv3_3",
                           "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("style_layers", "vgg_16/conv1/conv1_2,vgg_16/conv2/conv2_2,vgg_16/conv3/conv3_3,vgg_16/conv4/conv4_3",
                           "Which layers to extract style from")
# tf.app.flags.DEFINE_string("style_layers", "vgg_16/conv3/conv3_3",
#                            "Which layers to extract style from")
tf.app.flags.DEFINE_string("pretrained_path", "pretrained/vgg_16.ckpt", "pretrained path")
tf.app.flags.DEFINE_string("style_paths", "img/mosaic.jpg", "Styles to train")
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'vgg_16/fc',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 1, "Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 20, "Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 0.0, "Weight for total variation loss")
tf.app.flags.DEFINE_integer("EPOCH", 1, "EPOCH")
tf.app.flags.DEFINE_string("summary_path", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_string("model_path", "models", "Path to read/write trained models")

FLAGS = tf.app.flags.FLAGS


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    tf.logging.info('Use pretrained model %s' % FLAGS.pretrained_path)

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    print('Restored pretrained variable:')
    print([v.name for v in variables_to_restore])

    return slim.assign_from_checkpoint_fn(
        FLAGS.pretrained_path,
        variables_to_restore,
        ignore_missing_vars=True)


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(width * height * num_filters)

    return grams


def get_style_features(style_paths, style_layers):
    with tf.Graph().as_default() as g:
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=1,
            is_training=False)
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        image_unprocessing_fn = preprocessing_factory.get_unprocessing(preprocessing_name)

        size = FLAGS.image_size
        images = tf.pack([reader.get_image(path, size, size, image_preprocessing_fn) for path in style_paths])
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        for layer in style_layers:
            features.append(gram(endpoints_dict[layer]))

        with tf.Session() as sess:
            init_func = _get_init_fn()
            init_func(sess)
            with open('target_style.jpg', 'wb') as f:
                images = tf.cast(images, tf.uint8)
                value = tf.image.encode_jpeg(image_unprocessing_fn(images[0, :]))
                f.write(sess.run(value))
                tf.logging.info('Target style picture is saved to: %s.' % 'target_style.jpg')
            return sess.run(features)


def main(_):
    FLAGS.style_paths = FLAGS.style_paths.split(',')
    FLAGS.style_layers = FLAGS.style_layers.split(',')
    FLAGS.content_layers = FLAGS.content_layers.split(',')

    style_features_t = get_style_features(FLAGS.style_paths, FLAGS.style_layers)
    for feature in style_features_t:
        print(feature.shape)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """最后可以试下这个sess是不是可以放到后面去"""
            # Load Model
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=1,
                is_training=False)

            preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=False)
            image_unprocessing_fn = preprocessing_factory.get_unprocessing(preprocessing_name)
            processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 'train2014/', image_preprocessing_fn)

            generated = model.net(processed_images, training=True)
            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unpack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.pack(processed_generated)
            _, endpoints_dict = network_fn(tf.concat(0, [processed_generated, processed_images]), spatial_squeeze=False)

            """Content Loss"""
            content_loss = 0
            for layer in FLAGS.content_layers:
                generated_images, content_images = tf.split(0, 2, endpoints_dict[layer])
                size = tf.size(generated_images)
                content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper

            """Style Loss"""
            style_loss = 0
            style_loss_summary = {}
            for style_grams, layer in zip(style_features_t, FLAGS.style_layers):
                generated_images, _ = tf.split(0, 2, endpoints_dict[layer])
                print(layer, generated_images.get_shape())
                size = tf.size(generated_images)
                layer_style_loss = 0
                for style_gram in style_grams:
                    """这里与style的图片数目对应，一般使用的时候都是单张风格，这里就会执行一次"""
                    """多次的时候有没有问题？试过了，会在batch上广播的"""
                    layer_style_loss += tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
                style_loss_summary[layer] = layer_style_loss
                style_loss += layer_style_loss

            """TV Loss"""
            tv_loss = total_variation_loss(generated)  # use the unprocessed image

            loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.CONTENT_WEIGHT * content_loss + FLAGS.TV_WEIGHT * tv_loss

            global_step = tf.Variable(0, name="global_step", trainable=False)
            variable_to_train = []
            for variable in tf.trainable_variables():
                if not(variable.name.startswith(FLAGS.model_name)):
                    variable_to_train.append(variable)

            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            # Statistics
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
            tf.image_summary('processed_generated', processed_generated)
            tf.image_summary('origin', tf.pack([
                image_unprocessing_fn(image) for image in tf.unpack(processed_images, axis=0, num=FLAGS.batch_size)
            ]))
            tf.image_summary('processed_images', processed_images)
            summary = tf.merge_all_summaries()

            writer = tf.train.SummaryWriter(FLAGS.summary_path)
            variables_to_restore = []
            for v in tf.all_variables():
                if not(v.name.startswith(FLAGS.model_name)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore)
            sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])
            init_func = _get_init_fn()
            init_func(sess)
            # file = tf.train.latest_checkpoint(FLAGS.model_path)
            # if file:
            #     print('Restoring model from {}'.format(file))
            #     saver.restore(sess, file)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            tf.logging.info([v.name for v in variable_to_train])
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    if step % 1 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('add summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess, FLAGS.model_path + '/fast-style-model.ckpt', global_step=step)
            except tf.errors.OutOfRangeError:
                saver.save(sess, FLAGS.model_path + '/fast-style-model-done.ckpt')
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
