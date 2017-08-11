import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
	
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    graph = tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
	
    return image_input,keep_prob,layer3_out,layer4_out,layer7_out
	    
tests.test_load_vgg(load_vgg, tf)


# TODO: Implement function
# custom init with the seed set to 0 by default
def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
    return tf.random_normal(shape, dtype=dtype, seed=seed)

#tf.contrib.layers.xavier_initializer()
#tf.truncated_normal_initializer(stddev=0.01)

def conv_1x1(x, num_outputs):
    kernel_size = 1
    stride = 1
    #return tf.layers.conv2d(x, num_outputs, kernel_size, stride, kernel_initializer=custom_init)
    return tf.layers.conv2d(x,num_outputs,kernel_size,stride,kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
        
    ## layer7 1x1 convolution and upsample
    layer7_1x1 = conv_1x1(vgg_layer7_out,num_classes)
    upsample_layer7 = tf.layers.conv2d_transpose(layer7_1x1,num_classes, 4, strides=(2, 2),padding="same",kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    ## layer4 1x1 convolution
    layer4_1x1 = conv_1x1(vgg_layer4_out,num_classes)
        
    ##Skip connection pool4(layer4_conv) with upsample_layer7
    comb_4_7 = tf.add(upsample_layer7,layer4_1x1)
    
    ## upsample the comb_4_7
    upsample_comb_4_7 = tf.layers.conv2d_transpose(comb_4_7,num_classes, 4, strides=(2, 2),padding="same",kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    ## layer3 1x1 convolution
    layer3_1x1 = conv_1x1(vgg_layer3_out,num_classes)
    
    ##skip connection pool3(layer3_conv) with upsample_comb_4_7
    comb_3_4_7 = tf.add(upsample_comb_4_7,layer3_1x1)
    
    ## upsample
    layerlast_output = tf.layers.conv2d_transpose(comb_3_4_7,num_classes, 16, strides=(8, 8),padding="same",kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
   
    return layerlast_output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    labels_reshaped  = tf.reshape(correct_label, [-1, num_classes])
    logits_reshaped = tf.reshape(nn_last_layer, [-1, num_classes])
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_reshaped ,logits=logits_reshaped))
    

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    #train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits_reshaped, train_op, cross_entropy_loss
tests.test_optimize(optimize)


import matplotlib.pyplot as plt
plt.switch_backend('agg')

def learningcurvesplot(training_losses):
    fig = plt.figure()
        
    blue_line, = plt.plot(training_losses, "b-", markeredgewidth = 5,markersize=5)
    
    plt.legend([blue_line], ["training_losses"])
    fig.suptitle(' Loss plots')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    fig.savefig('learningcurves.jpg')
    

import numpy as np

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    ### Training
    training_losses = []
       
    sess.run(tf.global_variables_initializer())
    
    print("Training Cycle...")
    print()
    
    for epoch in range(epochs):
        total_loss = 0
        count_samples = 0
        for yield_value in get_batches_fn(batch_size):
            batch_x, batch_y = yield_value
            
            sess.run(train_op,feed_dict={input_image: batch_x, correct_label: batch_y, keep_prob:0.7})
            loss = sess.run(cross_entropy_loss,
                                       feed_dict={input_image: batch_x, correct_label: batch_y, keep_prob:1.0})
            
            total_loss += (loss * len(batch_x))
            count_samples += batch_size
        training_loss = total_loss / count_samples

        #Added for collecting training loss for plotting
        training_losses.append(training_loss)
                      
        #if epoch%10 == 0:
        print("EPOCH {} ...".format(1+epoch))
        print("Training Loss = {:.8f}".format(training_loss))
        print()
        
    learningcurvesplot(training_losses)

    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    learning_rate = 0.00005
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 30
        batch_size = 8
        
        correct_label = tf.placeholder(tf.float32,shape=[None, None, None, num_classes],name='correct_label')
        #learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layers_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn,train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
