import os
import time
import argparse
import tensorflow as tf
from network import DenseTransformerNetwork

def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_epoch', 100000, '# of step in an epoch')
    flags.DEFINE_integer('test_step', 100, '# of step to test a model')
    flags.DEFINE_integer('save_step', 100, '# of step to save a model')
    flags.DEFINE_integer('summary_step', 100, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('keep_prob', 0.9, 'dropout probability')
    flags.DEFINE_boolean('use_gpu', False, 'use GPU or not')
    #validing
    flags.DEFINE_integer('valid_start_epoch',1001,'start step to test a model')
    flags.DEFINE_string('valid_end_epoch',100001,'end step to test a model')
    flags.DEFINE_string('valid_stride_of_epoch',1000,'stride to test a model')
    # data
    flags.DEFINE_string('data_dir', '/Users/Juhn/Desktop/Research Assistant/Research3/dtn_2d/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'training.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'validation.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'testing.h5', 'Testing data')
    flags.DEFINE_integer('batch', 10, 'batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 256, 'height size')
    flags.DEFINE_integer('width', 256, 'width size')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('sample_dir', './samples/', 'Sample directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
    flags.DEFINE_integer('test_epoch', 98001, 'Test or predict epoch')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 21, 'output class number')
    flags.DEFINE_integer('start_channel_num', 64, 'start number of outputs')
    flags.DEFINE_string(
        'conv_name', 'conv2d', 'Use which conv op: conv2d or co_conv2d')
    flags.DEFINE_string(
        'deconv_name', 'deconv',
        'Use which deconv op: deconv, dilated_conv, co_dilated_conv')
    # Dense Transformer Networks  
    flags.DEFINE_boolean('add_dtn', True,
        'add Dense Transformer Networks or not')
    flags.DEFINE_integer('dtn_location', 3,'The Dense Transformer Networks location')
    flags.DEFINE_string('control_points_ratio', 8,
        'Setup the ratio of control_points comparing with the Dense transformer networks input size')   
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def train():
    model = DenseTransformerNetwork(tf.Session(), configure())
    model.train()

def valid():
    valid_loss = []
    valid_accuracy = []
    valid_m_iou = []
    conf =  configure()
    model = DenseTransformerNetwork(tf.Session(), conf)
    for i in range(conf.valid_start_epoch,conf.valid_end_epoch,conf.valid_stride_of_epoch):
        loss,acc,m_iou=model.test(i)
        valid_loss.append(loss)
        valid_accuracy.append(acc)
        valid_m_iou.append(m_iou)
        print('valid_loss',valid_loss)
        print('valid_accuracy',valid_accuracy)
        print('valid_m_iou',valid_m_iou)

def predict(): 
    predict_loss = []
    predict_accuracy = []
    predict_m_iou = []
    model = DenseTransformerNetwork(tf.Session(), configure())
    loss,acc,m_iou = model.predict()
    predict_loss.append(loss)
    predict_accuracy.append(acc)
    predict_m_iou.append(m_iou)
    print('predict_loss',predict_loss)
    print('predict_accuracy',predict_accuracy)
    print('predict_m_iou',predict_m_iou)

def main(_):
    start = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    # test
    elif args.action == 'test':
        valid()
    # predict
    elif args.action == 'predict':
        predict()
    # train
    else:
        train()
    end = time.clock()
    print("program total running time",(end-start)/60)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    tf.app.run()
