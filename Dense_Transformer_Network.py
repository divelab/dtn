#python 3.0
#coding:UTF-8
'''
@author Yongjun Chen 
'''
#2D version of Dense Transformer Networks
import tensorflow as tf
import numpy as np
from ops import *

class DSN_transformer(object):
    def __init__(self,input_shape,control_points_ratio):
        self.num_batch = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.num_channels = input_shape[3]
        self.out_height = self.height
        self.out_width = self.width
        self.Column_controlP_number = int(input_shape[1] / \
                        (control_points_ratio))
        self.Row_controlP_number = int(input_shape[2] / \
                        (control_points_ratio))
        init_x = np.linspace(-5,5,self.Column_controlP_number)
        init_y = np.linspace(-5,5,self.Row_controlP_number)
        x_s,y_s = np.meshgrid(init_x, init_y)       
        self.initial = np.array([x_s,y_s])

    def _local_Networks(self,input_dim,x):
        with tf.variable_scope('_local_Networks'):
            x = tf.reshape(x,[-1,self.height*self.width*self.num_channels])
            W_fc_loc1 = weight_variable([self.height*self.width*self.num_channels, 20])
            b_fc_loc1 = bias_variable([20])
            W_fc_loc2 = weight_variable([20, self.Column_controlP_number*self.Row_controlP_number*2])
            initial = self.initial.astype('float32')
            initial = initial.flatten()
            b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
            h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
            h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2)
            return h_fc_loc2

    def _makeT(self,cp):
        with tf.variable_scope('_makeT'): 
            cp = tf.reshape(cp,(-1,2,self.Column_controlP_number*self.Row_controlP_number))
            cp = tf.cast(cp,'float32')       
            N_f = tf.shape(cp)[0]             
            #c_s
            x,y = tf.linspace(-1.,1.,self.Column_controlP_number),tf.linspace(-1.,1.,self.Row_controlP_number)
            x,y = tf.meshgrid(x,y)
            xs,ys = tf.transpose(tf.reshape(x,(-1,1))),tf.transpose(tf.reshape(y,(-1,1)))
            cp_s = tf.concat([xs,ys],0)
            cp_s_trans = tf.transpose(cp_s)
            ##===Compute distance R
            xs_trans,ys_trans = tf.transpose(tf.stack([xs],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([ys],axis=2),perm=[1,0,2])
            xs, xs_trans = tf.meshgrid(xs,xs_trans);ys, ys_trans = tf.meshgrid(ys,ys_trans)
            Rx,Ry = tf.square(tf.subtract(xs,xs_trans)),tf.square(tf.subtract(ys,ys_trans))
            R = tf.add(Rx,Ry) 
            R = tf.multiply(R,tf.log(tf.clip_by_value(R,1e-10,1e+10)))
            ones = tf.ones([tf.multiply(self.Row_controlP_number,self.Column_controlP_number),1],tf.float32)
            ones_trans = tf.transpose(ones)
            zeros = tf.zeros([3,3],tf.float32)
            Deltas1 = tf.concat([ones, cp_s_trans, R],1)
            Deltas2 = tf.concat([ones_trans,cp_s],0)
            Deltas2 = tf.concat([zeros,Deltas2],1)          
            Deltas = tf.concat([Deltas1,Deltas2],0)
            ##get deltas_inv
            Deltas_inv = tf.matrix_inverse(Deltas)
            Deltas_inv = tf.expand_dims(Deltas_inv,0)
            Deltas_inv = tf.reshape(Deltas_inv,[-1])
            Deltas_inv_f = tf.tile(Deltas_inv,tf.stack([N_f]))
            Deltas_inv_f = tf.reshape(Deltas_inv_f,tf.stack([N_f,self.Column_controlP_number*self.Row_controlP_number+3, -1]))
            cp_trans =tf.transpose(cp,perm=[0,2,1])
            zeros_f_In = tf.zeros([N_f,3,2],tf.float32)
            cp = tf.concat([cp_trans,zeros_f_In],1)
            T = tf.transpose(tf.matmul(Deltas_inv_f,cp),[0,2,1])
            return T

    def _repeat(self,x, n_repeats, type):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, type)
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(self,im, x, y):
        with tf.variable_scope('_interpolate'):
            # constants
            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(self.height, 'float32')
            width_f = tf.cast(self.width, 'float32')
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0
            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = self.width
            dim1 = self.width*self.height
            base = self._repeat(tf.range(self.num_batch)*dim1, self.out_height*self.out_width,'int32')
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1
            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, self.num_channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)
            # Finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _bilinear_interpolate(self,im, im_org, x, y):
        with tf.variable_scope('_interpolate'):
            # constants
            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(self.height, 'float32')
            width_f = tf.cast(self.width, 'float32')
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0
            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = self.width
            dim1 = self.width*self.height
            base = self._repeat(tf.range(self.num_batch)*dim1, self.out_height*self.out_width, 'int32')
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = tf.expand_dims(base_y0 + x0, 1)
            idx_b = tf.expand_dims(base_y1 + x0, 1)
            idx_c = tf.expand_dims(base_y0 + x1, 1)
            idx_d = tf.expand_dims(base_y1 + x1, 1)
            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, self.num_channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.scatter_nd(idx_a, im_flat, [self.num_batch*self.out_height*self.out_width, self.num_channels])
            Ib = tf.scatter_nd(idx_b, im_flat, [self.num_batch*self.out_height*self.out_width, self.num_channels])
            Ic = tf.scatter_nd(idx_c, im_flat, [self.num_batch*self.out_height*self.out_width, self.num_channels])
            Id = tf.scatter_nd(idx_d, im_flat, [self.num_batch*self.out_height*self.out_width, self.num_channels])

            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.scatter_nd(idx_a, tf.expand_dims(((x1_f-x) * (y1_f-y)), 1), [self.num_batch*self.out_height*self.out_width, 1])
            wb = tf.scatter_nd(idx_b, tf.expand_dims(((x1_f-x) * (y-y0_f)), 1), [self.num_batch*self.out_height*self.out_width, 1])
            wc = tf.scatter_nd(idx_c, tf.expand_dims(((x-x0_f) * (y1_f-y)), 1), [self.num_batch*self.out_height*self.out_width, 1])
            wd = tf.scatter_nd(idx_d, tf.expand_dims(((x-x0_f) * (y-y0_f)), 1), [self.num_batch*self.out_height*self.out_width, 1])

            value_all = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            weight_all = tf.clip_by_value(tf.add_n([wa, wb, wc, wd]),1e-5,1e+10)
            flag = tf.less_equal(weight_all, 1e-5* tf.ones_like(weight_all))
            flag = tf.cast(flag, tf.float32)
            im_org = tf.reshape(im_org, [-1,self.num_channels])
            output = tf.add(tf.div(value_all, weight_all), tf.multiply(im_org, flag))
            return output

    def _meshgrid(self):
        with tf.variable_scope('_meshgrid'):
            x_t = tf.matmul(tf.ones(shape=tf.stack([self.out_height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, self.out_width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, self.out_height), 1),
                            tf.ones(shape=tf.stack([1, self.out_width])))
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            px,py = tf.stack([x_t_flat],axis=2),tf.stack([y_t_flat],axis=2)
            #source control points
            x,y = tf.linspace(-1.,1.,self.Column_controlP_number),tf.linspace(-1.,1.,self.Row_controlP_number)
            x,y = tf.meshgrid(x,y)
            xs,ys = tf.transpose(tf.reshape(x,(-1,1))),tf.transpose(tf.reshape(y,(-1,1)))
            cpx,cpy = tf.transpose(tf.stack([xs],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([ys],axis=2),perm=[1,0,2])
            px, cpx = tf.meshgrid(px,cpx);py, cpy = tf.meshgrid(py,cpy)           
            #Compute distance R
            Rx,Ry = tf.square(tf.subtract(px,cpx)),tf.square(tf.subtract(py,cpy))
            R = tf.add(Rx,Ry)          
            R = tf.multiply(R,tf.log(tf.clip_by_value(R,1e-10,1e+10)))
            #Source coordinates
            ones = tf.ones_like(x_t_flat) 
            grid = tf.concat([ones, x_t_flat, y_t_flat,R],0)
            grid = tf.reshape(grid,[-1])
            grid = tf.reshape(grid,[self.Column_controlP_number*self.Row_controlP_number+3,self.out_height*self.out_width])
            return grid

    def _transform(self, T, U, U_org, Trans):
        with tf.variable_scope('_transform'): 
            T = tf.reshape(T, (-1, 2, self.Column_controlP_number*self.Row_controlP_number+3))
            T = tf.cast(T, 'float32')
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            #output size is the same as input size
            grid = self._meshgrid()
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([self.num_batch]))
            grid = tf.reshape(grid, tf.stack([self.num_batch, self.Column_controlP_number*self.Row_controlP_number+3, -1]))
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(T, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            if Trans == 'Encoder':
                output_transformed = self._interpolate(
                    U, x_s_flat, y_s_flat)
            elif Trans == 'Decoder':
                output_transformed = self._bilinear_interpolate(U,U_org,x_s_flat,y_s_flat)
            else:
                print("error type")
                return
            output = tf.reshape(
                output_transformed, tf.stack([self.num_batch, self.out_height, self.out_width, self.num_channels]))
            return output

    def Encoder(self, U, U_local,name='SpatialTransformer', **kwargs):
        with tf.variable_scope(name):
            cp = self._local_Networks(U,U_local)
            self.T= self._makeT(cp)
            output = self._transform(self.T, U, U, Trans = 'Encoder')
            return output

    def Decoder(self,U, U_org,name='SpatialDecoderLayer', **kwargs):
        with tf.variable_scope(name):
            output = self._transform(self.T, U, U_org,Trans = 'Decoder')
            return output


