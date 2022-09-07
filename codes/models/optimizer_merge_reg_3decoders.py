from .clr import cyclic_learning_rate
import tensorflow.compat.v1 as tf
import numpy as np

def weight_variable_glorot2(input_dim, output_dim, name=""):
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=0,
        maxval=1,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)

class optimizer():
    def __init__(self, model, preds_specific, preds_common, lr, d_net1_indexs, d_net2_indexs, d_net3_indexs, fncellscount):

        global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=global_step, learning_rate=lr*0.1, max_lr=lr, mode='exp_range', gamma=.995))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        # with tf.variable_scope('pos_neg_weights'):
            # self.pos_neg_weights = {cellidx : weight_variable_glorot2(3, 1, name='pos_neg_weights_'+str(cellidx)) for cellidx in range(fncellscount) }

        #specific cost
        self.cost_specific = 0
        for cellidx in range(fncellscount):
            # cellidx = 0
            #net1
            net1_indexs_pos = d_net1_indexs[cellidx][0]
            net1_indexs_neg = d_net1_indexs[cellidx][1]
            net1_preds_pos = tf.gather(tf.reshape(preds_specific[cellidx][0],[-1,1]), net1_indexs_pos)
            net1_preds_neg = tf.gather(tf.reshape(preds_specific[cellidx][0],[-1,1]), net1_indexs_neg)
            net1_labels_pos = np.ones(( len(net1_indexs_pos), 1), dtype=np.float32)
            net1_labels_neg = np.zeros(( len(net1_indexs_neg), 1), dtype=np.float32)
            net1_pos_weight = len(net1_indexs_neg) / len(net1_indexs_pos)
            net1_cost = net1_pos_weight * tf.losses.mean_squared_error(net1_preds_pos, net1_labels_pos) +  tf.losses.mean_squared_error(net1_preds_neg, net1_labels_neg)

            #net2
            net2_indexs_pos = d_net2_indexs[cellidx][0]
            net2_indexs_neg = d_net2_indexs[cellidx][1]
            net2_preds_pos = tf.gather(tf.reshape(preds_specific[cellidx][0],[-1,1]), net2_indexs_pos)
            net2_preds_neg = tf.gather(tf.reshape(preds_specific[cellidx][0],[-1,1]), net2_indexs_neg)
            net2_labels_pos = np.ones(( len(net2_indexs_pos), 1), dtype=np.float32)
            net2_labels_neg = np.zeros(( len(net2_indexs_neg), 1), dtype=np.float32)
            net2_pos_weight = len(net2_indexs_neg) / len(net2_indexs_pos)
            net2_cost = net2_pos_weight * tf.losses.mean_squared_error(net2_preds_pos, net2_labels_pos) +  tf.losses.mean_squared_error(net2_preds_neg, net2_labels_neg)

            #net3
            net3_indexs_pos = d_net3_indexs[cellidx][0]
            net3_indexs_neg = d_net3_indexs[cellidx][1]
            net3_preds_pos = tf.gather(tf.reshape(preds_specific[cellidx][0],[-1,1]), net3_indexs_pos)
            net3_preds_neg = tf.gather(tf.reshape(preds_specific[cellidx][0],[-1,1]), net3_indexs_neg)
            net3_labels_pos = np.ones(( len(net3_indexs_pos), 1), dtype=np.float32)
            net3_labels_neg = np.zeros(( len(net3_indexs_neg), 1), dtype=np.float32)
            net3_pos_weight = len(net3_indexs_neg) / len(net3_indexs_pos)
            net3_cost = net3_pos_weight * tf.losses.mean_squared_error(net3_preds_pos, net3_labels_pos) +  tf.losses.mean_squared_error(net3_preds_neg, net3_labels_neg)

            self.cost_onecell = 1 / (len(net1_indexs_pos) + len(net1_indexs_neg)) * net1_cost + 1 / (len(net2_indexs_pos) + len(net2_indexs_neg)) * net2_cost + 1 / (len(net3_indexs_pos) + len(net3_indexs_neg)) * net3_cost
            self.cost_specific += self.cost_onecell
            
        #common cost
        self.cost_common = 0
        for cellidx in range(fncellscount):
            # cellidx = 0
            #net1
            net1_indexs_pos = d_net1_indexs[cellidx][0]
            net1_indexs_neg = d_net1_indexs[cellidx][1]
            net1_preds_pos = tf.gather(tf.reshape(preds_common[cellidx][0],[-1,1]), net1_indexs_pos)
            net1_preds_neg = tf.gather(tf.reshape(preds_common[cellidx][0],[-1,1]), net1_indexs_neg)
            net1_labels_pos = np.ones(( len(net1_indexs_pos), 1), dtype=np.float32)
            net1_labels_neg = np.zeros(( len(net1_indexs_neg), 1), dtype=np.float32)
            net1_pos_weight = len(net1_indexs_neg) / len(net1_indexs_pos)
            net1_cost = net1_pos_weight * tf.losses.mean_squared_error(net1_preds_pos, net1_labels_pos) +  tf.losses.mean_squared_error(net1_preds_neg, net1_labels_neg)

            #net2
            net2_indexs_pos = d_net2_indexs[cellidx][0]
            net2_indexs_neg = d_net2_indexs[cellidx][1]
            net2_preds_pos = tf.gather(tf.reshape(preds_common[cellidx][0],[-1,1]), net2_indexs_pos)
            net2_preds_neg = tf.gather(tf.reshape(preds_common[cellidx][0],[-1,1]), net2_indexs_neg)
            net2_labels_pos = np.ones(( len(net2_indexs_pos), 1), dtype=np.float32)
            net2_labels_neg = np.zeros(( len(net2_indexs_neg), 1), dtype=np.float32)
            net2_pos_weight = len(net2_indexs_neg) / len(net2_indexs_pos)
            net2_cost = net2_pos_weight * tf.losses.mean_squared_error(net2_preds_pos, net2_labels_pos) +  tf.losses.mean_squared_error(net2_preds_neg, net2_labels_neg)

            #net3
            net3_indexs_pos = d_net3_indexs[cellidx][0]
            net3_indexs_neg = d_net3_indexs[cellidx][1]
            net3_preds_pos = tf.gather(tf.reshape(preds_common[cellidx][0],[-1,1]), net3_indexs_pos)
            net3_preds_neg = tf.gather(tf.reshape(preds_common[cellidx][0],[-1,1]), net3_indexs_neg)
            net3_labels_pos = np.ones(( len(net3_indexs_pos), 1), dtype=np.float32)
            net3_labels_neg = np.zeros(( len(net3_indexs_neg), 1), dtype=np.float32)
            net3_pos_weight = len(net3_indexs_neg) / len(net3_indexs_pos)
            net3_cost = net3_pos_weight * tf.losses.mean_squared_error(net3_preds_pos, net3_labels_pos) +  tf.losses.mean_squared_error(net3_preds_neg, net3_labels_neg)

            self.cost_onecell = 1 / (len(net1_indexs_pos) + len(net1_indexs_neg)) * net1_cost + 1 / (len(net2_indexs_pos) + len(net2_indexs_neg)) * net2_cost + 1 / (len(net3_indexs_pos) + len(net3_indexs_neg)) * net3_cost
            self.cost_common += self.cost_onecell

        self.cost = self.cost_specific + self.cost_common

        self.opt_op = self.optimizer.minimize(self.cost, global_step=global_step)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

