import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from .layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, BilinearDecoder, DEDICOMDecoder, weight_variable_glorot

def weight_variable_glorot2(input_dim, output_dim, name=""):
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=0,
        maxval=1,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)

class mgdc():
    def __init__(self, placeholders, num_features,  num_features_nonzero, emb_dim, fncellscount, name, act=tf.nn.relu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.num_features_nonzero = num_features_nonzero
        self.emb_dim = emb_dim
        self.cellscount = fncellscount
        self.adjs1 = {cellidx : placeholders['net1_adj_norm_' + str(cellidx)] for cellidx in range(self.cellscount)}
        self.adjs2 = {cellidx : placeholders['net2_adj_norm_' + str(cellidx)] for cellidx in range(self.cellscount)}
        self.adjs3 = {cellidx : placeholders['net3_adj_norm_' + str(cellidx)] for cellidx in range(self.cellscount)}
        self.dropout = placeholders['dropout']
        self.act = act

        with tf.variable_scope(self.name + '_netweights'):
            self.netweights = {cellidx : weight_variable_glorot2(3, 1, name='netweights') for cellidx in range(self.cellscount) }

        with tf.variable_scope(self.name + '_layerweights'):
            self.layerweights = {cellidx : weight_variable_glorot2(3, 1, name='layerweights') for cellidx in range(self.cellscount)}

        # with tf.variable_scope(self.name + '_cellweights'):
        #     self.cellweights = weight_variable_glorot2(self.cellscount, 1, name='netweights') 

        with tf.variable_scope(self.name + '_loop_weights'):
            self.loop_weights0 = {cellidx:  weight_variable_glorot(self.input_dim,self.emb_dim, name='loop_weights0') for cellidx in range(self.cellscount)}
            self.loop_weights1 = {cellidx:  weight_variable_glorot(self.emb_dim,self.emb_dim, name='loop_weights1') for cellidx in range(self.cellscount)}
            self.loop_weights2 = {cellidx:  weight_variable_glorot(self.emb_dim,self.emb_dim, name='loop_weights2') for cellidx in range(self.cellscount)}

        # with tf.variable_scope(self.name + '_bilinear_weights'):
        #     self.bilinear_weights = {}
        #     for cellidx in range(self.cellscount):
        #         self.bilinear_weights['weights_'+str(cellidx)] = weight_variable_glorot(self.emb_dim, self.emb_dim, name='weights_'+str(cellidx))
        ##specific
        with tf.variable_scope(self.name + '_weights_global'):
            self.global_weights = {cellidx: weight_variable_glorot(self.emb_dim, self.emb_dim, name='weights_global') for cellidx in range(self.cellscount)}

        with tf.variable_scope(self.name + '_weights_local'):
            self.local_weights0 = {cellidx: tf.reshape( weight_variable_glorot(self.emb_dim, 1,name='weights_local_'+str(cellidx)), [-1]) for cellidx in range(self.cellscount)}
            self.local_weights1 = {cellidx: tf.reshape( weight_variable_glorot(self.emb_dim, 1,name='weights_local_'+str(cellidx)), [-1]) for cellidx in range(self.cellscount)}
            self.local_weights2 = {cellidx: tf.reshape( weight_variable_glorot(self.emb_dim, 1,name='weights_local_'+str(cellidx)), [-1]) for cellidx in range(self.cellscount)}

        #common 
        with tf.variable_scope(self.name + '_common_weights'):
            self.common_weights = weight_variable_glorot2(self.cellscount, 1, name='common_weights')

        with tf.variable_scope(self.name + '_weights_global_common'):
            self.global_weights_common = weight_variable_glorot(self.emb_dim, self.emb_dim, name='weights_global_common') 

        with tf.variable_scope(self.name + '_weights_local_common'):
            self.local_weights_common0 = {cellidx: tf.reshape( weight_variable_glorot(self.emb_dim, 1,name='weights_local_common_'+str(cellidx)), [-1]) for cellidx in range(self.cellscount)}
            self.local_weights_common1 = {cellidx: tf.reshape( weight_variable_glorot(self.emb_dim, 1,name='weights_local_common_'+str(cellidx)), [-1]) for cellidx in range(self.cellscount)}
            self.local_weights_common2 = {cellidx: tf.reshape( weight_variable_glorot(self.emb_dim, 1,name='weights_local_common_'+str(cellidx)), [-1]) for cellidx in range(self.cellscount)}

        #common denovo
        with tf.variable_scope(self.name + '_netweights_common'):
            self.netweights_common = {cellidx : weight_variable_glorot2(3, 1, name='netweights_common0') for cellidx in range(self.cellscount) }

        with tf.variable_scope(self.name + '_layerweights_common'):
            self.layerweights_common = weight_variable_glorot2(3, 1, name='layerweights_common')

        with tf.variable_scope(self.name + '_loop_weights_common'):
            self.loop_weights_common0 = {cellidx:  weight_variable_glorot(self.input_dim,self.emb_dim, name='loop_weights_common0') for cellidx in range(self.cellscount)}
            self.loop_weights_common1 = {cellidx:  weight_variable_glorot(self.emb_dim,self.emb_dim, name='loop_weights_common1') for cellidx in range(self.cellscount)}
            self.loop_weights_common2 = {cellidx:  weight_variable_glorot(self.emb_dim,self.emb_dim, name='loop_weights_common2') for cellidx in range(self.cellscount)}

        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        
        #specific
        #specific
        #specific
        self.embeddings_specific = {}
        self.reconstructions_specific = {}
        for cellidx in range(self.cellscount):
            # cellidx = 0
            #layer1
            self.layer1_specific = []
            #net1 syn
            self.layer1_specific.append( self.netweights[cellidx][0] * GraphConvolutionSparse(name='net1_layer1_'+str(cellidx),input_dim=self.input_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],num_features_nonzero=self.num_features_nonzero,dropout=self.dropout,act=lambda x:x)(self.inputs) )
            #net2 add
            self.layer1_specific.append( self.netweights[cellidx][1] * GraphConvolutionSparse(name='net2_layer1_'+str(cellidx),input_dim=self.input_dim,output_dim=self.emb_dim,adj=self.adjs2[cellidx],num_features_nonzero=self.num_features_nonzero,dropout=self.dropout,act=lambda x:x)(self.inputs) )
            #net3 ant
            self.layer1_specific.append( self.netweights[cellidx][2] * GraphConvolutionSparse(name='net3_layer1_'+str(cellidx),input_dim=self.input_dim,output_dim=self.emb_dim,adj=self.adjs3[cellidx],num_features_nonzero=self.num_features_nonzero,dropout=self.dropout,act=lambda x:x)(self.inputs) )
            self.layer1_specific.append( tf.matmul(tf.sparse.to_dense(self.inputs), self.loop_weights0[cellidx])) 
            self.layer1_specific = tf.add_n(self.layer1_specific)
            self.layer1_specific = tf.nn.relu( tf.nn.l2_normalize(self.layer1_specific, dim=0) )

            # #layer2
            self.layer2_specific = []
            #net1 syn
            self.layer2_specific.append( self.netweights[cellidx][0] * GraphConvolution(name='net1_layer2_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer1_specific) )
            #net2 add
            self.layer2_specific.append( self.netweights[cellidx][1] * GraphConvolution(name='net2_layer2_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs2[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer1_specific) )
            #net3 ant
            self.layer2_specific.append( self.netweights[cellidx][2] * GraphConvolution(name='net3_layer2_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs3[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer1_specific) )
            self.layer2_specific.append( tf.matmul(self.layer1_specific, self.loop_weights1[cellidx])) 
            self.layer2_specific = tf.add_n(self.layer2_specific)
            self.layer2_specific = tf.nn.relu( tf.nn.l2_normalize(self.layer2_specific, dim=0) )

            #layer3
            self.layer3_specific = []
            #net1 syn
            self.layer3_specific.append( self.netweights[cellidx][0] * GraphConvolution(name='net1_layer3_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer2_specific) )
            #net2 add
            self.layer3_specific.append( self.netweights[cellidx][1] * GraphConvolution(name='net2_layer3_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs2[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer2_specific) )
            #net3 ant
            self.layer3_specific.append( self.netweights[cellidx][2] * GraphConvolution(name='net3_layer3_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs3[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer2_specific) )
            self.layer3_specific.append( tf.matmul(self.layer2_specific, self.loop_weights2[cellidx])) 
            self.layer3_specific = tf.add_n(self.layer3_specific)
            self.layer3_specific = tf.nn.relu( tf.nn.l2_normalize(self.layer3_specific, dim=0) )

            #merge 3 layers
            self.embeddings_temp = tf.nn.l2_normalize((self.layerweights[cellidx][0] * self.layer1_specific + self.layerweights[cellidx][1] * self.layer2_specific + self.layerweights[cellidx][2] * self.layer3_specific), dim=0)
            self.embeddings_specific[cellidx] = self.embeddings_temp

            #decoder
            self.reconstructions_net1 = DEDICOMDecoder(name='decoder_net1', weights_global=self.global_weights[cellidx], weights_local=self.local_weights0[cellidx], act=tf.nn.sigmoid)(self.embeddings_temp)
            self.reconstructions_net2 = DEDICOMDecoder(name='decoder_net2', weights_global=self.global_weights[cellidx], weights_local=self.local_weights1[cellidx], act=tf.nn.sigmoid)(self.embeddings_temp)
            self.reconstructions_net3 = DEDICOMDecoder(name='decoder_net3', weights_global=self.global_weights[cellidx], weights_local=self.local_weights2[cellidx], act=tf.nn.sigmoid)(self.embeddings_temp)

            self.reconstructions_specific[cellidx] = [self.reconstructions_net1, self.reconstructions_net2, self.reconstructions_net3]

        #simple common simple
        #simple common simple
        #simple common simple
        self.embeddings_common = []
        for cellidx in range(self.cellscount):
            self.embeddings_common_onecell = self.common_weights[cellidx] * self.embeddings_specific[cellidx]
            self.embeddings_common.append(self.embeddings_common_onecell)

        self.embeddings_common = tf.add_n(self.embeddings_common)

        self.reconstructions_common = {}
        for cellidx in range(self.cellscount):
            self.reconstructions_common_net1 = DEDICOMDecoder(name='common_decoder_net1', weights_global=self.global_weights_common, weights_local=self.local_weights_common0[cellidx], act=tf.nn.sigmoid)(self.embeddings_common)
            self.reconstructions_common_net2 = DEDICOMDecoder(name='common_decoder_net2', weights_global=self.global_weights_common, weights_local=self.local_weights_common1[cellidx], act=tf.nn.sigmoid)(self.embeddings_common)
            self.reconstructions_common_net3 = DEDICOMDecoder(name='common_decoder_net3', weights_global=self.global_weights_common, weights_local=self.local_weights_common2[cellidx], act=tf.nn.sigmoid)(self.embeddings_common)

            self.reconstructions_common[cellidx] = [self.reconstructions_common_net1, self.reconstructions_common_net2, self.reconstructions_common_net3]

        # #common denovo
        # #common denovo
        # #common denovo
        # self.layer1_common = []
        # for cellidx in range(self.cellscount):
        #     #net1 syn
        #     self.layer1_common.append( self.netweights_common[cellidx][0] * GraphConvolutionSparse(name='net1_layer1_'+str(cellidx),input_dim=self.input_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],num_features_nonzero=self.num_features_nonzero,dropout=self.dropout,act=lambda x:x)(self.inputs) )
        #     #net2 add
        #     self.layer1_common.append( self.netweights_common[cellidx][1] * GraphConvolutionSparse(name='net2_layer1_'+str(cellidx),input_dim=self.input_dim,output_dim=self.emb_dim,adj=self.adjs2[cellidx],num_features_nonzero=self.num_features_nonzero,dropout=self.dropout,act=lambda x:x)(self.inputs) )
        #     #net3 ant
        #     self.layer1_common.append( self.netweights_common[cellidx][2] * GraphConvolutionSparse(name='net3_layer1_'+str(cellidx),input_dim=self.input_dim,output_dim=self.emb_dim,adj=self.adjs3[cellidx],num_features_nonzero=self.num_features_nonzero,dropout=self.dropout,act=lambda x:x)(self.inputs) )
        # self.layer1_common.append( tf.matmul(tf.sparse.to_dense(self.inputs), self.loop_weights_common0[cellidx])) 
        # self.layer1_common = tf.add_n(self.layer1_common)
        # self.layer1_common = tf.nn.relu( tf.nn.l2_normalize(self.layer1_common, dim=0) )

        # # #layer2
        # self.layer2_common = []
        # for cellidx in range(self.cellscount):
        #     #net1 syn
        #     self.layer2_common.append( self.netweights_common[cellidx][0] * GraphConvolution(name='net1_layer2_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer1_common) )
        #     #net2 add
        #     self.layer2_common.append( self.netweights_common[cellidx][1] * GraphConvolution(name='net2_layer2_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs2[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer1_common) )
        #     #net3 ant
        #     self.layer2_common.append( self.netweights_common[cellidx][2] * GraphConvolution(name='net3_layer2_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs3[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer1_common) )
        # self.layer2_common.append( tf.matmul(self.layer1_common, self.loop_weights_common1[cellidx])) 
        # self.layer2_common = tf.add_n(self.layer2_common)
        # self.layer2_common = tf.nn.relu( tf.nn.l2_normalize(self.layer2_common, dim=0) )

        # #layer3
        # self.layer3_common = []
        # for cellidx in range(self.cellscount):
        #     #net1 syn
        #     self.layer3_common.append( self.netweights_common[cellidx][0] * GraphConvolution(name='net1_layer3_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer2_common) )
        #     #net2 add
        #     self.layer3_common.append( self.netweights_common[cellidx][1] * GraphConvolution(name='net2_layer3_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs2[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer2_common) )
        #     #net3 ant
        #     self.layer3_common.append( self.netweights_common[cellidx][2] * GraphConvolution(name='net3_layer3_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs3[cellidx],dropout=self.dropout,act=lambda x:x)(self.layer2_common) )
        # self.layer3_common.append( tf.matmul(self.layer2_common, self.loop_weights_common2[cellidx])) 
        # self.layer3_common = tf.add_n(self.layer3_common)
        # self.layer3_common = tf.nn.relu( tf.nn.l2_normalize(self.layer3_common, dim=0) )

        # #merge 3 layers
        # self.embeddings_common = tf.nn.l2_normalize((self.layerweights_common[0] * self.layer1_common + self.layerweights_common[1] * self.layer2_common + self.layerweights_common[2] * self.layer3_common), dim=0)

        # self.reconstructions_common = {}
        # for cellidx in range(self.cellscount):
        #     self.reconstructions_common_net1 = DEDICOMDecoder(name='common_decoder_net1', weights_global=self.global_weights_common, weights_local=self.local_weights_common0[cellidx], act=tf.nn.sigmoid)(self.embeddings_common)
        #     self.reconstructions_common_net2 = DEDICOMDecoder(name='common_decoder_net2', weights_global=self.global_weights_common, weights_local=self.local_weights_common1[cellidx], act=tf.nn.sigmoid)(self.embeddings_common)
        #     self.reconstructions_common_net3 = DEDICOMDecoder(name='common_decoder_net3', weights_global=self.global_weights_common, weights_local=self.local_weights_common2[cellidx], act=tf.nn.sigmoid)(self.embeddings_common)

        #     self.reconstructions_common[cellidx] = [self.reconstructions_common_net1, self.reconstructions_common_net2, self.reconstructions_common_net3]

