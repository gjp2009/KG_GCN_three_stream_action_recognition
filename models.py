from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class Model_dense(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

        # calculate gradient with respect to input, only for dense model
        self.grads = tf.gradients(self.loss, self.inputs)[0]  # does not work on sparse vector

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN_dense_mse(Model_dense):   # 6层网络的对比
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_dense_mse, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                   self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                      self.placeholders['labels_mask'])

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden4,
                                            output_dim=FLAGS.hidden5,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden5,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return self.outputs


class Model_dense_2s(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers_att = []
        self.activations_att = []
        self.layers_all = []
        self.activations_all = []
        self.layers_node = []
        self.activations_node = []

        # self.classifier = tf.Variable(tf.truncated_normal([101, 300]))

        self.inputs_att = None
        self.outputs_att = None
        self.inputs_all = None
        self.outputs_all = None
        self.inputs_node = None
        self.outputs_node = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model for two streams 为两个流构建顺序层模型
        self.activations_att.append(self.inputs_att)
        for layer in self.layers_att:
            hidden = layer(self.activations_att[-1])
            self.activations_att.append(hidden)
        self.outputs_att = self.activations_att[-1]

        self.activations_all.append(self.inputs_all)
        for layer in self.layers_all:
            hidden = layer(self.activations_all[-1])
            self.activations_all.append(hidden)
        self.outputs_all = self.activations_all[-1]

        self.activations_node.append(self.inputs_node)
        for layer in self.layers_node:
            hidden = layer(self.activations_node[-1])
            self.activations_node.append(hidden)
        self.outputs_node = self.activations_node[-1]

        # Store model variables for easy access  存储模型变量以方便访问
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)   # 函数可以用来获取GLOBAL_VARIABLES 集合中的所有元素，返回一个列表
        self.vars = {var.name: var for var in variables}

        # Build metrics指标
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

        # calculate gradient with respect to input, only for dense model
        self.grads = tf.gradients(self.loss, [self.inputs_att, self.inputs_all])[0]  # does not work on sparse vector

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None, save_path='tmp'):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, save_path+"/%s_%s_%s_%s_%s.ckpt" % (self.name, FLAGS.use_little, FLAGS.w2v_type, str(FLAGS.num_class), str(FLAGS.split_ind)))
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None, save_path='tmp'):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = save_path+"/%s_%s_%s_%s_%s.ckpt" % (self.name, FLAGS.use_little, FLAGS.w2v_type, str(FLAGS.num_class), str(FLAGS.split_ind))
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def attention(self, x, ch, sn=True, scope='attention', reuse=False):    # x句柄，ch为1588
        with tf.variable_scope(scope, reuse=reuse):     # 用于定义创建变量(或层)的操作
            f = conv(x, ch, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']  ch // 2
            g = conv(x, ch, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']  ch // 2
            h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

            # N = h * w，两个矩阵g和f相乘
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N]，hw_flatten 为转换数据格式，f(Vt)*g(Vs)

            beta = tf.nn.softmax(s, dim=-1)  # attention map，每一片的行累加和为1，exp(x1)/Σj exp(xi)

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]，softmax(g*f)*h
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))  # 创建变量gamma

            o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C], o变成x.shape,就是同样的shape了
            x = gamma * o + x       # x为 V^s = γΣαs,t h(Vt) + Vs

        return x


class GCN_dense_mse_2s_ini(Model_dense_2s):
    def __init__(self, placeholders, lookup_table_act_att, input_dim, **kwargs):
        super(GCN_dense_mse_2s_ini, self).__init__(**kwargs)

        self.inputs_all = placeholders['features_all']
        if FLAGS.use_self_attention:
            # self.label_num = placeholders['label_num']
            x = placeholders['features_att']
            x = self.attention(x, FLAGS.num_class)
            x = tf.squeeze(x)
            x = tf.reduce_sum(x, axis=1)        # (48,1588)
            x = tf.expand_dims(x, 2)            # (48,1588,1)
            x = tf.tile(x,[1,1,FLAGS.w2v_dim])  # (48,1588,500)
            att_feature = tf.slice(self.inputs_all, [tf.cast(FLAGS.label_num,tf.int32), 0], [-1, -1])   # (1689,500)剪切(101,0) (1588,500)
            att_feature = tf.expand_dims(att_feature,0)     # (1,1588,500)
            att_feature = tf.tile(att_feature,[FLAGS.batch_size,1,1])   # (48,1588,500)
            att_feature = tf.multiply(att_feature,x)
            self.inputs_att = tf.reshape(att_feature,[-1,FLAGS.w2v_dim])
        else:
            self.inputs_att = placeholders['features_att']

        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        # self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.output_dim = FLAGS.output_dim
        self.placeholders = placeholders
        self.lookup_table_act_att = lookup_table_act_att

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers_all)):
            for var in self.layers_all[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for i in range(len(self.layers_att)):
            for var in self.layers_att[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Masked Classification error
        if FLAGS.use_softmax:
            loss_tmp, self.classifier, self.attend_feat = mask_classification_softmax_loss(self.outputs_all,
                                                                                           self.outputs_att,
                                                                                           self.placeholders[
                                                                                               'train_mask'],
                                                                                           self.lookup_table_act_att,
                                                                                           self.placeholders[
                                                                                               'labels'])
        else:
            loss_tmp, self.classifier, self.attend_feat = mask_classification_loss(self.outputs_all,
                                                                                   self.outputs_att,
                                                                                   self.placeholders[
                                                                                       'train_mask'],
                                                                                   self.lookup_table_act_att,
                                                                                   self.placeholders['labels'])
        self.loss += loss_tmp

    def _accuracy(self):
        if FLAGS.use_softmax:
            self.accuracy = mask_classification_softmax_accuracy(self.outputs_all, self.outputs_att,
                                                                 self.placeholders['train_mask'],
                                                                 self.lookup_table_act_att,
                                                                 self.placeholders['labels'])
        else:
            self.accuracy = mask_classification_accuracy(self.outputs_all, self.outputs_att,
                                                         self.placeholders['train_mask'],
                                                         self.lookup_table_act_att, self.placeholders['labels'])

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
###########For the attribute nodes stream###############################################################
        self.layers_att.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))

        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))

        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))

        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))

        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden4,
                                            output_dim=FLAGS.hidden5,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))

        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden5,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging,
                                            type='att'))


###########For the all nodes stream###############################################################
        self.layers_all.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))

        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))

        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))

        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))

        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden4,
                                            output_dim=FLAGS.hidden5,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))

        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden5,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging,
                                            type='all'))

    def predict(self):
        return self.outputs


class GCN_dense_mse_2s_little(Model_dense_2s):
    def __init__(self, placeholders, lookup_table_act_att, input_dim, **kwargs):
        super(GCN_dense_mse_2s_little, self).__init__(**kwargs)
        self.inputs_all = placeholders['features_all']
        if FLAGS.use_self_attention:
            # self.label_num = placeholders['label_num']
            x = placeholders['features_att']
            x = self.attention(x, FLAGS.num_class)
            x = tf.squeeze(x)
            x = tf.reduce_sum(x, axis=1)
            x = tf.expand_dims(x, 2)
            x = tf.tile(x,[1,1,FLAGS.w2v_dim])
            att_feature = tf.slice(self.inputs_all, [tf.cast(FLAGS.label_num,tf.int32), 0], [-1, -1])
            att_feature = tf.expand_dims(att_feature,0)
            att_feature = tf.tile(att_feature,[FLAGS.batch_size,1,1])
            att_feature = tf.multiply(att_feature,x)
            self.inputs_att = tf.reshape(att_feature,[-1,FLAGS.w2v_dim])
        else:
            self.inputs_att = placeholders['features_att']

        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        # self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.output_dim = FLAGS.output_dim
        self.placeholders = placeholders
        self.lookup_table_act_att = lookup_table_act_att

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers_all)):
            for var in self.layers_all[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for i in range(len(self.layers_att)):
            for var in self.layers_att[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Masked Classification error
        if FLAGS.use_softmax:
            loss_tmp, self.classifier, self.attend_feat = mask_classification_softmax_loss(self.outputs_all,
                                                                                           self.outputs_att,
                                                                                           self.placeholders['train_mask'],
                                                                                           self.lookup_table_act_att,
                                                                                           self.placeholders['labels'],
                                                                                           self.placeholders['transductive'])
        else:
            loss_tmp, self.classifier, self.attend_feat = mask_classification_loss(self.outputs_all,
                                                                                   self.outputs_att,
                                                                                   self.placeholders[
                                                                                       'train_mask'],
                                                                                   self.lookup_table_act_att,
                                                                                   self.placeholders['labels'])
        self.loss += loss_tmp

    def _accuracy(self):
        if FLAGS.use_softmax:
            self.accuracy = mask_classification_softmax_accuracy(self.outputs_all, self.outputs_att,
                                                                 self.placeholders['train_mask'],
                                                                 self.lookup_table_act_att,
                                                                 self.placeholders['labels'])
        else:
            self.accuracy = mask_classification_accuracy(self.outputs_all, self.outputs_att,
                                                         self.placeholders['train_mask'],
                                                         self.lookup_table_act_att, self.placeholders['labels'])

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
###########For the attribute nodes stream###############################################################
        self.layers_att.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))


        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging,
                                            type='att'))


###########For the all nodes stream###############################################################
        self.layers_all.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))


        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging,
                                            type='all'))


###########For the delaonei stream###############################################################
        self.layers_node.append(GraphConvolution(input_dim=self.input_dim,
                                                 output_dim=FLAGS.hidden1,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: tf.maximum(x, 0.2 * x),
                                                 dropout=False,
                                                 sparse_inputs=False,
                                                 logging=self.logging,
                                                 type='node'))

        self.layers_node.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                 output_dim=FLAGS.hidden2,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: tf.maximum(x, 0.2 * x),
                                                 dropout=False,
                                                 sparse_inputs=False,
                                                 logging=self.logging,
                                                 type='node'))

        self.layers_node.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                                 output_dim=FLAGS.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: tf.maximum(x, 0.2 * x),
                                                 dropout=False,
                                                 sparse_inputs=False,
                                                 logging=self.logging,
                                                 type='node'))


def predict(self):
        return self.outputs


class GCN_dense_mse_2s(Model_dense_2s):     # placeholders字典，lookup_table_act_att稀疏张量(101,50)，input_dim=500，True
    def __init__(self, placeholders, lookup_table_act_att, input_dim, **kwargs):
        super(GCN_dense_mse_2s, self).__init__(**kwargs)    # 继承类，覆盖初始化化def init，并继承初始化属性name，可以调用

        self.inputs_all = placeholders['features_all']  # 先初始化，(1689,500)词向量
        if FLAGS.use_self_attention:
            # self.label_num = placeholders['label_num']
            x = placeholders['features_att']        # 返回一个可以用作提供值的句柄的 SparseTensor，x添加值
            x = self.attention(x, FLAGS.num_class)  # 调用父类的attention方法，x句柄和 num_class为1588，加上attention
            x = tf.squeeze(x)               # 降维，删除维度为1的部分，（48,16,1,1588）变成（48,16,1588）
            x = tf.reduce_sum(x, axis=1)    # 计算张量tensor沿着1维度行的和 -> （48，1588）
            x = tf.expand_dims(x, 2)        # 添加维度，(48,1588,1)
            x = tf.tile(x, [1, 1, FLAGS.w2v_dim])      # 翻倍，[1,1,500],变成(48,1588,500)
            att_feature = tf.slice(self.inputs_all, [tf.cast(FLAGS.label_num,tf.int32), 0], [-1, -1])   # 切分( (1689,500),[101,0],[-1,-1])，成(1588,500)
            att_feature = tf.expand_dims(att_feature, 0)    # (1,1588,500)
            att_feature = tf.tile(att_feature, [FLAGS.batch_size,1,1])  # 翻倍,(48,1588,500),48为batch_size
            att_feature = tf.multiply(att_feature, x)    # 这里应该只是把  Xins o = V^s * So
            self.inputs_att = tf.reshape(att_feature, [-1, FLAGS.w2v_dim])    # -1表示行不定，500列以后，行就是1689，48(1689,500)
        else:
            self.inputs_att = placeholders['features_att']

        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        # self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.output_dim = FLAGS.output_dim
        self.placeholders = placeholders
        self.lookup_table_act_att = lookup_table_act_att

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss，得到q-n-i
        for i in range(len(self.layers_all)):
            for var in self.layers_all[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for i in range(len(self.layers_att)):
            for var in self.layers_att[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Masked Classification error(隐藏分类错误) 应该是softmax求p-n-i,再得到交叉熵损失函数
        if FLAGS.use_softmax:
            loss_tmp, self.classifier, self.attend_feat = mask_classification_softmax_loss(self.outputs_all,
                                                                                           self.outputs_att,
                                                                                           self.placeholders['train_mask'],
                                                                                           self.lookup_table_act_att,
                                                                                           self.placeholders['labels']
                                                                                          )
        else:
            loss_tmp, self.classifier, self.attend_feat = mask_classification_loss(self.outputs_all,
                                                                                   self.outputs_att,
                                                                                   self.placeholders[
                                                                                       'train_mask'],
                                                                                   self.lookup_table_act_att)
        self.loss += loss_tmp

    def _accuracy(self):
        if FLAGS.use_softmax:
            self.accuracy = mask_classification_softmax_accuracy(self.outputs_all, self.outputs_att,
                                                                 self.placeholders['train_mask'],
                                                                 self.lookup_table_act_att,
                                                                 self.placeholders['labels'])
        else:
            self.accuracy = mask_classification_accuracy(self.outputs_all, self.outputs_att,
                                                         self.placeholders['train_mask'],
                                                         self.lookup_table_act_att, self.placeholders['labels'])

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
###########For the attribute nodes stream###############################################################
        self.layers_att.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))

        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))

        self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='att'))
        # self.layers_att.append(GraphConvolution(input_dim=FLAGS.hidden3,
        #                                     output_dim=self.output_dim,
        #                                     placeholders=self.placeholders,
        #                                     act=lambda x: tf.nn.l2_normalize(x, dim=1),
        #                                     dropout=True,
        #                                     logging=self.logging,
        #                                     type='att'))



###########For the all nodes stream###############################################################
        self.layers_all.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))

        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))

        self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            type='all'))
        # self.layers_all.append(GraphConvolution(input_dim=FLAGS.hidden3,
        #                                     output_dim=self.output_dim,
        #                                     placeholders=self.placeholders,
        #                                     act=lambda x: tf.nn.l2_normalize(x, dim=1),
        #                                     dropout=True,
        #                                     logging=self.logging,
        #                                     type='all'))


###########For the delaonei stream###############################################################
        self.layers_node.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=lambda x: tf.maximum(x, 0.2 * x),
                                                dropout=False,
                                                sparse_inputs=False,
                                                logging=self.logging,
                                                type='node'))

        self.layers_node.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden2,
                                                placeholders=self.placeholders,
                                                act=lambda x: tf.maximum(x, 0.2 * x),
                                                dropout=False,
                                                sparse_inputs=False,
                                                logging=self.logging,
                                                type='node'))

        self.layers_node.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                                output_dim=FLAGS.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: tf.maximum(x, 0.2 * x),
                                                dropout=False,
                                                sparse_inputs=False,
                                                logging=self.logging,
                                                type='node'))


    def predict(self):
        return self.outputs
