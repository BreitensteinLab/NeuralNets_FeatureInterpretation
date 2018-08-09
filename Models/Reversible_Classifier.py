from .Net import Net
import numpy as np
import tensorflow as tf
import os

class Classifier(Net):

    def __init__(self, config, data, frac):
        #Tensorflow Access Pointers
        self.outputs = []
        self.G_outputs = []
        self.dense_layers = []
        self.G_dense_layers = []

        super().__init__(config)

        # Split data into training and validation
        self.Data = data
        data = self.Data.splitData(frac, self.random_seed)
        self.Training_data = data[0]
        self.Validation_data = data[1]

        # Upsample imbalanced data
        self.Training_data.balanceData(seed = 123)
        self.Validation_data.balanceData(seed = 123)

        # Get a test set for testing
        self.Test_data = None
        if(len(frac) >= 2):
            self.Test_data = data[2]

    ## Build graph
    def build_graph(self, graph):

        # Get Graph properties
        g = graph

        specifications = self.config['graph']

        layers = specifications['layers']

        N = len(specifications['layers'])

        nonlinearities = specifications['nonlinearities']

        # Get scope names
        scope_names = ['input_layer']
        for idx in range(1, N-2):
            scope_names.append('hidden_layer_' + str(idx))
        scope_names.append('output_layer')

        with tf.variable_scope('Classifier') as scope:
            classifier_scope = scope.name + '/'

        ## Build classify function
        self.classify, self.dense_layers = Classifier.build_network( g,
            layers,
            nonlinearities,
            scope_names,
            scope)

        with g.as_default():
            tf.set_random_seed(self.random_seed)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            # Get inputs and labels as class variables
            self.input = tf.placeholder(tf.float32, shape=(None, layers[0]), name='input')
            self.labels = tf.placeholder(tf.bool, shape=(None, layers[-1]), name='labels')
            logits, self.outputs = self.classify(self.input)


        # Get the output of a layer a certain amount in front of the output layer
        self.hidden_output = self.outputs[ - 1 - specifications['depth']]


        # Build Inverse Layers
        self.inverse_layers = []
        self.inverse_output = []

        with g.as_default():
            last_layer = self.hidden_output
            with tf.variable_scope('inverse'):
                for idx in reversed(range(0, N - 1 - specifications['depth'])):
                    with tf.variable_scope(scope_names[idx] + '_inverse'):

                        wb = self.dense_layers[idx]
                        last_layer = Classifier.inverse_relu(last_layer)

                        pinv = Classifier.pinv(wb['weight'])
                        last_layer = tf.subtract(last_layer, wb['bias'])
                        last_layer = tf.matmul(last_layer, pinv)

                        self.inverse_layers.append(pinv)
                        self.inverse_output.append(last_layer)

            self.inverted_hidden_layer = last_layer


        # Generator Specifications
        G_input_dim = layers[N - 1 - specifications['depth']]
        layers = [G_input_dim] + specifications['G_layers'] + [layers[0]]
        nonlinearities = specifications['G_nonlinearities']
        N = len(layers)

        # Get scope names
        scope_names = ['input_layer']
        for idx in range(1, N-2):
            scope_names.append('hidden_layer_' + str(idx))
        scope_names.append('output_layer')

        with tf.variable_scope('Generator') as scope:
            generator_scope = scope.name + '/'

        self.generator, _ = Classifier.build_network( g,
            layers,
            nonlinearities,
            scope_names,
            scope)

        self.input_like, _ = self.generator(self.hidden_output)

        self.input_like_classification, _ = self.classify(self.input_like)

        with g.as_default():
            with tf.variable_scope('ops'):
                # Other Operations
                self.pred = tf.nn.softmax(logits, name='softmax')
                tf.summary.scalar('pred', self.pred)
                self.Y = tf.placeholder(tf.float32, shape=(None, logits.shape[1]), name='Y')

                self.Y_pred = tf.placeholder(tf.float32,
                    shape=(None, logits.shape[1]), name='Y_pred')
                self.confusion = tf.confusion_matrix(
                    tf.argmax(self.Y, 1), tf.argmax(self.Y_pred, 1))
            self.define_ops(g)
            self.define_G_ops(g)
            self.init_graph = tf.global_variables_initializer()
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='Generator')
            self.G_init = tf.variables_initializer(var_list, name='Generator_init')
        return g

    ## Optimization component for Classifier
    def define_ops(self, g):
        with tf.variable_scope('D_loss', reuse=tf.AUTO_REUSE):
            # Method to define ops
            self.unregularized_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = self.labels,
                logits = self.outputs[-1]))
            tf.summary.scalar('Discriminator_loss_unregularized', self.unregularized_loss)

            # Geting regularizer
            # L2 regularizer is used
            all_weights = []
            for layer in self.dense_layers:
                # Weights required for regularizer
                all_weights.append(tf.nn.l2_loss(layer['weight']))

            self.l2_loss = tf.add_n(all_weights)
            tf.summary.scalar('Discriminator_regularization', self.unregularized_loss)

            # Define Optimizer hyperparameters
            hp = self.hyperparameters['Classifier']
            lr = tf.placeholder_with_default(hp['lr'], [], name='Learning_rate')
            beta1 = tf.placeholder_with_default(hp['beta1'], [], name='Beta1')
            beta2 = tf.placeholder_with_default(hp['beta2'], [], name='Beta2')
            epsilon = tf.placeholder_with_default(hp['epsilon'], [], name='Epsilon')
            l2_penalty = tf.placeholder_with_default(hp['l2'], [], name='l2_penalty')
            self.max_epochs = hp['max_epochs']

            # AdamOptimizer for optimization
            self.D_optimizer = tf.train.AdamOptimizer(
                lr,
                beta1,
                beta2,
                epsilon
                )

           # Define a dictionary of hyperparameters
            self.D_hp = {
                    'lr': lr,
                    'beta1': beta1,
                    'beta2': beta2,
                    'epsilon': epsilon,
                    'l2_penalty': l2_penalty}

            ## Loss with regularization
            self.D_loss = self.unregularized_loss + l2_penalty * self.l2_loss

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='Classifier')

            # Minimize the regularized loss
            self.D_train_op = self.D_optimizer.minimize(self.D_loss,
                global_step = self.global_step,
                var_list=var_list)

        return

    ## Optimization component for generator
    def define_G_ops(self, g):
        with tf.variable_scope('G_loss', reuse=tf.AUTO_REUSE):
            # Loss Operations
            self.input_loss = tf.losses.mean_squared_error(self.input, self.input_like)
            self.discriminator_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = self.labels,
                logits = self.input_like_classification))
        # Softmax activation layer to get probabilities
        # Method to define ops

           # Define Optimizer hyperparameters
            hp = self.hyperparameters['Generator']
            alpha = tf.placeholder_with_default(hp['alpha'], [], name='Alpha')
            lr = tf.placeholder_with_default(hp['lr'], [], name='Learning_rate')
            beta1 = tf.placeholder_with_default(hp['beta1'], [], name='Beta1')
            beta2 = tf.placeholder_with_default(hp['beta2'], [], name='Beta2')
            epsilon = tf.placeholder_with_default(hp['epsilon'], [], name='Epsilon')
            self.max_epochs = hp['max_epochs']

            # Define a dictionary of hyperparameters
            self.G_hp = {
                    'alpha': alpha,
                    'lr': lr,
                    'beta1': beta1,
                    'beta2': beta2,
                    'epsilon': epsilon}

            # Define overall generator loss
            self.G_loss = self.input_loss + alpha * self.discriminator_loss
            tf.summary.scalar('Generator_Loss', self.G_loss)

            #Adamoptimizer for optimization
            self.G_optimizer = tf.train.AdamOptimizer(
                lr,
                beta1,
                beta2,
                epsilon )

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='Generator')

            # Generator loss is minimized
            self.G_train_op = self.G_optimizer.minimize(self.G_loss,
                global_step = self.global_step,
                var_list = var_list)

        return

    ## Create and intializes weight and bias matrices
    @classmethod
    def create_weight_bias(cls, N_in, N_out, weight_name = 'weight', bias_name = 'bias'):
        w = tf.get_variable(
            shape=[N_in, N_out],
            name=weight_name,
            initializer=tf.contrib.layers.xavier_initializer()
            )

        b = tf.get_variable(
            shape = [N_out],
            name = bias_name,
            initializer = tf.glorot_uniform_initializer()
            )
        return w, b


    ## Define default values for parameters and hyperparameters
    @classmethod
    def get_default_configs(cls, N_inputs, N_outputs):
        config = {
                'random_seed': 123123123,
                'result_dir' : os.path.join('.','results'),
                'data_dir'   : os.path.join('.','Data'),
                'report_dir' : os.path.join('.','summaries'),
                'hyperparameters':
                    {'Classifier': {
                        'max_epochs'    : 30,
                        'lr'            : 3e-4,
                        'beta1'         : 0.9,
                        'beta2'         : 0.999,
                        'epsilon'       : 1e-08,
                        'l2'            : 0.
                        },
                     'Generator' : {
                        'max_epochs' : 30,
                        'lr'         : 3e-4,
                        'beta1'      : 0.9,
                        'beta2'      : 0.999,
                        'epsilon'    : 1e-08,
                        'alpha'      : 1.
                    }},
                'graph':
                    {'layers'   : [N_inputs, 512, 256, 128, N_outputs],
                    'nonlinearities':['lrelu', 'lrelu', 'lrelu', 'none'],
                    'G_layers' : [128, 256, 512],
                    'G_nonlinearities':['lrelu', 'lrelu', 'lrelu', 'relu'],
                    'depth': 1
                    },
                  }
        return config

    ## Build the neural network
    @classmethod
    def build_network(cls,g, layers, nonlinearities, scope_names, net_name):

        # Specify the slope of leaky relu activation to be used
        def lrelu(x, name=''):
            return tf.nn.leaky_relu(x, alpha=0.2, name=name)
        # Define Activatio Functions
        activation_functions = {
                'relu':tf.nn.relu,
                'sigmoid':tf.nn.sigmoid,
                'lrelu': lrelu}

        # Determe parameters of model
        N = len(layers)
        num_of_inputs = layers[0]
        num_of_classes = layers[-1]

        dense_layers = []

        # Build Network
        with tf.variable_scope(net_name):
            net = tf.get_variable_scope().name + '/'
            #print(net)

        # Define classifier neural network
        def network(X):
            last_layer = X
            outputs=[]
            # Build network
            with g.as_default():
                for idx in range(1, N):
                    with tf.variable_scope(net + scope_names[idx-1] , reuse=tf.AUTO_REUSE):
                        w, b = Classifier.create_weight_bias(layers[idx-1], layers[idx])
                        dense_layers.append({'weight':w, 'bias':b})

                        # Get weights and biases
                        w = dense_layers[idx-1]['weight']
                        b = dense_layers[idx-1]['bias']
                    with tf.variable_scope(net + scope_names[idx-1] + '/', reuse=tf.AUTO_REUSE):
                        pre_activation = tf.add(tf.matmul(last_layer, w), b)

                        if (nonlinearities[idx-1] in activation_functions):
                            activation = activation_functions[nonlinearities[idx-1]]
                            last_layer = activation(pre_activation,
                            name=nonlinearities[idx-1])
                        else:
                            last_layer = pre_activation

                        outputs.append(last_layer)

            return last_layer, outputs
        return network, dense_layers

    ## Calculate the pseudoinverse of A
    def pinv(A, tol=1e-15):
        S, U, V = tf.svd(A)

        threshold = tol * tf.reduce_max(S)
        zero_removed = tf.greater(S, threshold)

        reciprocal = tf.where(zero_removed, tf.reciprocal(S), tf.zeros(S.shape))
        return(tf.matmul(tf.matmul(V, tf.diag(reciprocal)), U, transpose_b = True))

    ## Inverse_relu used for network inversion
    def inverse_relu(A):
        return tf.nn.relu(A) - 5 * tf.nn.relu(-A)

    ## Function to get tensors for classifier training
    def learn(self, hyperparameters={}):

        # Update hyperparameters of the model
        updated_hyperparameters = Classifier.zipDicts(self.D_hp, hyperparameters)

        # Initialize the training loss
        self.epoch_loss = 0

        # Specify number of iterations for training
        iters = self.Training_data.Num_of_batches
        for i in range(0, iters):
            batch = self.Training_data.getBatch(i)

            # X = input data
            batch_x = batch['X']

            # Y = input labels
            batch_y = batch['Y']

            feed_dict = {self.input: batch_x, self.labels: batch_y}
            feed_dict.update(updated_hyperparameters)

           # Train the classifier and calculates loss
            _, c = self.sess.run([self.D_train_op, self.D_loss],
                    feed_dict = feed_dict
                )
            self.epoch_loss = self.epoch_loss + c

        # Prints the training loss for each epoch
        print('\tEpoch Completed.  Training loss: %10.5f' % (self.epoch_loss))
        return

    ## Function for model evaluation with validation data
    def evaluate(self):

        print('Evaluating model performance with validation data')

        # Initializing validation loss
        self.val_loss = 0

        # prediction made by the model on the validation data
        val_pred = []
        # correct label
        correct_Y = []
        # Specify number of iterations
        iters = self.Validation_data.Num_of_batches
        for i in range(0, iters):
            batch = self.Validation_data.getBatch(i)
            # X = input validation data
            val_x = batch['X']
            # Y = input validation labels
            val_y = batch['Y']

            # Calculate loss and prediction on the validation data
            c, pred = self.sess.run([self.D_loss, self.pred],
            feed_dict = {self.input:val_x, self.labels: val_y}
            )

            self.val_loss = self.val_loss + c
            val_pred = val_pred + [pred]
            correct_Y = correct_Y + [val_y]

        self.val_pred = np.concatenate(val_pred, axis=0)
        Y = np.concatenate(correct_Y, axis=0)

        self.checking = Y

        # Evaluate the validation confusion matrix
        con_mat = self.confusion.eval(session=self.sess,
                feed_dict = {self.Y: Y, self.Y_pred: self.val_pred})
        print('\t Validation Loss: %10.5f' % (self.val_loss))
        print('Confusion Matrix\n\t', str(con_mat).replace('\n', '\n\t'))

        return

    ## Function for generator training
    def train_generator(self, save_every = 1, eval_every = 1, hyperparameters={}):

        self.sess.run(self.G_init)

        # Specify the max epochs hyperparameter
        if ('max_epochs' in hyperparameters.keys()):
            self.hyperparameters['Generator']['max_epochs'] = hyperparameters['max_epochs']
        max_epochs = self.hyperparameters['Generator']['max_epochs']


        for epoch in range(0, max_epochs):
            print('starting Epoch %d of %d' %
                    (epoch + 1, max_epochs))
            self.G_learn(hyperparameters)

            # Save Model
            if save_every > 0 and epoch % save_every == 0:
                self.save(epoch)

            # Evaluate model
            if eval_every > 0 and epoch % eval_every == 0:
                self.G_evaluate()


        if eval_every > 0:
            self.G_evaluate()

    ## Function to get tensors for generator training
    def G_learn(self, hyperparameters={}):

        # Update hyperparameters
        updated_hyperparameters = Classifier.zipDicts(self.G_hp, hyperparameters)

        # Initialize training loss
        self.epoch_loss = 0
        # Specify number of iterations
        iters = self.Training_data.Num_of_batches
        for i in range(0, iters):
            batch = self.Training_data.getBatch(i)
            # X = input data for training generator
            batch_x = batch['X']
            # Y = input labels for training generator
            batch_y = batch['Y']

            feed_dict = {self.input: batch_x, self.labels: batch_y}
            feed_dict.update(updated_hyperparameters)

            # Feed input tensors to the model
            # Train the generator and calculate the generator loss
            _, c = self.sess.run([self.G_train_op, self.G_loss],
                feed_dict = feed_dict
                )
            self.epoch_loss = self.epoch_loss + c

        # Evaluate and print the training loss for the generator
        self.epoch_loss = self.epoch_loss / self.Training_data.Num_of_samples
        print('\tEpoch Completed.  Training loss: %10.5f' % (self.epoch_loss))
        return

    ## Function to evaluate generator performance using validation data
    def G_evaluate(self):

        print('Evaluating model performance with validation data')

        # initialize validation loss
        self.val_loss = 0

        # correct label
        correct_Y = []
        # prediction made by the generator on the validation data
        val_pred = []
        # specify number of iterations
        iters = self.Validation_data.Num_of_batches

        for i in range(0, iters):
            #batch = self.Validation_data.getBatch(i)
            batch = self.Validation_data.getBatch(i)
            # X = input validation data
            val_x = batch['X']
            # Y = input validation labels
            val_y = batch['Y']

            # Evaluate generator loss
            c, pred = self.sess.run([self.G_loss, self.input_like_classification],
            feed_dict = {self.input:val_x, self.labels: val_y}
            )

            # Calculate validation loss after each iteration
            self.val_loss = self.val_loss + c
            val_pred = val_pred + [pred]
            correct_Y = correct_Y + [val_y]


        # Compare the predicted labels with true labels
        self.val_pred = np.concatenate(val_pred, axis=0)
        Y = np.concatenate(correct_Y, axis=0)

        self.checking = Y
        con_mat = self.confusion.eval(session=self.sess,
                feed_dict = {self.Y: Y, self.Y_pred: self.val_pred})

        # Evaluate and print the validation loss
        self.val_loss = self.val_loss / self.Validation_data.Num_of_samples

        print('\t Validation Loss: %10.5f' % (self.val_loss))

        # Print the confusion matrix
        print('Confusion Matrix\n\t', str(con_mat).replace('\n', '\n\t'))

        return

    ## Function to concatenate two dictionaries
    @staticmethod
    def zipDicts(dict1, dict2):
        return_dict = {}
        for key in set(dict1).intersection(dict2):
            return_dict[dict1[key]] = dict2[key]
        return return_dict
