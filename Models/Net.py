import os, copy, json
import numpy as np
import tensorflow as tf

class Net:

    def __init__(self, config):
        ## Method to build

        # Copying configuration to avoid mutation
        self.config = copy.deepcopy(config)

        # Creating none data to hold data
        self.data = None

        # Random Seed
        self.random_seed = self.config['random_seed']

        # File read and write locations
        self.result_dir = self.config['result_dir']
        self.data_dir   = self.config['data_dir']
        self.report_dir = self.config['report_dir']

        # Hyperparameters
        self.hyperparameters = self.config['hyperparameters']

        # Building Graphs
        self.graph = self.build_graph(tf.Graph())

        # Saving net
        if 'max_kept' in self.config:
            max_kept = self.config['max_kept']
        else:
            max_kept = 10

        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep = max_kept
                )

        ## Filewriter for obtaining Tensorboard graph
        self.sess = tf.Session(graph=self.graph)
        self.sw = tf.summary.FileWriter(self.report_dir, self.sess.graph)

    ## Functions that need to be overriden in subclasses
    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden')

    @classmethod
    def get_default_configs(cls):
        raise Exception('The get_default_configs function must be overriden')

    def learn(self, hyperparameters={}):
        raise Exception('The learn function must be overriden')

    ## Training per epoch
    def train(self, save_every=1, eval_every=1, hyperparameters={}):

        if(self.Data == None):
            raise Exception('No Data loaded into model')
            return
        tf.set_random_seed(self.random_seed)
        print('Initializing Variables with seed =', self.random_seed)
        self.sess.run(self.init_graph)

        ##Using max_epochs from tuned hyperparamenters
        if ('max_epochs' in hyperparameters.keys()):
            self.max_epochs = hyperparameters['max_epochs']

        for epoch in range(0, self.max_epochs):
            print('Starting Epoch %d of %d' %
                    (epoch + 1, self.max_epochs))
            self.learn(hyperparameters)

            # Save Model
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self.save(epoch)

            # Evaluate model
            if eval_every > 0 and (epoch + 1) % eval_every == 0:
                self.evaluate()

        if eval_every > 0:
            self.evaluate()

    ## Saving model
    def save(self, epoch):
        # Get number of batches seen
        global_step = tf.train.global_step(self.sess, self.global_step)

        print('Saving to %s with global_step %d...' % (self.result_dir, global_step), end='')
        save_path = self.saver.save(self.sess,
                os.path.join(self.result_dir,'Net_epoch_'+ str(epoch)),
                global_step = global_step)
        print('Done!')

        # Saving config file
        if not os.path.isfile(os.path.join(self.result_dir, 'config.json')):
            config = self.config
            with open(os.path.join(self.result_dir, 'config.josn'), 'w') as f:
                    json.dump(config, f)

    ##Function to be overriden in subclass
    def evaluate(self):
        raise Exception('The evaluate function must be overidden')
