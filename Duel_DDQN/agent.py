import tensorflow as tf
import numpy as np
from .experience_replay import Exp
import tensorflow.compat.v2 as tf2


# Evaluates behavior policy while improving target policy
class duel_DDQN_agent:
    """
    Args:
            - num_actions: number of actions possible
            - obs_size: size of the state/observation. size of image
            - nhidden: hidden nodes for network
            - epoch: variable that helps know when to do experience replay and training (through modulo)
            - epsilon: epsilon decay, exploration vs exploitation
            - gamma: TODO: var used in dueling? used as discount factor
            - learning_rate: for gradient descent network training
            - replace: can be 'soft' or 'hard'. different types of replacement.
            - polyak: var used in soft replacement formula. how much to update in soft replacement
            - tau_step: hard replacement var. Used to know when to do replacement
            - mem_size: max memory used for exp replay buffer
            - minibatch_size: minibatch for training size
    """

    def __init__(
        self,
        num_actions,
        obs_size,
        nhidden,
        epoch,
        epsilon,
        gamma,
        learning_rate,
        replace,
        polyak,
        tau_step,
        mem_size,
        minibatch_size,
        is_conv=False,
        img_size=None,
    ):

        super(duel_DDQN_agent, self).__init__()

        self.actions = range(num_actions)
        self.num_actions = num_actions
        self.obs_size = obs_size  # number of features
        self.nhidden = nhidden  # hidden nodes
        self.img_size = img_size  # (185, 200, 3)
        self.is_conv = is_conv

        # for epsilon decay & to decide when to start training
        # used in epsilon decay function for modulo to know when to decay
        self.epoch = epoch

        self.epsilon = epsilon  # for exploration
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate  # learning rate alpha

        # for params replacement
        self.replace = replace  # type of replacement
        self.polyak = polyak  # for soft replacement
        self.tau_step = tau_step  # for hard replacement
        self.learn_step = 0  # steps after learning # count of hard replacement

        # for Experience replay
        self.mem = Exp(self.obs_size, mem_size, img_size=img_size)  # memory that holds experiences

        self.minibatch_size = minibatch_size

        self.step = 0  # each step in a episode. Increment after taking 1 action.

        # for tensorflow ops
        self.built_graph()  # call function that builds tf graph and sets up network
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_replace_hard)

        self.cum_loss_per_episode = 0  # incremented loss for charting display

    # decay epsilon after each epoch
    def epsilon_decay(self):
        if self.step % self.epoch == 0:
            # TODO: make decay rate a var in __init__
            self.epsilon = max(0.01, self.epsilon * 0.95)

    # epsilon-greedy behaviour policy for action selection
    def act(self, s):
        # Get action either randomly or from network
        if np.random.random() < self.epsilon:
            i = np.random.randint(0, len(self.actions))
        else:
            if self.is_conv:
                input_state = np.reshape(s, (1, self.img_size[0], self.img_size[1], self.img_size[2]))
            else:
                input_state = np.reshape(s, (1, s.shape[0]))
            # get Q(s,a) from model network
            Q_val = self.sess.run(
                self.model_Q_val, feed_dict={self.s: input_state}
            )
            # get index of largest Q(s,a)
            i = np.argmax(Q_val)

        action = self.actions[i]

        self.step += 1
        self.epsilon_decay()

        return action

    def learn(self, s, a, r, done):
        # stores observation in memory as experience at each time step
        # self.mem is class object from experience_replay
        self.mem.store(s, a, r, done)
        # starts training a minibatch from experience after 1st epoch
        if self.step > self.epoch:
            self.replay()  # start training with experience replay

    def td_target(self, s_next, r, done, model_s_next_Q_val, target_Q_val):
        # This function does Bellman update and is used to calculate loss of network
        # select action with largest Q value from model network
        model_max_a = tf.argmax(model_s_next_Q_val, axis=1, output_type=tf.dtypes.int32)

        arr = tf.range(tf.shape(model_max_a)[0], dtype=tf.int32)  # create row indices
        indices = tf.stack([arr, model_max_a], axis=1)  # create 2D indices
        max_target_Q_val = tf.gather_nd(
            target_Q_val, indices
        )  # select minibatch actions from target network
        max_target_Q_val = tf.reshape(max_target_Q_val, (self.minibatch_size, 1))

        # if state = done, td_target = r
        # Bellman update
        td_target = (1.0 - tf.cast(done, tf.float32)) * tf.math.multiply(
            self.gamma, max_target_Q_val
        ) + r
        # exclude td_target in gradient computation
        td_target = tf.stop_gradient(td_target)

        return td_target

    # select Q(s,a) from actions using e-greedy as behaviour policy from model network
    def predicted_Q_val(self, a, model_Q_val):
        # create 1D tensor of length = number of rows in a
        arr = tf.range(tf.shape(a)[0], dtype=tf.int32)

        # stack by column to create indices for Q(s,a) selections based on a
        indices = tf.stack([arr, a], axis=1)

        # select Q(s,a) using indice from model_Q_val
        Q_val = tf.gather_nd(model_Q_val, indices)
        Q_val = tf.reshape(Q_val, (self.minibatch_size, 1))

        return Q_val

    # contruct neural network
    # For non image state representation
    def built_net_features(
        self, var_scope, w_init, b_init, features, num_hidden, num_output
    ):
        with tf.variable_scope(var_scope):
            feature_layer = tf.contrib.layers.fully_connected(
                features,
                num_hidden,
                activation_fn=tf.nn.relu,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            V = tf.contrib.layers.fully_connected(
                feature_layer,
                1,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            A = tf.contrib.layers.fully_connected(
                feature_layer,
                num_output,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            Q_val = V + (
                A - tf.reduce_mean(A, reduction_indices=1, keepdims=True)
            )  # refer to eqn 9 from the original paper
        return Q_val

    # Neural net with no feature layer
    # This is for using with Resnet/Inception
    def built_net(self, var_scope, w_init, b_init, features, num_hidden, num_output):
        with tf.variable_scope(var_scope):
            V = tf.contrib.layers.fully_connected(
                features,
                1,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            A = tf.contrib.layers.fully_connected(
                features,
                num_output,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            Q_val = V + (
                A - tf.reduce_mean(A, reduction_indices=1, keepdims=True)
            )  # refer to eqn 9 from the original paper
        return Q_val

    # Neural Net with 2 conv layer on top
    def built_conv_net(
        self, var_scope, w_init, b_init, features, num_hidden, num_output
    ):
        with tf.variable_scope(var_scope):
            # First conv
            conv1 = tf.contrib.layers.conv2d(
                features, 64, (5, 5), padding="same", stride=2
            )
            bn1 = tf.contrib.layers.batch_norm(conv1, activation=tf.nn.relu)
            # Second conv
            conv2 = tf.contrib.layers.conv2d(bn1, 64, (5, 5), padding="same", stride=2)
            bn2 = tf.contrib.layers.batch_norm(conv2, activation=tf.nn.relu)
            # Third conv
            conv3 = tf.contrib.layers.conv2d(bn2, 32, (5, 5), padding="same", stride=2)
            bn3 = tf.contrib.layers.batch_norm(conv3, activation=tf.nn.relu)
            # Flatten
            flatten = tf.contrib.layers.flatten(bn3)
            # Fully connected layer
            feature_layer = tf.contrib.layers.fully_connected(
                flatten,
                num_hidden,
                activation_fn=tf.nn.relu,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            # Get V and A
            V = tf.contrib.layers.fully_connected(
                features,
                1,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            A = tf.contrib.layers.fully_connected(
                features,
                num_output,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            Q_val = V + (
                A - tf.reduce_mean(A, reduction_indices=1, keepdims=True)
            )  # refer to eqn 9 from the original paper
        return Q_val

    # Super Basic CNN for testing
    def built_basic_conv_net(
        self, var_scope, w_init, b_init, features, num_hidden, num_output
    ):
        with tf.variable_scope(var_scope):
            # First conv
            conv1 = tf.contrib.layers.conv2d(
                features, 64, (5, 5), padding="same", stride=2, activation_fn=tf.nn.relu
            )
            # Second conv
            conv2 = tf.contrib.layers.conv2d(
                conv1, 32, (5, 5), padding="same", stride=2, activation_fn=tf.nn.relu
            )
            # Flatten
            flatten = tf.contrib.layers.flatten(conv2)
            # Fully connected layer
            feature_layer = tf.contrib.layers.fully_connected(
                flatten,
                num_hidden,
                activation_fn=tf.nn.relu,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            # Get V and A
            V = tf.contrib.layers.fully_connected(
                feature_layer,
                1,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            A = tf.contrib.layers.fully_connected(
                feature_layer,
                num_output,
                activation_fn=None,
                weights_initializer=w_init,
                biases_initializer=b_init,
            )
            Q_val = V + (
                A - tf.reduce_mean(A, reduction_indices=1, keepdims=True)
            )  # refer to eqn 9 from the original paper
        return Q_val

    # contruct CNN
    def preprocess_image_model(self):
        ## InceptionV3 pretrained on ImageNet:
        # pretrained_model = tf2.keras.applications.InceptionV3(input_shape=self.img_size,
        #                                            include_top=False,
        #                                            pooling='avg',
        #                                            weights='imagenet')
        ## ResNet50 pretrained on ImageNet:
        pretrained_model = tf2.keras.applications.ResNet50(
            input_shape=self.img_size,
            include_top=False,
            pooling="avg",
            weights="imagenet",
        )
        pretrained_model.trainable = False
        return pretrained_model

    # contruct tensorflow graph
    def built_graph(self):
        tf.reset_default_graph()

        # Initialize all variables
        if self.is_conv:
            self.s = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], self.img_size[2]], name="s")
            self.s_next = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], self.img_size[2]], name="s_next")
        else:
            self.s = tf.placeholder(tf.float32, [None, self.obs_size], name="s")
            self.s_next = tf.placeholder(tf.float32, [None, self.obs_size], name="s_next")

        self.a = tf.placeholder(
            tf.int32,
            [
                None,
            ],
            name="a",
        )
        self.r = tf.placeholder(tf.float32, [None, 1], name="r")
        self.done = tf.placeholder(tf.int32, [None, 1], name="done")
        self.model_s_next_Q_val = tf.placeholder(
            tf.float32, [None, self.num_actions], name="model_s_next_Q_val"
        )

        # weight, bias initialization
        w_init = tf.initializers.lecun_uniform()
        b_init = tf.initializers.he_uniform(1e-4)


        # If is_conv we train a conv model from scratch
        # Else we have a non image representation of the state: ResNet preprocessing
        if self.is_conv:
            self.model_Q_val = self.built_basic_conv_net(
                "model_net", w_init, b_init, self.s, self.nhidden, self.num_actions
            )
            self.target_Q_val = self.built_basic_conv_net(
                "target_net", w_init, b_init, self.s_next, self.nhidden, self.num_actions
            )
        else:
            self.model_Q_val = self.built_net(
                "model_net", w_init, b_init, self.s, self.nhidden, self.num_actions
            )
            self.target_Q_val = self.built_net(
                "target_net", w_init, b_init, self.s_next, self.nhidden, self.num_actions
            )

        with tf.variable_scope("td_target"):
            td_target = self.td_target(
                self.s_next,
                self.r,
                self.done,
                self.model_s_next_Q_val,
                self.target_Q_val,
            )
        with tf.variable_scope("predicted_Q_val"):
            predicted_Q_val = self.predicted_Q_val(self.a, self.model_Q_val)
        with tf.variable_scope("loss"):
            self.loss = tf.losses.huber_loss(td_target, predicted_Q_val)
        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate
            ).minimize(self.loss)

        # get network params
        with tf.variable_scope("params"):
            self.target_net_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net"
            )
            self.model_net_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="model_net"
            )

        # replace target net params with model net params
        with tf.variable_scope("hard_replace"):
            self.target_replace_hard = [
                t.assign(m)
                for t, m in zip(self.target_net_params, self.model_net_params)
            ]
        with tf.variable_scope("soft_replace"):
            self.target_replace_soft = [
                t.assign(self.polyak * m + (1 - self.polyak) * t)
                for t, m in zip(self.target_net_params, self.model_net_params)
            ]

    # decide soft or hard params replacement
    def replace_params(self):
        if self.replace == "soft":
            # Move weights partially: 0.9 target + 0.1 model
            # soft params replacement
            self.sess.run(self.target_replace_soft)
        else:
            # copy weight from trained to target every tau steps
            # hard params replacement
            if self.learn_step % self.tau_step == 0:
                self.sess.run(self.target_replace_hard)
            self.learn_step += 1

    def replay(self):
        # select minibatch of experiences from memory for training
        (s, a, r, s_next, done) = self.mem.minibatch(self.minibatch_size)

        # select actions from model network
        model_s_next_Q_val = self.sess.run(self.model_Q_val, feed_dict={self.s: s_next})

        # training
        _, loss = self.sess.run(
            [self.optimizer, self.loss],
            feed_dict={
                self.s: s,
                self.a: a,
                self.r: r,
                self.s_next: s_next,
                self.done: done,
                self.model_s_next_Q_val: model_s_next_Q_val,
            },
        )
        self.cum_loss_per_episode += loss
        self.replace_params()
