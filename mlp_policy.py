"""
from stable_baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
"""
import gym
import tensorflow as tf

import stable_baselines.common.tf_util as tf_util
from stable_baselines.acktr.utils import dense
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.ppo1.mlp_policy import BasePolicy


class MlpPolicy(BasePolicy):
    recurrent = False

    def __init__(self, name, *args, sess=None, reuse=False, placeholders=None, **kwargs):
        """
        MLP policy for Gail

        :param name: (str) the variable scope name
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param hid_size: (int) the size of the hidden layers
        :param num_hid_layers: (int) the number of hidden layers
        :param sess: (TensorFlow session) The current TensorFlow session containing the variables.
        :param reuse: (bool) allow resue of the graph
        :param placeholders: (dict) To feed existing placeholders if needed
        :param gaussian_fixed_var: (bool) fix the gaussian variance
        """
        super(MlpPolicy, self).__init__(placeholders=placeholders)
        self.sess = sess
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):

        # observation_space 와 action_space
        # ob_space = env.observation_space
        # ac_space = env.action_space
        obs, pdtype = self.get_obs_and_pdtype(ob_space, ac_space)
        # return ::
        # obs :: placeholder
        # pdtype :: action_space의 distribution정보를 담은 객체

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)


        # DeepMind의 observation normalization
        obz = tf.clip_by_value((obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)


        # ===========================[Value function prediction Model]====================================================
        # dense()는 tf.nn.bias_add(tf.matmul(input_tensor, weight), bias) 를 리턴해준다.
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=tf_util.normc_initializer(1.0)))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=tf_util.normc_initializer(1.0))[:, 0]
        # dense(last, 1) 의 경우 return shape 가 [None, 1]이므로, [None]을 얻기 위해서 [:, 0]를 해준다.

        # ===========================[Policy function MOdel]==============================================================
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1),
                                        weight_init=tf_util.normc_initializer(1.0)))


        # 선택한 gym 환경의 action space가 Box 형태인 경우 다음과 같이
        # action distribution의 mean 과 std 를 concate 한 값을
        # action output으로 사용한다.
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", tf_util.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                     initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", tf_util.normc_initializer(0.01))

        # 이 모델에서의 action output은 다음 proba_distribution 에 담기도록 합니다.
        self.proba_distribution = pdtype.proba_distribution_from_flat(pdparam)
        self.state_in = []
        self.state_out = []

        # ppo1/mlp_policy class를 상송했으므로
        # 해당 클래스에서 정의되어 있는 act 함수를 재정의
        # _act를 정의해주면
        # act()를 통해서 action / value 를 얻을수 있다.
        self.stochastic_ph = tf.placeholder(dtype=tf.bool, shape=(), name="stochastic")
        action = tf_util.switch(self.stochastic_ph, self.proba_distribution.sample(), self.proba_distribution.mode())
        self.action = action
        self._act = tf_util.function([self.stochastic_ph, obs], [action, self.vpred])
