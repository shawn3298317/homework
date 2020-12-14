import tensorflow as tf
import numpy as np
from build_dataset import *

class BuildDatasetTest(tf.test.TestCase):

	def setUp(self):
		self.observations, self.actions = load_expert_data("expert_data/train/HalfCheetah-v2.pkl")


	def test_load_expert_data(self):
		obs_shape = self.observations.shape
		act_shape = self.actions.shape
		assert(isinstance(self.observations, np.ndarray))
		assert(isinstance(self.actions, np.ndarray))
		assert(obs_shape[-1] == 17)
		assert(act_shape[-1] == 6)

	def test_build_in_mem_tf_dataset(self):
		dataset = build_in_mem_tf_dataset(self.observations, self.actions)
		assert(isinstance(dataset, Tf))

if __name__ == "__main__":
	tf.test.main()