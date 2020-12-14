import pickle
import tensorflow as tf
import numpy as np
import logging

def load_expert_data(fn):

	with open(fn, "rb") as f:
		raw_data = pickle.load(f)
		assert("observations" in raw_data and "actions" in raw_data)
		return (raw_data["observations"], raw_data["actions"])


def build_in_mem_tf_dataset(observations, actions):

	dataset = tf.data.Dataset.from_tensor_slices((observations, actions.squeeze()))
	dataset = dataset.map(preprocess)
	return dataset


def build_in_mem_tf_seq_dataset(observations, actions, n_window=5):

	# Build sliding window dataset for non-markovian behavior
	obs_window = []
	act_window = []
	for i in range(observations.shape[0]):
		if (i + n_window - 1) >= observations.shape[0]:
			break
		obs_window.append(observations[i: i+n_window])
		# act_window.append(actions[i: i+N_WINDOW])
		act_window.append(actions[i+n_window-1])
	obs_window = np.array(obs_window)
	act_window = np.array(act_window)

	dataset = tf.data.Dataset.from_tensor_slices((obs_window, act_window.squeeze()))
	dataset = dataset.map(preprocess)
	return dataset


def cos_sin_encoding(action):
	# action: 17x1
	# output: 34x1
	return tf.concat([tf.cos(action/2), tf.sin(action/2)], -1)

def decode_cos_sin(a):
    dim = a.shape[1]
    return 2*np.arctan(np.divide(a[:, dim//2:], a[:, :dim//2]))


def preprocess(o, a):
	return o, tf.cast(cos_sin_encoding(a), tf.float64)

if __name__ == "__main__":
	observations, actions = load_expert_data("../expert_data/train/Humanoid-v2.pkl")
	print("observations:", observations.shape, type(observations))
	print("actions:", actions.shape, type(actions))


	dataset = build_in_mem_tf_dataset(observations, actions)

		# for obs, act in dataset.as_numpy_iterator():
	for obs, act in dataset.batch(64):
		print(obs.shape, act.shape)
		break