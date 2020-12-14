import os
import sys
import gym

import argparse
import logging
import tensorflow as tf
import numpy as np


from build_dataset import load_expert_data, build_in_mem_tf_dataset, build_in_mem_tf_seq_dataset, decode_cos_sin
from load_policy import load_policy

# Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

FORMAT = '[%(levelname)s]\t%(asctime)-15s\t[%(module)s]\t%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT) #, filename='log/main.log'

# Arg parse
parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str)
parser.add_argument('--algorithm', type=str, default='bc', help='Available algorithms: bc / dag')
parser.add_argument('--dagger_n_iter', type=int, default=5)
parser.add_argument('--dagger_sample_ratio', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=1.0) # SMILe
parser.add_argument('--save_name', type=str)

parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--policy', type=str, default="dnn", help='Available policy: dnn / rnn')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--n_windows', type=int, default=5)

FLAGS = vars(parser.parse_args())

logging.info("Input args: %s" % FLAGS)

# Constants


def main():

    # Load & preprocess dataset
    logging.info("Loading expert data...")

    tr_observations, tr_actions = load_expert_data("expert_data/train/%s-v2.pkl" % FLAGS["env_name"])
    tst_observations, tst_actions = load_expert_data("expert_data/test/%s-v2.pkl" % FLAGS["env_name"])

    logging.debug("Train obs shape: %s", tr_observations.shape)
    logging.debug("Train act shape: %s", tr_actions.shape)
    logging.debug("Test obs shape: %s", tst_observations.shape)
    logging.debug("Test act shape: %s", tst_actions.shape)

    if FLAGS["policy"] == "dnn":
        tr_dataset = build_in_mem_tf_dataset(tr_observations, tr_actions)
        tst_dataset = build_in_mem_tf_dataset(tst_observations, tst_actions)
    else:
        if FLAGS["policy"] != "rnn":
            logging.error("Got invalid policy: %s", FLAGS["policy"])
            exit(-1)
        tr_dataset = build_in_mem_tf_seq_dataset(tr_observations, tr_actions, FLAGS["n_windows"])
        tst_dataset = build_in_mem_tf_seq_dataset(tst_observations, tst_actions, FLAGS["n_windows"])
    # if FLAGS["shuffle"]:
        #tr_dataset = tr_dataset.shuffle(tr_observations.shape[0])
    tr_dataset = tr_dataset.batch(FLAGS["batch_size"])
    tst_dataset = tst_dataset.batch(FLAGS["batch_size"])
    FLAGS["output_dim"] = tr_dataset.element_spec[1].shape[-1]
    FLAGS["input_shape"] = tr_dataset.element_spec[0].shape
    
    logging.debug("Train datset: %s", tr_dataset)
    logging.debug("Test datset: %s", tr_dataset)

    # Load BC model
    model = None
    if FLAGS["policy"] == "dnn":
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(FLAGS["output_dim"], activation="tanh")
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(256, activation="relu", return_sequences=True), # state embedding
            tf.keras.layers.LSTM(128, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(64, activation="relu"),
            tf.keras.layers.Dense(FLAGS["output_dim"], activation="tanh")
        ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError())

    model.build(FLAGS["input_shape"])
    logging.debug("Model summary %s", model.summary())

    # TODO: allow loading from pre-trained checkpoint
    logging.info("Start fitting...")
    model.fit(tr_dataset, epochs=FLAGS["epochs"])

    logging.info("Evaluating on test set...")
    model.evaluate(tst_dataset)

    if FLAGS["algorithm"] == "bc": # behavior cloning
        # Prepare policy function
        cur_policy = get_policy_from_model(model)
        expert_policy = load_policy("experts/%s-v2.pkl" % FLAGS["env_name"])

        logging.info("Testing with trained %s-%s policy...", FLAGS["algorithm"], FLAGS["policy"])
        agent_scores = test_gym(cur_policy, rolling=(FLAGS["policy"] == "rnn"))
        logging.info("Testing with expert policy...")
        expert_scores = test_gym(expert_policy, rolling=False)

        logging.info("Agent scores = %.2f +- %.2f", np.average(agent_scores), np.std(agent_scores))
        logging.info("Expert scores = %.2f +- %.2f", np.average(expert_scores), np.std(expert_scores))

    elif FLAGS["algorithm"] == "dag": # dagger: https://arxiv.org/pdf/1011.0686.pdf

        # Init behavior cloning policy \pi_i
        cur_policy = get_policy_from_model(model)
        expert_policy = load_policy("experts/%s-v2.pkl" % FLAGS["env_name"])

        logging.info("Evaluate in gym before applying DAGGER...")
        init_agent_scores = test_gym(cur_policy, rolling=(FLAGS["policy"] == "rnn"))
        expert_scores = test_gym(expert_policy, rolling=False)
        N_SAMPLE = len(tr_dataset)*FLAGS["dagger_sample_ratio"]*FLAGS["batch_size"]

        for it in range(FLAGS["dagger_n_iter"]):

            beta = (1 - FLAGS["alpha"]) ** (it + 1)
            sample_policy = get_weighted_policy(expert_policy, cur_policy, beta)
            logging.info("Sampling policy, beta = %f", beta)

            tr_dagger_dataset = sample_dagger_data(sample_policy,
                                                   rolling=(FLAGS["policy"] == "rnn"),
                                                   n_sample=N_SAMPLE)
            logging.info("Dagger data: %s, length = %i", tr_dagger_dataset, len(tr_dagger_dataset))

            # D <- D U D_i
            logging.info("After unbatch: %s, length = %i", tr_dataset, len(tr_dataset))

            tr_dataset = tr_dataset.concatenate(tr_dagger_dataset)
            tr_dataset = tr_dataset.shuffle(len(tr_dataset), reshuffle_each_iteration=True)#.batch(FLAGS["batch_size"])
            logging.info("Aggregated dataset size: %s, length = %i", tr_dataset, len(tr_dataset))

            # Retrain
            # reset keras model weight
            tf.keras.backend.clear_session()
            logging.info("Start fitting...")
            model.fit(tr_dataset, epochs=FLAGS["epochs"])

            logging.info("Evaluating on test set...")
            model.evaluate(tst_dataset)
            # TODO: keep track of the best policy?

        logging.info("Testing with trained %s-%s policy...", FLAGS["algorithm"], FLAGS["policy"])
        agent_scores = test_gym(cur_policy, rolling=(FLAGS["policy"] == "rnn"))

        logging.info("[Before Dagger] Agent scores = %.2f +- %.2f", np.average(init_agent_scores), np.std(init_agent_scores))
        logging.info("Agent scores = %.2f +- %.2f", np.average(agent_scores), np.std(agent_scores))
        logging.info("Expert scores = %.2f +- %.2f", np.average(expert_scores), np.std(expert_scores))

    else:
        exit()


def get_weighted_policy(expert_policy, cur_policy, beta):
    if FLAGS["policy"] == "rnn":
        return lambda x: (beta * expert_policy(x[:, -1, :]) + (1 - beta) * cur_policy(x))
    else:
        return lambda x: (beta * expert_policy(x) + (1 - beta) * cur_policy(x))

def get_policy_from_model(model):
    return lambda x: decode_cos_sin(model.predict(x))


def sample_dagger_data(policy, rolling, n_sample=500):

    env = gym.make("%s-v2" % FLAGS["env_name"])
    max_steps = env.spec.timestep_limit

    observations, actions = [], []

    # Keep generating samples until meet got enough samples
    while len(observations) < n_sample:
        
        obs = env.reset()
        done = False
        steps = 0

        rolling_input = []
        n_windows = FLAGS["n_windows"]
        while not done:
            if len(rolling_input) >= n_windows:
                    rolling_input = np.append(rolling_input[1:], obs[None, :], 0)
            else:
                rolling_input.append(obs[:])
                rolling_input = np.append(np.zeros(((n_windows - len(rolling_input)), FLAGS["input_shape"][-1])), np.array(rolling_input), 0)

            obs_input = np.array([rolling_input])
            if rolling:
                action = policy(obs_input)
            else:
                obs_input = obs_input[:, -1, :]
                action = policy(obs_input)
            
            observations.append(obs_input)
            actions.append(action)
            obs, r, done, _ = env.step(tf.cast(action, tf.float32))
            steps += 1
            if steps >= max_steps or len(observations) > n_sample:
                break
        logging.debug("Gathered %i/%i samples so far...", len(observations), n_sample)

    observations, actions = np.array(observations).squeeze(1), np.array(actions)
    logging.info("dagger obs: %s", observations.shape)
    logging.info("actions obs: %s", actions.shape)
    tr_dagger_dataset = build_in_mem_tf_dataset(observations, actions)
    # train_dagger_dataset = train_dagger_dataset.shuffle(len(train_dagger_dataset), reshuffle_each_iteration=True).batch(64)

    return tr_dagger_dataset.shuffle(len(tr_dagger_dataset)).batch(FLAGS["batch_size"])

def test_gym(policy, rolling):
    env = gym.make("%s-v2" % FLAGS["env_name"])
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    expert_actions = []
    for i in range(3):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        
        rolling_input = []
        n_windows = FLAGS["n_windows"]
        while not done:
            if len(rolling_input) >= n_windows:
                rolling_input = np.append(rolling_input[1:], obs[None, :], 0)
            else:
                rolling_input.append(obs[:])
                rolling_input = np.append(np.zeros(((n_windows - len(rolling_input)), FLAGS["input_shape"][-1])), np.array(rolling_input), 0)

            obs_input = np.array([rolling_input])
            if rolling:
                action = policy(obs_input)
            else:
                # logging.info("obs_input shape: %s", obs_input.shape)
                obs_input = obs_input[:, -1, :]
                # logging.info("obs_input after squeeze shape: %s", obs_input.shape)
                action = policy(obs_input)
            
            # observations.append(np.array(rolling_input))
            observations.append(obs[:])
            actions.append(action)
            # expert_actions.append(expert_policy_fn(obs[None, :]).numpy())
            obs, r, done, _ = env.step(tf.cast(action, tf.float32))
            # print("r:", r)
            totalr += r
            steps += 1
            # if True:
                # env.render()
            if steps % 100 == 0: 
                logging.debug("Gym simulate progress: %i/%i", steps, max_steps)
            if steps >= max_steps:
                break
        logging.debug("total: %.2f, %i", totalr, steps)
        logging.debug("Average reward: %.2f", (float(totalr)/steps))
        returns.append(totalr)
    
    return np.array(returns)


if __name__ == '__main__':
    main()