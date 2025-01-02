import os
from datetime import datetime

import torch
import numpy as np

from PPO import PPO
from prop_ideal import *
from env import *
import argparse
################################### Training ###################################
def train():
    print("============================================================================================")

    parser = argparse.ArgumentParser(description="Holography_PPO")
    parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image_dir', default='./data/image')
    parser.add_argument('--depth_dir', default='./data/depth')
    parser.add_argument('--channel', default='1', type=int, help='blue:0 ; green:1; red: 2')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ## load data
    dataset = FocalstackDDataset(args.image_dir, args.depth_dir)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for _, images_mask in enumerate(dataloader, 0): 
        focal_stack = torch.cat(images_mask[0], dim=0)
        target_amp = focal_stack[:, args.channel, :, :].unsqueeze(1)
        mask = images_mask[1].permute(1, 0, 2, 3)

    ####### initialize environment hyperparameters ######
    env_name = "Motion Holography"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 8                  # max timesteps in one episode
    max_training_timesteps = int(30000)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(100)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2000)   # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 20      # update policy every n timesteps
    K_epochs = 200                         # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0001       # learning rate for actor network
    lr_critic = 0.0003       # learning rate for critic network

    random_seed = 1234         # set random seed if required (0 = no random seed)
    #####################################################

    ################ Propagation hyperparameters ################
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    prop_dist = [-3 * mm, -2 * mm, -1 * mm, 0 * mm, 1 * mm, 2 * mm, 3 * mm]
    mid_prop = 0 * cm
    prop_dist = [x + mid_prop for x in prop_dist]

    wavelength= 520 * nm
    feature_size = (3.74 * um, 3.74 * um)
    model_citl = Propagation(prop_dist, wavelength, feature_size, 'ASM', F_aperture=1, dim=1)
    #####################################################

    print("training environment name : " + env_name)
    env = Holography_env(img_h=64, img_w=64, model= model_citl)

    # state space dimension
    state_dim = 3

    # action space dimension
    action_dim = env.num_actions

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + 'step_' + str(max_ep_len) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 5          #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, max_ep_len)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # record best way
    best_reward = 0
    best_way = []

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset(target_amp, mask)
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action, t)
            best_way.append(action)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes

                log_avg_reward = round(log_avg_reward.item(), 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward.item(), 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                if best_reward < current_ep_reward:
                    best_reward = current_ep_reward
                    model_save_path = "./best_model.pth"
                    with open("best_path.txt","w") as f:
                        for w in best_way:
                            f.write(str(w)+"\n")
                    ppo_agent.save(model_save_path)
                best_way = []
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
