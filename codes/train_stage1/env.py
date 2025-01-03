import torch
from algorithm import *


class Holography_env():
    
    def __init__(self, img_h, img_w, model):
        self.actor_space = self.action_space()
        self.num_actions = len(self.actor_space)
        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = 7
        self.obs = None
        self.position = None
        self.image = None
        self.mask = None
        self.recon_value = None
        self.done = False
        self.recon = None
        self.return_to_go_rew = None
        self.algorithm = Hologram(model)

    def action_space(self):

        actor_space = [[0, 2], [0, -1], [0, -2],
                        [1, 0], [2, 0],  [-1, 0],  [-2, 0], 
                        [0, 1], [-3,0], [3,0], [0,3], [0,-3]]             

        return actor_space

    def reset(self, image, mask):

        self.obs = torch.zeros([self.img_h, self.img_w])
        self.position = torch.tensor([32, 32, 0])

        self.obs[self.position[0]][self.position[1]] = 1
        self.image = image
        self.mask = mask

        self.done = False
        repath = torch.zeros([self.img_h, self.img_w])
        repath[self.position[0]][self.position[1]] = 1
        self.recon, recon_value = self.algorithm.forward(self.image, self.mask, repath)
        self.recon_value = recon_value
        self.return_to_go_rew = recon_value

        print('environment is reset')
        print('baseline value:', self.recon_value)
        # return 
        return self.position

    def step(self, action, t): 

        old_position = torch.zeros([1,3])
        old_position = self.position

        x = self.actor_space[action][0]
        y = self.actor_space[action][1]
        
        self.position[0] += x
        self.position[1] += y
        self.position[2] += 1

        new_position = torch.zeros([1, 3])
        new_position = self.position

        old_obs = self.obs
        self.obs[self.position[0]][self.position[1]] = 1

        pre_return_to_go_rew = self.return_to_go_rew

        tmp_obs = self.obs.cuda()

        self.recon, recon_value = self.algorithm.forward(self.image, self.mask, tmp_obs)

        self.recon_value = recon_value
        self.return_to_go_rew = recon_value

        reward = self.return_to_go_rew - pre_return_to_go_rew 

        if t >= 8:
            self.done = True
        else:
            self.done = False

        return new_position, reward, self.done, self.recon
