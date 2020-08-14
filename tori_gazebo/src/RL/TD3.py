# author: nikhilbarhate99
# source: https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2/blob/master/TD3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, layer_height=100):
        super(Actor, self).__init__()
        self.deep = True
        
        if self.deep == True:
            self.l1 = nn.Linear(state_dim, layer_height)
            self.l2 = nn.Linear(layer_height, layer_height)
            self.l3 = nn.Linear(layer_height, layer_height)
            self.l4 = nn.Linear(layer_height, action_dim)
        elif self.deep == False:
            self.l1 = nn.Linear(state_dim, 400)
            self.l2 = nn.Linear(400, 300)
            self.l3 = nn.Linear(300, action_dim)

        
        self.max_action = max_action
        
    def forward(self, state):
        if self.deep == False:
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            a = torch.tanh(self.l3(a)) * self.max_action
            return a
        elif self.deep == True:
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            a = F.relu(self.l3(a))
            a = torch.tanh(self.l4(a)) * self.max_action
            return a
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layer_height=100):
        super(Critic, self).__init__()
        self.deep = True
        if self.deep == True:
            self.l1 = nn.Linear(state_dim + action_dim, layer_height)
            self.l2 = nn.Linear(layer_height, layer_height)
            self.l3 = nn.Linear(layer_height, layer_height)
            self.l4 = nn.Linear(layer_height, 1)
        else:
            self.l1 = nn.Linear(state_dim + action_dim, 400)
            self.l2 = nn.Linear(400, 300)
            self.l3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        if self.deep == True:
            q = F.relu(self.l1(state_action))
            q = F.relu(self.l2(q))
            q = F.relu(self.l3(q))
            q = self.l4(q)
            return q
        elif self.deep == False:
            q = F.relu(self.l1(state_action))
            q = F.relu(self.l2(q))
            q = self.l3(q)
            return q
    
class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action, layer_height):
        
        self.actor = Actor(state_dim, action_dim, max_action, layer_height).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, layer_height).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic_1 = Critic(state_dim, action_dim, layer_height).to(device)
        self.critic_1_target = Critic(state_dim, action_dim, layer_height).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        
        self.critic_2 = Critic(state_dim, action_dim, layer_height).to(device)
        self.critic_2_target = Critic(state_dim, action_dim, layer_height).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.max_action = max_action
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        
        for i in range(n_iter):
            if n_iter > 10000:
                if i % 10000 == 0:
                    print('updated {}'.format(i))
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
    '''
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
    '''
    
    def load(self, directory, name, additional_dims=0, additional_actions=0, remove_dimensions_=[]):
        #add_dims = False
        
        torch.set_printoptions(threshold=500)
        #state_dicts = []
        actor_sd = torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage)
        #print('ACTOR:{}'.format(actor_sd))
        actor_target_sd = torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage)
        
        critic_1_sd = torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage)
        critic_1_target_sd = torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage)
        
        critic_2_sd = torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage)
        critic_2_target_sd = torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage)
        
        #print('SD:{}\n\n\n'.format(sd))
        #weights = sd['l1.weight'] #400 rows, 21 columns
        #print('l1 weights: {}'.format(weights.size()))
        if len(remove_dimensions_) > 0:
            actor_sd['l1.weight'] = self.remove_dimensions(actor_sd['l1.weight'], remove_dimensions_)
            actor_target_sd['l1.weight'] = self.remove_dimensions(actor_target_sd['l1.weight'], remove_dimensions_)
            
            critic_1_sd['l1.weight'] = self.remove_dimensions(critic_1_sd['l1.weight'], remove_dimensions_)
            critic_1_target_sd['l1.weight'] = self.remove_dimensions(critic_1_target_sd['l1.weight'], remove_dimensions_)
            
            critic_2_sd['l1.weight'] = self.remove_dimensions(critic_2_sd['l1.weight'], remove_dimensions_)
            critic_2_target_sd['l1.weight'] = self.remove_dimensions(critic_2_target_sd['l1.weight'], remove_dimensions_)
        
        if additional_dims > 0:
            actor_sd['l1.weight'] = self.add_dimensions(actor_sd['l1.weight'], additional_dims)
            actor_target_sd['l1.weight'] = self.add_dimensions(actor_target_sd['l1.weight'], additional_dims)
            
            critic_1_sd['l1.weight'] = self.add_dimensions(critic_1_sd['l1.weight'], additional_dims + additional_actions)
            critic_1_target_sd['l1.weight'] = self.add_dimensions(critic_1_target_sd['l1.weight'], additional_dims + additional_actions)
            
            critic_2_sd['l1.weight'] = self.add_dimensions(critic_2_sd['l1.weight'], additional_dims + additional_actions)
            critic_2_target_sd['l1.weight'] = self.add_dimensions(critic_2_target_sd['l1.weight'], additional_dims + additional_actions)
        '''
        elif additional_dims < 0:
            actor_sd['l1.weight'] = self.remove_dimensions(actor_sd['l1.weight'], additional_dims)
            actor_target_sd['l1.weight'] = self.remove_dimensions(actor_target_sd['l1.weight'], additional_dims)
            
            critic_1_sd['l1.weight'] = self.remove_dimensions(critic_1_sd['l1.weight'], additional_dims)
            critic_1_target_sd['l1.weight'] = self.remove_dimensions(critic_1_target_sd['l1.weight'], additional_dims)
            
            critic_2_sd['l1.weight'] = self.remove_dimensions(critic_2_sd['l1.weight'], additional_dims)
            critic_2_target_sd['l1.weight'] = self.remove_dimensions(critic_2_target_sd['l1.weight'], additional_dims)
        '''
        if additional_actions > 0:
            #print(actor_sd)
            
            actor_sd['l4.bias'] = self.add_actions(actor_sd['l4.bias'], additional_dims)
            actor_target_sd['l4.bias'] = self.add_actions(actor_target_sd['l4.bias'], additional_dims)
            '''
            critic_1_sd['l4.bias'] = self.add_actions(critic_1_sd['l4.bias'], additional_dims)
            critic_1_target_sd['l4.bias'] = self.add_actions(critic_1_target_sd['l4.bias'], additional_dims)
            
            critic_2_sd['l4.bias'] = self.add_actions(critic_2_sd['l4.bias'], additional_dims)
            critic_2_target_sd['l4.bias'] = self.add_actions(critic_2_target_sd['l4.bias'], additional_dims)
            '''
            
            actor_sd['l4.weight'] = self.add_action_weight(actor_sd['l4.weight'], additional_dims)
            actor_target_sd['l4.weight'] = self.add_action_weight(actor_target_sd['l4.weight'], additional_dims)
            '''
            critic_1_sd['l4.weight'] = self.add_action_weight(critic_1_sd['l4.weight'], additional_dims)
            critic_1_target_sd['l4.weight'] = self.add_action_weight(critic_1_target_sd['l4.weight'], additional_dims)
            
            critic_2_sd['l4.weight'] = self.add_action_weight(critic_2_sd['l4.weight'], additional_dims)
            critic_2_target_sd['l4.weight'] = self.add_action_weight(critic_2_target_sd['l4.weight'], additional_dims)
            '''
            
            
        self.actor.load_state_dict(actor_sd)
        self.actor_target.load_state_dict(actor_target_sd)
        
        self.critic_1.load_state_dict(critic_1_sd)
        self.critic_1_target.load_state_dict(critic_1_target_sd)
        
        self.critic_2.load_state_dict(critic_2_sd)
        self.critic_2_target.load_state_dict(critic_2_target_sd)
        
        #weights = actor_sd['l1.weight'] #400 rows, 21 columns
        #biases = actor_sd['l1.bias']
        #print('weights:{}'.format(weights))
        #print('biases:{}'.format(biases))
        
    def add_action_weight(self, t, additional_dims, dim=-1):
        connections = t.size()[1]
        #print('CONNECTIONS:{}'.format(connections))
        #print('TENSOR:{}'.format(t))
        #print('connections:{}'.format(connections))
        #print('inputs:{}'.format(t.size()[1]))
        
        zero_tensor = torch.zeros(additional_dims, connections)
        #print('ZERO TENSOR:{}'.format(zero_tensor))
        tensor_out = torch.cat((t, zero_tensor), dim=-2)
        return(tensor_out)
    
    def add_dimensions(self, t, additional_dims, dim=-1):
        connections = t.size()[0]
        #print('TENSOR:{}'.format(t))
        #print('connections:{}'.format(connections))
        #print('inputs:{}'.format(t.size()[1]))
        
        zero_tensor = torch.zeros(connections, additional_dims)
        tensor_out = torch.cat((t, zero_tensor), dim=-1)
        return(tensor_out)
    
    def add_actions(self, t, additional_dims):
        connections = t.size()[0]
        #print('TENSOR:{}'.format(t))
        #print('connections:{}'.format(connections))
        #print('inputs:{}'.format(t.size()[1]))
        
        zero_tensor = torch.zeros(additional_dims)
        #print('ZERO TENSOR:{}'.format(zero_tensor))
        tensor_out = torch.cat((t, zero_tensor), dim=-1)
        return(tensor_out)
    


        
    def remove_dimensions(self, t, remove_list):
        #connections = t.size()[0] 
        tensor_out = t
        tensor_list = []
        for i, item in enumerate(remove_list):
            '''
            if i == 0:
                tensor = t[:, :remove_list[i]]
            elif i == len(remove_list) - 1:
                tensor = t[:, remove_list[i]+1:]
            else:
                '''
            #start: i == 0
            #between: i > 0 and i < len(remove_list)-1
            #end: i == len(remove_list)-1
            if i == 0:
                tensor = t[:, :remove_list[i]]
            if i < len(remove_list) - 1:
                tensor = t[:, remove_list[i]+1:remove_list[i+1]]
                if i == 0:
                    tensor = torch.cat((t[:, :remove_list[i]], tensor), dim=-1)
            #if i == 0:
                #tensor = torch.cat((t[:, :remove_list[i]], tensor), dim=-1)
            if i == len(remove_list) - 1:
                if i > 0:
                    tensor = t[:, remove_list[i]+1:]
                elif i == 0:
                    tensor = torch.cat((tensor, t[:, remove_list[i]+1:]), dim=-1)
            tensor_list.append(tensor)
            #print(tensor)
        #print(tensor_list)
        tensor_out = torch.cat(tuple(tensor_list), dim=-1)
        return(tensor_out)
        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
    def save_model(self, file_name):
        torch.save(self, file_name)
        #pass
    def load_model(file_name):
        return(torch.load(file_name))
      
        
