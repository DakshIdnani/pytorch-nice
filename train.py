import os
import torch
import torch.optim as optim
from torchvision import transforms, datasets

from config import cfg
from nice import NICE

# Data
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg['TRAIN_BATCH_SIZE'],
                                         shuffle=True, pin_memory=True)

model = NICE(data_dim=784, num_coupling_layers=cfg['NUM_COUPLING_LAYERS'])
if cfg['USE_CUDA']:
  device = torch.device('cuda')
  model = model.to(device)

# Train the model
model.train()

opt = optim.Adam(model.parameters())

for i in range(cfg['TRAIN_EPOCHS']):
  mean_likelihood = 0.0
  num_minibatches = 0

  for batch_id, (x, _) in enumerate(dataloader):
      x = x.view(-1, 784) + torch.rand(784) / 256.
      if cfg['USE_CUDA']:
        x = x.cuda()

      x = torch.clamp(x, 0, 1)

      z, likelihood = model(x)
      loss = -torch.mean(likelihood)   # NLL

      loss.backward()
      opt.step()
      model.zero_grad()

      mean_likelihood -= loss
      num_minibatches += 1

  mean_likelihood /= num_minibatches
  print('Epoch {} completed. Log Likelihood: {}'.format(i, mean_likelihood))

  if epoch % 5 == 0:
    save_path = os.path.join(cfg['MODEL_SAVE_PATH'], '{}.pt'.format(epoch))
    torch.save(model.state_dict(), save_path)

