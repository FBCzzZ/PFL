import numpy as np

acc = np.load('./save/fedavg/acc_total.npy')
loss = np.load('./save/fedavg/loss_total.npy')

print(acc.shape)
print(loss.shape)


acc_o = np.load('./save/ours/all/acc_total.npy')
loss_o = np.load('./save/ours/all/loss_total.npy')

print(acc_o.shape)
print(loss_o.shape)