import numpy as np
import matplotlib.pyplot as plt





def comparative_plot(path1, path2):
    
    dev_acc1 = np.load(path1+'/dev_epoch_accuracies.npy', allow_pickle=True)
    dev_loss1 = np.load(path1+'/dev_epoch_losses.npy', allow_pickle=True)
    train_acc1 = np.load(path1+'/epoch_accuracies.npy', allow_pickle=True)
    train_loss1 = np.load(path1+'/epoch_losses.npy', allow_pickle=True)

    dev_acc2 = np.load(path2+'/dev_epoch_accuracies.npy', allow_pickle=True)
    dev_loss2 = np.load(path2+'/dev_epoch_losses.npy', allow_pickle=True)
    train_acc2 = np.load(path2+'/epoch_accuracies.npy', allow_pickle=True)
    train_loss2 = np.load(path2+'/epoch_losses.npy', allow_pickle=True)

    # dev_acc3 = np.load('experiments/exp_a1_mask_adam'+'/dev_epoch_accuracies.npy', allow_pickle=True)
    # dev_loss3 = np.load('experiments/exp_a1_mask_adam'+'/dev_epoch_losses.npy', allow_pickle=True)
    # train_acc3 = np.load('experiments/exp_a1_mask_adam'+'/epoch_accuracies.npy', allow_pickle=True)
    # train_loss3 = np.load('experiments/exp_a1_mask_adam'+'/epoch_losses.npy', allow_pickle=True)

    # dev_acc4 = np.load('experiments/exp_a1_heavymask'+'/dev_epoch_accuracies.npy', allow_pickle=True)
    # dev_loss4 = np.load('experiments/exp_a1_heavymask'+'/dev_epoch_losses.npy', allow_pickle=True)
    # train_acc4 = np.load('experiments/exp_a1_heavymask'+'/epoch_accuracies.npy', allow_pickle=True)
    # train_loss4 = np.load('experiments/exp_a1_heavymask'+'/epoch_losses.npy', allow_pickle=True)


    name1 = 'Target hidden representation'
    name2 = 'Last hidden representation'
    name3 = 'Target weight 1.1, Other weight 1.0'
    name4 = 'Target weight 5.0, Other weight 1.0'
    print(f"Max accuracy {name1}: ", np.max(dev_acc1))
    print(f'Best Model:', np.argmax(dev_acc1)+1)
    print(f"Max accuracy {name2}: ", np.max(dev_acc2))
    print(f'Best Model:', np.argmax(dev_acc2)+1)

    plt.subplot(121)
    plt.title('Training Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(np.arange(len(train_acc1)), train_acc1, label=name1)
    plt.plot(np.arange(len(train_acc2)), train_acc2, label=name2)
    # plt.plot(np.arange(len(train_acc3)), train_acc3, label=name3)
    # plt.plot(np.arange(len(train_acc4)), train_acc4, label=name4)

    plt.legend()
    # plt.show()
    plt.subplot(122)
    plt.title('Validation Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(np.arange(len(dev_acc1)), dev_acc1, label=name1)
    plt.plot(np.arange(len(dev_acc2)), dev_acc2, label=name2)
    # plt.plot(np.arange(len(dev_acc3)), dev_acc3, label=name3)
    # plt.plot(np.arange(len(dev_acc4)), dev_acc4, label=name4)

    plt.legend()
    # plt.savefig('plots/'+n_model)
    plt.show()
if __name__ == '__main__':
    comparative_plot('experiments/exp_a2_hidden', 'experiments/exp_a2_last')