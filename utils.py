import matplotlib.pyplot as plt

def plot_training(costs, accs):
    train_acc = accs['train']
    val_acc = accs['val']
    train_cost = costs['train']
    val_cost = costs['val']
    epochs = range(len(train_acc))

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1,)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, val_acc)
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_cost)
    plt.plot(epochs, val_cost)
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Cost')
    
    plt.show()