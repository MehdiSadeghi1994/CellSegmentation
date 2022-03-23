class MetricHistory():
    def __init__(self, metrics, start_epoch=0, epoch_step=1):
        self.start_epoch = start_epoch
        self.epoch_step = epoch_step
        self.epochs = []
        self.metrics = metrics
        self.metrich_history = {metric.__name__: [] for metric in metrics}



    def step(self, predict, target):
        self.epochs.append(self.start_epoch)
        self.start_epoch += self.epoch_step
        for func in self.metrics:
            self.metrich_history[func.__name__].append(func(predict, target))


    def plot(self, name):
        if isinstance(name, (list)):
            pass
        if isinstance(name, (str)):
            pass