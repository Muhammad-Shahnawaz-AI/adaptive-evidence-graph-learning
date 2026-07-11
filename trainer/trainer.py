class Trainer:

    def __init__(

        self,

        model,

        optimizer,

        scheduler,

        train_loader,

        val_loader

    ):

        ...

    def train(self):

        ...

    def validate(self):

        ...

    def save_checkpoint(self):

        ...