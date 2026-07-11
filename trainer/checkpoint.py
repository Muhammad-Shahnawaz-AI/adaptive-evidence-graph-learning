class CheckpointManager:

    def save(

        self,

        path,

        model,

        optimizer,

        epoch

    ):

        ...

    def load(

        self,

        path,

        model

    ):

        ...