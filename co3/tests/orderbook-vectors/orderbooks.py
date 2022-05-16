class Orderbooks:
    def __init__(self, *args, **kwargs):

        if "config" in kwargs:

            self.config = kwargs["config"]

