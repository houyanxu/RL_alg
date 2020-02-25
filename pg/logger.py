from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self,logger_config):
        self.writer = SummaryWriter(log_dir=logger_config['log_dir'])

    def save(self,info):
        pass
