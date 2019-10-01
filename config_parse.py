"""
Parser for the config file `config.yaml`.

"""
import yaml
with open("config.yaml", 'r') as config_file:
    config = yaml.load(config_file)


class NetworkConfig:
    def __init__(self):
        net_config = config['autoencoder']
        self.batch_size = net_config['batch_size']
        self.num_epochs = net_config['num_epochs']
        self.hidden_1 = net_config['hidden_1']
        self.hidden_state_size = net_config['hidden_state_size']

        lstm_config = config['lstm']
        self.windows_size = lstm_config['windows_size']
        self.output_window = lstm_config['output_window']
        self.learning_rate = lstm_config['learning_rate']
        self.rnn_hidden = lstm_config['rnn_hidden']
        self.lstm_epochs = lstm_config['num_epochs']
        self.lstm_from_file = lstm_config['from_file']
        self.dropout_prob = lstm_config['dropout_prob']

        portfolio_config = config['portfolio']
        self.frequency = portfolio_config['frequency']
        self.risk_aversion = portfolio_config['risk_aversion']
        self.frequency_optimize_lstm = portfolio_config['frequency_optimize_lstm']

net_config = NetworkConfig()
