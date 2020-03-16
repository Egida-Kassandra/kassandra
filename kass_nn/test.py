from kass_nn.parse_logs import LogParser

if __name__ == '__main__':
    train_data = []
    train_labels = []
    parse_logs = LogParser()
    train_data = parse_logs.parse_file('fool.log', train_data)
    print(len(train_data))