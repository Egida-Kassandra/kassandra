import parse_logs

if __name__ == '__main__':
    train_data = []
    train_labels = []
    train_data = parse_logs.parse_file('fool.log', train_data)
    print(len(train_data))