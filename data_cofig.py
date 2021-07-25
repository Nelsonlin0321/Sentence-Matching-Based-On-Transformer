from collections import namedtuple

CONFIG = namedtuple("config", ['train_data_path','test_data_path'])
train_data_path = "./data/MRPC/train_data.csv"
test_data_path = "./data/MRPC/test_data.csv"

data_config = CONFIG(train_data_path,test_data_path)

if __name__ == "__main__":
    print(data_config.train_data_path)
