import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=4, help="number of users: K")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--noIid', type=bool, default=True, help="imbalance data")
    parser.add_argument('--imb_type', type=str, default="random", help="type of imbalance")
    parser.add_argument('--imb_factor', type=float, default=0.1, help="Unbalanced proportions")


    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--p', type=float, default=0.5, help="non iid sampling prob for class")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=403, help='random seed (default: 1)')
    parser.add_argument('--data_path', type=str, default='./DataSets/data', help="path of dataset")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    args = parser.parse_args()
    return args
