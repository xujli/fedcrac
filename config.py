def add_args(parser):
    # Training settings
    parser.add_argument(
        "--method",
        type=str,
        default="fedavg",
        metavar="N",
        help="Options are: fedavg, fedprox, moon, mixup, stochdepth, gradaug, fedalign",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/IC/cifar10",
        help="data directory: data/cifar100, data/cifar10, or another dataset",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local clients",
    )

    parser.add_argument(
        "--partition_size",
        type=int,
        default=600,
        metavar="N",
        help="how to partition the dataset on local clients",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=0.5,
        metavar="PA",
        help="alpha value for Dirichlet distribution partitioning of data(default: 0.5)",
    )

    parser.add_argument(
        "--client_number",
        type=int,
        default=10,
        metavar="NN",
        help="number of clients in the FL system",
    )

    parser.add_argument(
        "--silos_number",
        type=int,
        default=5,
        metavar="NN",
        help="number of silos in the FL system",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--net",
        type=str,
        default="modVGG",
        metavar="N",
        help="network arch:[resnet18, resnet56, SimpleCNN, modVGG, Sent140LSTM]",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--momentum", type=bool, default=False, metavar="LR", help="momentum"
    )

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0)

    parser.add_argument("--seed", help="random seed", type=int, default=1)

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="EP",
        help="how many epochs will be trained locally per round",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=100,
        help="how many rounds of communications are conducted",
    )

    parser.add_argument(
        "--pretrained", action="store_true", default=False, help="test pretrained model"
    )

    parser.add_argument(
        "--mu", type=float, default=1, metavar="MU", help="mu value for various methods"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        metavar="MU",
        help="mu value for various methods",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=0.25,
        metavar="WI",
        help="minimum width for subnet training",
    )

    parser.add_argument(
        "--mult",
        type=float,
        default=1.0,
        metavar="MT",
        help="multiplier for subnet training",
    )

    parser.add_argument(
        "--num_subnets",
        type=int,
        default=3,
        help="how many subnets sampled during training",
    )

    parser.add_argument(
        "--save_client",
        action="store_true",
        default=False,
        help="Save client checkpoints each round",
    )

    parser.add_argument(
        "--thread_number",
        type=int,
        default=4,
        metavar="NN",
        help="number of parallel training threads",
    )

    parser.add_argument(
        "--client_sample",
        type=float,
        default=0.4,
        metavar="MT",
        help="Fraction of clients to sample",
    )

    parser.add_argument(
        "--stoch_depth", default=0.5, type=float, help="stochastic depth probability"
    )

    parser.add_argument(
        "--gamma", default=0.1, type=float, help="hyperparameter gamma for mixup"
    )

    parser.add_argument(
        "--additional_experiment_name",
        default="",
        type=str,
        help="",
    )

    parser.add_argument("--match_epoch", type=int, default=100)

    parser.add_argument("--crt_epoch", type=int, default=300)

    parser.add_argument("--times", type=int, default=1)

    parser.add_argument("--LT", type=int, default=1)

    parser.add_argument("--Log", type=int, default=0)

    parser.add_argument("--imbalance_ratio", type=float, default=0.5)

    parser.add_argument("--server_device", type=int, default=0)
    args = parser.parse_args()

    return args
