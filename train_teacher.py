import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
)
from train_and_eval import run_transductive, run_inductive


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_false",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=5, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="ACM", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")

    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configuration",
    )
    parser.add_argument("--teacher", type=str, default="RSAGE", help="Teacher model")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Model number of layers"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Model hidden layer dimensions"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=2000000)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for sampler"
    )

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0.,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )

    args = parser.parse_args()
    return args


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    output_path = args.output_path
    if args.feature_noise != 0:
        output_path = Path.cwd().joinpath(
            output_path, "noisy_features", f"noise_{args.feature_noise}"
        )

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            output_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")

    """ Load data """
    g, feats, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path
    )

    args.views = list(g.keys())
    # logger.info(f"Total {g.number_of_nodes()} nodes.")
    # logger.info(f"Total {g.number_of_edges()} edges.")
    args.feat_dim = feats.shape[1]
    args.label_dim = int(labels.max()) + 1

    if 0 < args.feature_noise <= 1:
        feats = (
                        1 - args.feature_noise
                ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")

    """ Model init """
    model = Model(conf)
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator(conf["dataset"])

    """ Data split and run """
    loss_and_score = []
    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)

        out, score_val, score_test = run_transductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test]

    elif args.exp_setting == "ind":
        indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed)

        out, score_val, score_test, score_test_tran, score_test_ind = run_inductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test, score_test_tran, score_test_ind]

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving teacher outputs """
    if isinstance(out, tuple):
        out_all_view = out[0]
        out_all_view = out_all_view.log_softmax(dim=1)
        out_all_view = out_all_view.detach().cpu().numpy()
        np.savez(output_dir.joinpath("out_Whole"), out_all_view)
        for view, out_per_view in out[1].items():
            out_per_view = out_per_view.log_softmax(dim=1)
            out_per_view = out_per_view.detach().cpu().numpy()
            np.savez(output_dir.joinpath(f"out_{view}"), out_per_view)
    else:
        out_np = out.detach().cpu().numpy()
        np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")

    print('This run', score_lst)

    return score_lst


def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    print(score_str)


if __name__ == "__main__":
    main()
