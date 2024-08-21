import os
from datetime import datetime
from stat import S_IREAD
import sys
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

from nephi.config import Config
from nephi.utils import make_dir, get_git_revisions_hash, set_seed_for_demo, path_import
from nephi.training import train_with_tensorboard


def prepare(args):
    output_path = args.output_path
    exp_name = args.exp_name
    data_path = args.data_path
    # continue_from = args.continue_from
    # is_continue = True if continue_from is not None else False
    dataset_name = data_path.split("/")[-1]

    # Create experiment folder
    timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    exp_folder_path = os.path.join(output_path, dataset_name, exp_name, timestamp)
    make_dir(exp_folder_path)

    # Create checkpoint path, record path and log path
    checkpoint_path = os.path.join(exp_folder_path, "checkpoints")
    make_dir(checkpoint_path)
    record_path = os.path.join(exp_folder_path, "records")
    make_dir(record_path)
    log_path = os.path.join(exp_folder_path, "logs")
    make_dir(log_path)
    test_path = os.path.join(exp_folder_path, "tests")
    make_dir(test_path)

    setting = Config()
    # Update setting file with command input
    setting["dataset"]["data_path"] = data_path
    setting["train"]["output_path"] = exp_folder_path
    # setting["train"]["continue_from"] = args.continue_from
    setting["train"]["gpu_ids"] = args.gpu_id

    # # Write the commit hash for current codebase
    label = get_git_revisions_hash()
    setting["exp"]["git_commit"] = label

    # Write the command argument list to the setting file
    setting["exp"]["command_line"] = " ".join(sys.argv)

    task_output_path = os.path.join(exp_folder_path, "setting.json")
    setting.save_json(task_output_path)

    # Make the setting file read-only
    os.chmod(task_output_path, S_IREAD)

    return setting, exp_folder_path


if __name__ == "__main__":
    """
    A training interface for learning methods.
    Arguments:
        --output_path/ -o: the path of output folder
        --data_path/ -d: the path to the dataset folder
        --exp_name/ -e: the name of the experiment
        --setting_path/ -s: the path to the folder where settings are saved
        --gpu_id/ -g: gpu_id to use
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="An easy interface for training registration models"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        default=None,
        help="the path of output folder",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        required=True,
        type=str,
        default="",
        help="the path to the data folder",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        required=True,
        type=str,
        default=None,
        help="the name of the experiment",
    )
    parser.add_argument(
        "--train_config",
        required=True,
        type=str,
        default="",
        help="the path to the config.py file.",
    )
    parser.add_argument(
        "--epochs_pretrain",
        required=False,
        type=int,
        default=1000,
        help="the number of epochs to train.",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=1000,
        help="the number of epochs to train.",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=1,
        help="batch_size per GPU",
    )
    parser.add_argument(
        "--eval_period",
        required=False,
        type=int,
        default=100,
        help="wait the number of epochs to evaluate.",
    )
    parser.add_argument(
        "--save_period",
        required=False,
        type=int,
        default=100,
        help="wait the number of epochs to save the weight.",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        required=False,
        type=int,
        default=[0],
        nargs="+",
        help="gpu_id to use",
    )
    parser.add_argument(
        "--resume_from",
        required=False,
        type=str,
        default="",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--lr",
        required=False,
        type=float,
        default=5e-5,
        help="The learning rate.",
    )
    parser.add_argument(
        "--lr_step",
        required=False,
        type=int,
        default=400,
        help="The steps after which the learning rate will be reduced",
    )
    parser.add_argument(
        "--with_augmentation",
        required=False,
        type=int,
        default=1,
        help="Whether to use augmentation.",
    )

    args = parser.parse_args()
    print(args)

    # Set gpu
    if args.gpu_id[0] != -1:
        torch.cuda.set_device(args.gpu_id[0])
        # torch.set_default_device(args.gpu_id[0])
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = f"cuda:{args.gpu_id[0]}"
    else:
        device = "cpu"
    print(f"The current device: {torch.cuda.current_device()}")

    # Get device count
    device_count = len(args.gpu_id)

    torch.autograd.set_detect_anomaly(True)

    set_seed_for_demo()
    setting, exp_folder = prepare(args)
    print(f"Experiment information is saved at {exp_folder}")

    writer = SummaryWriter(
        os.path.join(exp_folder, "logs")
        + "/"
        + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )

    assert args.train_config is not None, "Need specify train config python file."
    train_config = path_import(args.train_config).train_config()

    # Save training config
    shutil.copy(args.train_config, f"{exp_folder}/train_config.py")
    shutil.copy(__file__, f"{exp_folder}/train.py")

    # get dataset
    dataset = (
        train_config.make_dataset(args.data_path, args.batch_size * device_count)
        if args.eval_period > 0
        else train_config.make_dataset(
            args.data_path, args.batch_size * device_count, phases=["train"]
        )
    )

    if args.resume_from != "":
        weight_path = args.resume_from
        path_segs = args.resume_from.split("/")
        path_segs[-1] = path_segs[-1].replace("_net_", "_opt_")
        opt_path = "/".join(path_segs)
        print(f"Loading model from {weight_path}")
        print(f"Loading optimizer from {opt_path}")
    else:
        weight_path = None

    start_epoch = 0
    ###### pretrain
    pretrain_net = train_config.make_pretrain_net()
    if weight_path is not None:
        state_dict = torch.load(
            weight_path,
            map_location="cpu",
        )
        if hasattr(pretrain_net, "encoder") and hasattr(pretrain_net, "phi"):
            pretrain_net.encoder.load_state_dict(state_dict["encoder"])
            print("Load encoder")
            pretrain_net.phi.load_state_dict(state_dict["phi"])
            print("Load phi")
            if hasattr(pretrain_net, "phi_inv"):
                pretrain_net.phi_inv.load_state_dict(state_dict["phi_inv"])
                print("Load phi inverse")
        else:
            pretrain_net.load_state_dict(state_dict["net"])
            print("Load model")

        start_epoch = state_dict["epochs"]

    if len(args.gpu_id) > 1:
        pretrain_net = torch.nn.parallel.DataParallel(
            pretrain_net, device_ids=args.gpu_id, output_device=args.gpu_id[0]
        )
    pretrain_net.to(device)

    # optim = torch.optim.Adam(pretrain_net.parameters(), lr=args.lr)
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, pretrain_net.parameters()), lr=args.lr
    )
    print(f"parameter count: {len(optim.param_groups[0]['params'])}")
    if weight_path is not None:
        optim.load_state_dict(torch.load(opt_path)["optimizer"])
        print("load opt")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=args.lr_step,
        gamma=0.9,
    )
    if weight_path is not None:
        lr_scheduler.load_state_dict(torch.load(opt_path)["lr_scheduler"])
        print("load lr scheduler.")

    # TODO: Remove this code later. This is for backward compatibility.
    # for i in range(start_epoch):
    #     optim.zero_grad()
    #     optim.step()
    #     lr_scheduler.step()
    print(lr_scheduler.last_epoch)
    print(f"Learing rate: {lr_scheduler.get_last_lr()[0]}")

    train_with_tensorboard(
        pretrain_net,
        optim,
        dataset,
        total_epochs=start_epoch + args.epochs_pretrain,
        device=device,
        eval_period=args.eval_period,
        save_period=args.save_period,
        writer=writer,
        output_folder=exp_folder,
        start_epoch=start_epoch,
        with_augmentation=args.with_augmentation > 0,
        lr_scheduler=lr_scheduler,
    )

    # delete model
    # del pretrain_net, optim
    # with torch.cuda.device(device):
    #     torch.cuda.empty_cache()

    #### finetune
    # weights_files = os.listdir(f"{exp_folder}/checkpoints")
    # weights_files = list(filter(lambda x: "pretrain_net" in x, weights_files))
    # weights_files = sorted(weights_files)
    # print(f"Load weight from {exp_folder}/checkpoints/{weights_files[-1]}")

    # finetune_net = train_config.make_finetune_net(
    #     f"{exp_folder}/checkpoints/{weights_files[-1]}"
    # )

    # # finetune_net = train_config.make_finetune_net("/playpen-raid2/lin.tian/projects/NePhi/results/hybrid_encoder/w_regularizer/diffusion/2023_01_29_01_23_13/checkpoints/pretrain_net_00260", device)
    # finetune_net.cuda()

    # optim = torch.optim.Adam(finetune_net.parameters(), lr=2e-5)

    # train_with_tensorboard(
    #     finetune_net,
    #     optim,
    #     dataset,
    #     args.epochs,
    #     device=device,
    #     eval_period=args.eval_period,
    #     save_period=args.save_period,
    #     writer=writer,
    #     output_folder=exp_folder,
    #     start_epoch=start_epoch,
    # )

    writer.close()
