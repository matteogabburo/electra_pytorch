import os
import random
import torch
import datasets
import numpy as np
from torch import nn
from argparse import ArgumentParser
from functools import partial

from transformers.models.auto.configuration_auto import AutoConfig
from fastai.text.all import (
    ParamScheduler,
    Learner,
    Adam,
    accuracy,
    MatthewsCorrCoef,
    F1Score,
    PearsonCorrCoef,
    SpearmanCorrCoef,
    CrossEntropyLossFlat,
    TensorText,
    noop,
    TensorCategory,
)
from _utils.would_like_to_pr import MyMSELossFlat
from _utils.wsc_trick import (
    ELECTRAWSCTrickLoss,
    wsc_trick_accuracy,
    ELECTRAWSCTrickModel,
)
from finetune_utils import (
    tokenize_sents_max_len,
    SentencePredictor,
    get_layer_lrs,
    collate_roberta_fn
)
from params_splitters import hf_roberta_param_splitter, hf_electra_param_splitter
from hugdatafast.fastai import HF_Datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from _utils.utils import (
    Adam_no_bias_correction,
    linear_warmup_and_then_decay,
    linear_warmup_and_decay,
)
from _utils.would_like_to_pr import GradientClipping

GLUE_TASKS = {
    "MRPC": "mrpc",
    "CoLA": "cola",
    "SST2": "sst2",
    "STSB": "stsb",
    "QQP": "qqp",
    "MNLI": "mnli",
    "QNLI": "qnli",
    "RTE": "rte",
    "WNLI": "wnli",
    "AX": "ax",
}


def setup_tasks(wsc_trick):

    METRICS = {
        **{task: [MatthewsCorrCoef()] for task in ["cola"]},
        **{
            task: [accuracy]
            for task in ["sst2", "mnli", "qnli", "rte", "wnli", "snli", "ax"]
        },
        **{task: [F1Score(), accuracy] for task in ["mrpc", "qqp"]},
        **{task: [PearsonCorrCoef(), SpearmanCorrCoef()] for task in ["stsb"]},
    }
    NUM_CLASS = {
        **{task: 1 for task in ["stsb"]},
        **{task: 2 for task in ["cola", "sst2", "mrpc", "qqp", "qnli", "rte", "wnli"]},
        **{task: 3 for task in ["mnli", "ax"]},
    }
    TEXT_COLS = {
        **{task: ["question", "sentence"] for task in ["qnli"]},
        **{
            task: ["sentence1", "sentence2"] for task in ["mrpc", "stsb", "wnli", "rte"]
        },
        **{task: ["question1", "question2"] for task in ["qqp"]},
        **{task: ["premise", "hypothesis"] for task in ["mnli", "ax"]},
        **{task: ["sentence"] for task in ["cola", "sst2"]},
    }
    LOSS_FUNC = {
        **{
            task: CrossEntropyLossFlat()
            for task in [
                "cola",
                "sst2",
                "mrpc",
                "qqp",
                "mnli",
                "qnli",
                "rte",
                "wnli",
                "ax",
            ]
        },
        **{task: MyMSELossFlat(low=0.0, high=5.0) for task in ["stsb"]},
    }
    if wsc_trick:
        LOSS_FUNC["wnli"] = ELECTRAWSCTrickLoss()
        METRICS["wnli"] = [wsc_trick_accuracy]

    return {
        "metrics": METRICS,
        "num_class": NUM_CLASS,
        "text_cols": TEXT_COLS,
        "loss_func": LOSS_FUNC,
    }


def store_datasets(
    task,
    tasks_param,
    hf_tokenizer,
    config,
    cache_dir,
    max_length,
    double_unordered,
    num_workers,
):
    glue_dsets = {}
    glue_dls = {}

    use_token_typen_ids = (config.model_type != "roberta")
    if not use_token_typen_ids:
        print("Warning: not using token_typen_ids")

    # Load / download datasets.
    dsets = datasets.load_dataset("glue", task, cache_dir=cache_dir)

    # There is two samples broken in QQP training set
    if task == "qqp":
        dsets["train"] = dsets["train"].filter(
            lambda e: e["question2"] != "",
            cache_file_name=os.path.join(
                dsets["train"].cache_directory(), "fixed_train.arrow"
            ),
        )

    # Load / Make tokenized datasets
    tok_func = partial(
        tokenize_sents_max_len,
        hf_tokenizer=hf_tokenizer,
        cols=tasks_param["text_cols"][task],
        max_len=max_length,
        use_token_type_ids=use_token_typen_ids
    )
    glue_dsets[task] = dsets.my_map(
        tok_func, cache_file_names=f"tokenized_{max_length}_{{split}}"
    )

    if double_unordered and task in ["mrpc", "stsb"]:
        swap_tok_func = partial(
            tokenize_sents_max_len,
            hf_tokenizer=hf_tokenizer,
            cols=tasks_param["text_cols"][task],
            max_len=max_length,
            swap=True,
            use_token_type_ids=use_token_typen_ids
        )
        swapped_train = dsets["train"].my_map(
            swap_tok_func, cache_file_name=f"swapped_tokenized_{max_length}_train"
        )
        glue_dsets[task]["train"] = datasets.concatenate_datasets(
            [glue_dsets[task]["train"], swapped_train]
        )

    # Load / Make dataloaders
    hf_dsets = HF_Datasets(
        glue_dsets[task],
        hf_toker=hf_tokenizer,
        n_inp=3,
        cols={
            "inp_ids": TensorText,
            "attn_mask": noop,
            "token_type_ids": noop,
            "label": TensorCategory,
        },
    )
    if double_unordered and task in ["mrpc", "stsb"]:
        dl_kwargs = {"train": {"cache_name": f"double_dl_{max_length}_train.json"}}
    else:
        dl_kwargs = None
    glue_dls[task] = hf_dsets.dataloaders(
        bs=32,
        shuffle_train=True,
        num_workers=num_workers,
        cache_name=f"dl_{max_length}_{{split}}.json",
        dl_kwargs=dl_kwargs,
        create_batch=partial(collate_roberta_fn, pad_token=hf_tokenizer.pad_token_id) if not use_token_typen_ids else None
    )
    return glue_dls


def get_glue_learner(
    task,
    tasks_param,
    config,
    model,
    dataloaders,
    weight_decay,
    hf_tokenizer,
    wsc_trick,
    device,
    lr,
    layer_lr_decay,
    adam_bias_correction,
    schedule,
    seed=None,
    run_name=None,
    inference=False,
):
    is_wsc_trick = task == "wnli" and wsc_trick

    # Num_epochs
    if task in ["rte", "stsb"]:
        num_epochs = 10
    else:
        num_epochs = 3

    # Dataloaders
    dls = dataloaders[task]
    if isinstance(device, str):
        dls.to(torch.device(device))
    elif isinstance(device, list):
        dls.to(torch.device("cuda", device[0]))
    else:
        dls.to(torch.device("cuda:0"))

    # Seeds & PyTorch benchmark
    torch.backends.cudnn.benchmark = True
    if seed:
        dls[0].rng = random.Random(seed)  # for fastai dataloader
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create finetuning model
    if is_wsc_trick:
        model = ELECTRAWSCTrickModel(model, hf_tokenizer.pad_token_id)
    else:
        model = SentencePredictor(
            model.base_model,
            model.base_model.config.hidden_size,
            num_class=tasks_param["num_class"][task],
            xavier_reinited_outlayer=True,
        )

    # Discriminative learning rates
    if config.model_type == "electra":
        splitter = partial(hf_electra_param_splitter, wsc_trick=is_wsc_trick)
    elif config.model_type == "roberta":
        splitter = partial(hf_roberta_param_splitter, wsc_trick=is_wsc_trick)
    else:
        raise ValueError(f"Not supported model type {config.model_type}")
    layer_lrs = get_layer_lrs(
        lr=lr,
        original_lr_layer_decays=True,
        decay_rate=layer_lr_decay,
        num_hidden_layers=model.base_model.config.num_hidden_layers,
    )

    # Optimizer
    if adam_bias_correction:
        opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=weight_decay)
    else:
        opt_func = partial(
            Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=weight_decay
        )

    # Learner
    learn = Learner(
        dls,
        model,
        loss_func=tasks_param["loss_func"][task],
        opt_func=opt_func,
        metrics=tasks_param["metrics"][task],
        splitter=splitter,  # if not inference else trainable_params,
        lr=layer_lrs if not inference else lr,
        path="./checkpoints/glue",
        model_dir=run_name,
    )

    # Multi gpu
    if isinstance(device, list) or device is None:
        learn.create_opt()
        learn.model = nn.DataParallel(learn.model, device_ids=device)

    # Mixed precision
    # learn.to_native_fp16(init_scale=2.0 ** 14)

    # Gradient clip
    learn.add_cb(GradientClipping(1.0))

    # Learning rate schedule
    if schedule == "one_cycle":
        return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs, lr_max=layer_lrs)
    elif schedule == "adjusted_one_cycle":
        return learn, partial(
            learn.fit_one_cycle,
            n_epoch=num_epochs,
            lr_max=layer_lrs,
            div=1e5,
            pct_start=0.1,
        )
    else:
        lr_shed_func = (
            linear_warmup_and_then_decay
            if schedule == "separate_linear"
            else linear_warmup_and_decay
        )
        lr_shedule = ParamScheduler(
            {
                "lr": partial(
                    lr_shed_func,
                    lr_max=np.array(layer_lrs),
                    warmup_pct=0.1,
                    total_steps=num_epochs * (len(dls.train)),
                )
            }
        )
        return learn, partial(learn.fit, n_epoch=num_epochs, cbs=[lr_shedule])


def main(args):

    task = GLUE_TASKS[args.task_name]

    # prepare output dirs
    if not os.path.isdir(args.output_dir):
        print('Prepare output dir "{}"'.format(args.output_dir))
        os.makedirs(args.output_dir)

    # the datasets will be stored in <args_output_dir>/datasets/
    datasets_cache_dir = os.path.join(args.output_dir, "datasets")
    if not os.path.isdir(datasets_cache_dir):
        os.makedirs(datasets_cache_dir)

    # the checkpoints will be stored in <args_output_dir>/checkpoints/
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # task params
    tasks_param = setup_tasks(args.wsc_trick)

    # tokenizer & model
    print('Loading weights and tokenizer from "{}"'.format(args.model_name_or_path))
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    print("Model config: ", config)

    # datasets setup
    dataloaders = store_datasets(
        task,
        tasks_param,
        hf_tokenizer=tokenizer,
        config=config,
        cache_dir=datasets_cache_dir,
        max_length=args.max_seq_length,
        double_unordered=args.double_unordered,
        num_workers=args.num_workers,
    )

    # finetune
    if args.do_train:

        learn, fit_fc = get_glue_learner(
            task,
            tasks_param,
            config,
            model,
            dataloaders,
            args.weight_decay,
            tokenizer,
            args.wsc_trick,
            args.device,
            args.learning_rate,
            args.layer_lr_decay,
            args.adam_bias_correction,
            args.schedule,
            seed=args.seed,
            run_name=args.output_dir,
            inference=True,
        )
        fit_fc()
        learn.save(f"{task}_{args.seed}")

        # save measures
        measures = [
            (measure, str(learn.recorder.log[i]))
            for i, measure in enumerate(learn.recorder.metric_names)
        ]
        print('Saving in "{}"'.format(args.output_dir))
        with open(
            os.path.join(args.output_dir, "eval_results_{}.txt".format(task)), "w"
        ) as f:
            f.write("\n".join(" = ".join(m) for m in measures))


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True, choices=GLUE_TASKS.keys())
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--weight_decay", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=3e-04)
    parser.add_argument("--layer_lr_decay", type=float, default=0.8)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--double_unordered", type=bool, default=True)
    parser.add_argument("--wsc_trick", type=bool, default=False)
    parser.add_argument("--adam_bias_correction", type=bool, default=False)

    parser.add_argument("--schedule", type=str, default="original_linear")

    # get NameSpace of paramters
    args = parser.parse_args()

    main(args)
