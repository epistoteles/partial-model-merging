import torch
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
import os
from codecarbon import track_emissions
from itertools import product
from tqdm import tqdm
from src.utils import (
    load_model,
    get_loaders,
    parse_model_name,
    get_evaluations_dir,
    permute_model,
    interpolate_models,
    smart_interpolate_models,
    ensure_numpy,
    expand_model,
    reset_bn_stats,
    repair,
    partial_repair,
    get_num_params, prune_classifier,
)
from src.models.VGG import VGG
from src.models.ResNet import ResNet18
from src.models.MLP import MLP


def get_acc_and_loss(model: torch.nn.Module, loader):
    """
    Evaluates a single model in terms of accuracy and loss
    :param model: the model
    :param loader: a matching FFCV data loader
    :return: (accuracy, loss)
    """
    model = model.cuda()
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
            total += len(labels)
            loss = F.cross_entropy(outputs, labels.cuda())
            losses.append(loss.item())
    return correct / total, np.array(losses).mean()


@track_emissions()
def evaluate_single_model(model_name: str):
    """
    Evaluates a single model in terms of accuracy and loss (and saves the result)
    :param model_name: the name of the model checkpoint
    :return: (train accuracy, train loss, test accuracy, test loss)
    """
    dataset, model_type, size, batch_norm, width, variant = parse_model_name(model_name)
    evaluations_dir = get_evaluations_dir(subdir="single_model")
    filepath = os.path.join(evaluations_dir, f"{model_name}.csv")
    columns = ("train_acc", "train_loss", "test_acc", "test_loss")

    if os.path.exists(filepath):
        train_acc, train_loss, test_acc, test_loss = np.genfromtxt(filepath, delimiter=",", skip_header=1)
        values = (train_acc, train_loss, test_acc, test_loss)
        print(f"📤 Loaded saved accuracies and losses for {model_name}")
    else:
        model = load_model(model_name).cuda()
        _, train_noaug_loader, test_loader = get_loaders(dataset)

        train_acc, train_loss = get_acc_and_loss(model, train_noaug_loader)
        test_acc, test_loss = get_acc_and_loss(model, test_loader)

        values = (train_acc, train_loss, test_acc, test_loss)
        np.savetxt(filepath, [columns, values], delimiter=",", fmt="%s")

        print(f"📥 Accuracies and losses saved for {model_name}")

    for c, v in zip(columns, values):
        print(f"{c:<12} {v}")

    return train_acc, train_loss, test_acc, test_loss


@track_emissions()
def only_save_correlations(model_name_a: str, model_name_b: str = None, threshold=1.1) -> None:
    """
    Permutes model b and saves the correlations and perm_maps, then does nothing.
    :param model_name_a: the name of the first (reference) model checkpoint
    :param model_name_b: the name of the second model checkpoint
    :return:
    """
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    print(f"Collecting correlations for {model_name_a}, {model_name_b}")

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    save_corr_path = os.path.join(get_evaluations_dir(subdir="correlations"), f"{model_name_a}{variant_b}.safetensors")

    model_a = load_model(model_name_a).cuda()
    model_b = load_model(model_name_b).cuda()
    train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)
    _ = permute_model(
        reference_model=model_a,
        model=model_b,
        loader=train_aug_loader,
        save_corr_path=save_corr_path,
        threshold=threshold,
    )
    print("📥 Correlations and perm_maps saved as .safetensors")


@track_emissions()
def evaluate_two_models(
    model_name_a: str,
    model_name_b: str = None,
    interpolation_steps: int = 21,
    expansions: list[float] = None,
    threshold=1.1,
):
    """
    Evaluates two models in terms of accuracy and loss with different combination techniques (and saves the result)
    :param model_name_a: the name of the first (reference) model checkpoint
    :param model_name_b: the name of the second model checkpoint
    :param interpolation_steps: number of interpolation steps between the models, i.e. 21 = 1.0, 0.95, 0.9, 0.85, ...
    :param expansions: which expansions to evaluate (original width + buffer width)
    :return: TODO
    """
    if expansions is None:
        expansions = [1.1, 1.5]
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    print(f"Evaluating {model_name_a}, {model_name_b}\n" f"steps: {interpolation_steps}; expansions: {expansions}")

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    if dataset_a[-1] in ["A", "B", "C", "D"]:
        dataset_a = dataset_a[:-1]  # we don't want to use subsampled activations for alignment and REPAIR - or do we?

    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="two_models")
    filepath = os.path.join(evaluations_dir, f"{model_name_a}{variant_b}.safetensors")

    model_a = load_model(model_name_a).cuda()
    model_b = load_model(model_name_b).cuda()
    train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

    if os.path.exists(filepath):
        metrics = load_file(filepath)
        print(f"📤 Loaded saved metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        metrics = {"alphas": torch.linspace(0.0, 1.0, interpolation_steps)}

        print("Collecting ensembling metrics ...")
        metrics["ensembling_train_accs"], metrics["ensembling_train_losses"] = evaluate_two_models_ensembling(
            model_a, model_b, train_noaug_loader, interpolation_steps
        )
        metrics["ensembling_test_accs"], metrics["ensembling_test_losses"] = evaluate_two_models_ensembling(
            model_a, model_b, test_loader, interpolation_steps
        )
        print(f"Midpoint test acc: {metrics['ensembling_test_accs'][10]}")

        print("Collecting naive merging metrics ...")
        metrics["naive_train_accs"], metrics["naive_train_losses"] = evaluate_two_models_merging(
            model_a, model_b, train_noaug_loader, interpolation_steps
        )
        metrics["naive_test_accs"], metrics["naive_test_losses"] = evaluate_two_models_merging(
            model_a, model_b, test_loader, interpolation_steps
        )
        print(f"Midpoint test acc: {metrics['naive_test_accs'][10]}")

        print("Collecting permuted merging metrics ...")
        save_corr_path = os.path.join(
            get_evaluations_dir(subdir="correlations"), f"{model_name_a}{variant_b}.safetensors"
        )
        model_b_perm = permute_model(
            reference_model=model_a,
            model=model_b,
            loader=train_aug_loader,
            save_corr_path=save_corr_path,
            threshold=threshold,
        )
        metrics["merging_train_accs"], metrics["merging_train_losses"] = evaluate_two_models_merging(
            model_a, model_b_perm, train_noaug_loader, interpolation_steps
        )
        metrics["merging_test_accs"], metrics["merging_test_losses"] = evaluate_two_models_merging(
            model_a, model_b_perm, test_loader, interpolation_steps
        )
        print(f"Midpoint test acc: {metrics['merging_test_accs'][10]}")

        print("Collecting permuted merging + REPAIR metrics ...")
        (
            metrics["merging_REPAIR_train_accs"],
            metrics["merging_REPAIR_train_losses"],
        ) = evaluate_two_models_merging_REPAIR(
            model_a, model_b_perm, train_noaug_loader, train_aug_loader, interpolation_steps
        )
        metrics["merging_REPAIR_test_accs"], metrics["merging_REPAIR_test_losses"] = evaluate_two_models_merging_REPAIR(
            model_a, model_b_perm, test_loader, train_aug_loader, interpolation_steps
        )
        print(f"Midpoint test acc: {metrics['merging_REPAIR_test_accs'][10]}")

        save_evaluation_checkpoint(metrics, filepath)
        metrics = load_file(filepath)  # necessary because of a safetensors bug

    for k in expansions:
        if f"partial_merging_REPAIR_{k}_test_accs" not in metrics.keys():
            print(f"Collecting partial merging metrics ({k}) ...")
            model_a = load_model(model_name_a).cuda()
            model_b = load_model(model_name_b).cuda()

            model_a = expand_model(model_a, k).cuda()
            model_b = expand_model(model_b, k).cuda()
            model_b_perm = permute_model(
                reference_model=model_a, model=model_b, loader=train_aug_loader, threshold=threshold
            )
            (
                metrics[f"partial_merging_{k}_train_accs"],
                metrics[f"partial_merging_{k}_train_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, train_noaug_loader, interpolation_steps)
            (
                metrics[f"partial_merging_{k}_test_accs"],
                metrics[f"partial_merging_{k}_test_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, test_loader, interpolation_steps)
            print(f"Midpoint test acc: {metrics[f'partial_merging_{k}_test_accs'][10]}")

            print(f"Collecting partial merging + REPAIR metrics ({k}) ...")
            (
                metrics[f"partial_merging_REPAIR_{k}_train_accs"],
                metrics[f"partial_merging_REPAIR_{k}_train_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, train_noaug_loader, train_aug_loader, interpolation_steps
            )
            (
                metrics[f"partial_merging_REPAIR_{k}_test_accs"],
                metrics[f"partial_merging_REPAIR_{k}_test_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, test_loader, train_aug_loader, interpolation_steps
            )
            print(f"Midpoint test acc: {metrics[f'partial_merging_REPAIR_{k}_test_accs'][10]}")

            save_evaluation_checkpoint(metrics, filepath)
            metrics = load_file(filepath)  # necessary because of a safetensors bug

    return metrics


def save_evaluation_checkpoint(metrics, filepath, csv: bool = True):
    save_file(metrics, filename=filepath)
    if csv:
        np.savetxt(
            filepath.replace(".safetensors", ".csv"),
            np.asarray([list(metrics.keys()), *list(zip(*[ensure_numpy(x) for x in metrics.values()]))]),
            delimiter=",",
            fmt="%s",
        )
    print(f"📥 Metrics checkpointed as {'.csv and ' if csv else ''}.safetensors")


def evaluate_two_models_ensembling(
    model_a: torch.nn.Module, model_b: torch.nn.Module, loader, interpolation_steps: int = 21
):
    model_a.eval()
    model_b.eval()

    losses = torch.zeros(interpolation_steps)
    correct = torch.zeros(interpolation_steps)
    total = 0
    batches = 0

    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs_a = model_a(inputs.cuda()).cpu()
            outputs_b = model_b(inputs.cuda()).cpu()
            labels = labels.cpu()

            alphas = torch.linspace(0.0, 1.0, interpolation_steps).reshape(interpolation_steps, 1, 1)
            outputs_a = outputs_a.unsqueeze(0).repeat(interpolation_steps, 1, 1)
            outputs_b = outputs_b.unsqueeze(0).repeat(interpolation_steps, 1, 1)
            outputs = (1 - alphas) * outputs_a + alphas * outputs_b

            pred = outputs.reshape(outputs.shape[1] * interpolation_steps, -1).argmax(dim=1).reshape(outputs.shape[:-1])
            correct += (labels == pred).sum(dim=1)
            losses += torch.Tensor([F.cross_entropy(x, labels) for x in outputs])  # this is faster than torch.vmap
            total += len(labels)
            batches += 1

    return correct / total, losses / batches


def evaluate_two_models_merging(
    model_a: torch.nn.Module, model_b: torch.nn.Module, loader, interpolation_steps: int = 21
):
    print("Interpolating and evaluating model")

    model_a.eval()
    model_b.eval()

    accs = []
    losses = []

    for alpha in tqdm(torch.linspace(0.0, 1.0, interpolation_steps) if interpolation_steps > 1 else [0.5]):
        model_merged = smart_interpolate_models(model_a, model_b, alpha)
        acc, loss = get_acc_and_loss(model_merged, loader)
        accs.append(acc)
        losses.append(loss)

    return torch.FloatTensor(accs), torch.FloatTensor(losses)


def evaluate_two_models_merging_REPAIR(
    model_a: torch.nn.Module, model_b: torch.nn.Module, loader, repair_loader, interpolation_steps: int = 21
):
    print("Interpolating, REPAIRing, and evaluating model")

    model_a.eval()
    model_b.eval()

    accs = []
    losses = []

    for alpha in tqdm(torch.linspace(0.0, 1.0, interpolation_steps) if interpolation_steps > 1 else [0.5]):
        model_merged = smart_interpolate_models(model_a, model_b, alpha)
        model_merged.eval()
        if model_a.bn:  # there is no special REPAIR for models with bn
            reset_bn_stats(model_merged.cuda(), repair_loader)
            model_repaired = model_merged
        else:
            model_repaired = partial_repair(model_merged, model_a, model_b, repair_loader, alpha)
        acc, loss = get_acc_and_loss(model_repaired, loader)
        accs.append(acc)
        losses.append(loss)

    return torch.FloatTensor(accs), torch.FloatTensor(losses)


def experiment_b(model_name_a: str, model_name_b: str = None, threshold=1.1):
    """
    Conducts leave-one-out experiments with full merging vs. ensembling and saves the results
    """
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="experiment_b")
    filepath = os.path.join(evaluations_dir, f"experiment-b-{model_name_a}{variant_b}.safetensors")

    if os.path.exists(filepath):
        metrics = load_file(filepath)
        print(f"📤 Loaded saved leave-one-out metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        model_a = load_model(model_name_a).cuda()
        model_b = load_model(model_name_b).cuda()

        if dataset_a[-1].upper() in ['A', 'B', 'C', 'D', 'E', 'F']:
            dataset_a = dataset_a[:-1]
        train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

        num_layers = model_a.num_layers
        default_num_params = get_num_params(model_a)

        metrics = {}

        all_expansions = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

        metrics["default_num_params"] = torch.ones(len(all_expansions), num_layers) * default_num_params

        metrics["only_expand_layer_i_train_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_train_losses"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_test_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_test_losses"] = torch.zeros(len(all_expansions), num_layers)

        metrics["only_expand_layer_i_REPAIR_train_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_REPAIR_train_losses"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_REPAIR_test_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_REPAIR_test_losses"] = torch.zeros(len(all_expansions), num_layers)

        metrics["only_expand_layer_i_num_params"] = torch.zeros(len(all_expansions), model_a.num_layers)

        for i in range(num_layers):
            for exp_idx, exp in enumerate(all_expansions):
                print(f"Only expanding layer {i+1} of {num_layers} with factor {exp}")
                expansions = torch.ones(model_a.num_layers)
                expansions[i] = exp
                model_a_exp = expand_model(model_a, expansions).cuda()
                model_b_exp = expand_model(model_b, expansions).cuda()
                model_b_exp_perm = permute_model(model_a_exp, model_b_exp, train_aug_loader, threshold=threshold).cuda()
                train_acc, train_loss = evaluate_two_models_merging(
                    model_a_exp, model_b_exp_perm, train_noaug_loader, 1
                )
                test_acc, test_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, test_loader, 1)
                train_acc_REPAIR, train_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                    model_a_exp, model_b_exp_perm, train_noaug_loader, train_aug_loader, 1
                )
                test_acc_REPAIR, test_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                    model_a_exp, model_b_exp_perm, test_loader, train_aug_loader, 1
                )
                num_params = get_num_params(smart_interpolate_models(model_a_exp, model_b_exp_perm), ignore_zeros=True)

                metrics["only_expand_layer_i_train_accs"][exp_idx][i] = train_acc
                metrics["only_expand_layer_i_train_losses"][exp_idx][i] = train_loss
                metrics["only_expand_layer_i_test_accs"][exp_idx][i] = test_acc
                metrics["only_expand_layer_i_test_losses"][exp_idx][i] = test_loss

                metrics["only_expand_layer_i_REPAIR_train_accs"][exp_idx][i] = train_acc_REPAIR
                metrics["only_expand_layer_i_REPAIR_train_losses"][exp_idx][i] = train_loss_REPAIR
                metrics["only_expand_layer_i_REPAIR_test_accs"][exp_idx][i] = test_acc_REPAIR
                metrics["only_expand_layer_i_REPAIR_test_losses"][exp_idx][i] = test_loss_REPAIR

                metrics["only_expand_layer_i_num_params"][exp_idx][i] = num_params

                print(f"Layer {i}, expansion {exp}: {test_acc=}, {test_acc_REPAIR=}")
                save_evaluation_checkpoint(metrics, filepath, csv=False)
                metrics = load_file(filepath)  # necessary because of a safetensors bug

        # only merging layer i
        # train_accs = []
        # train_losses = []
        # test_accs = []
        # test_losses = []
        #
        # train_accs_REPAIR = []
        # train_losses_REPAIR = []
        # test_accs_REPAIR = []
        # test_losses_REPAIR = []
        #
        # nums_params = []
        #
        # for i in range(model_a.num_layers):
        #     print(f"Only merging layer {i+1} of {num_layers}")
        #     expansions = torch.ones(model_a.num_layers) * 2
        #     expansions[i] = 1.0
        #     model_a_exp = expand_model(model_a, expansions).cuda()
        #     model_b_exp = expand_model(model_b, expansions).cuda()
        #     model_b_exp_perm = permute_model(model_a_exp, model_b_exp, train_aug_loader).cuda()
        #     train_acc, train_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, train_noaug_loader, 1)
        #     test_acc, test_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, test_loader, 1)
        #     train_acc_REPAIR, train_loss_REPAIR = evaluate_two_models_merging_REPAIR(
        #         model_a_exp, model_b_exp_perm, train_noaug_loader, train_aug_loader, 1
        #     )
        #     test_acc_REPAIR, test_loss_REPAIR = evaluate_two_models_merging_REPAIR(
        #         model_a_exp, model_b_exp_perm, test_loader, train_aug_loader, 1
        #     )
        #     num_params = get_num_params(model_a_exp)
        #
        #     train_accs.append(train_acc.item())
        #     train_losses.append(train_loss.item())
        #     test_accs.append(test_acc.item())
        #     test_losses.append(test_loss.item())
        #
        #     train_accs_REPAIR.append(train_acc_REPAIR.item())
        #     train_losses_REPAIR.append(train_loss_REPAIR.item())
        #     test_accs_REPAIR.append(test_acc_REPAIR.item())
        #     test_losses_REPAIR.append(test_loss_REPAIR.item())
        #
        #     nums_params.append(num_params)
        #
        # metrics["only_merge_i_train_accs"] = torch.FloatTensor(train_accs)
        # metrics["only_merge_i_train_losses"] = torch.FloatTensor(train_losses)
        # metrics["only_merge_i_test_accs"] = torch.FloatTensor(test_accs)
        # metrics["only_merge_i_test_losses"] = torch.FloatTensor(test_losses)
        #
        # metrics["only_merge_i_REPAIR_train_accs"] = torch.FloatTensor(train_accs_REPAIR)
        # metrics["only_merge_i_REPAIR_train_losses"] = torch.FloatTensor(train_losses_REPAIR)
        # metrics["only_merge_i_REPAIR_test_accs"] = torch.FloatTensor(test_accs_REPAIR)
        # metrics["only_merge_i_REPAIR_test_losses"] = torch.FloatTensor(test_losses_REPAIR)
        #
        # metrics["only_merge_i_num_params"] = torch.FloatTensor(nums_params)
        #
        # save_evaluation_checkpoint(metrics, filepath)
        # metrics = load_file(filepath)  # necessary because of a safetensors bug
        # print(f"📥 Saved leave-ont-out for {model_name_a}{variant_b} as .csv and .safetensors")

    return metrics


def experiment_b_ResNet18(model_name_a: str, model_name_b: str = None, threshold=1.1):
    """
    Conducts leave-one-out experiments with full merging vs. ensembling and saves the results
    """
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="experiment_b")
    filepath = os.path.join(evaluations_dir, f"experiment-b-{model_name_a}{variant_b}.safetensors")

    if os.path.exists(filepath):
        metrics = load_file(filepath)
        print(f"📤 Loaded saved leave-one-out metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        model_a = load_model(model_name_a).cuda()
        model_b = load_model(model_name_b).cuda()

        train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

        num_layers = 12
        default_num_params = get_num_params(model_a)

        metrics = {}

        all_expansions = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

        metrics["default_num_params"] = torch.ones(len(all_expansions), num_layers) * default_num_params

        metrics["only_expand_layer_i_train_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_train_losses"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_test_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_test_losses"] = torch.zeros(len(all_expansions), num_layers)

        metrics["only_expand_layer_i_REPAIR_train_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_REPAIR_train_losses"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_REPAIR_test_accs"] = torch.zeros(len(all_expansions), num_layers)
        metrics["only_expand_layer_i_REPAIR_test_losses"] = torch.zeros(len(all_expansions), num_layers)

        metrics["only_expand_layer_i_num_params"] = torch.zeros(len(all_expansions), model_a.num_layers)

        indices = {
            0: [1],
            1: [3],
            2: [5],
            3: [7],
            4: [9],
            5: [11],
            6: [13],
            7: [15],
            8: [0, 2, 4],
            9: [6, 8],
            10: [10, 12],
            11: [14, 16],
        }

        for i in range(12):
            for exp_idx, exp in enumerate(all_expansions):
                # if exp not in [1.2, 1.5, 1.8]:
                #     continue
                print(
                    f"Doing expansion {i+1} of 12 on layers {','.join([str(x) for x in indices[i]])} with factor {exp}"
                )
                expansions = torch.ones(model_a.num_layers)
                for index in indices[i]:
                    expansions[index] = exp
                model_a_exp = expand_model(model_a, expansions).cuda()
                model_b_exp = expand_model(model_b, expansions).cuda()
                model_b_exp_perm = permute_model(
                    model_a_exp, model_b_exp, train_noaug_loader, threshold=threshold
                ).cuda()
                train_acc, train_loss = evaluate_two_models_merging(
                    model_a_exp, model_b_exp_perm, train_noaug_loader, 1
                )
                test_acc, test_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, test_loader, 1)
                train_acc_REPAIR, train_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                    model_a_exp, model_b_exp_perm, train_noaug_loader, train_noaug_loader, 1
                )
                test_acc_REPAIR, test_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                    model_a_exp, model_b_exp_perm, test_loader, train_noaug_loader, 1
                )
                num_params = get_num_params(smart_interpolate_models(model_a_exp, model_b_exp_perm), ignore_zeros=True)

                metrics["only_expand_layer_i_train_accs"][exp_idx][i] = train_acc
                metrics["only_expand_layer_i_train_losses"][exp_idx][i] = train_loss
                metrics["only_expand_layer_i_test_accs"][exp_idx][i] = test_acc
                metrics["only_expand_layer_i_test_losses"][exp_idx][i] = test_loss

                metrics["only_expand_layer_i_REPAIR_train_accs"][exp_idx][i] = train_acc_REPAIR
                metrics["only_expand_layer_i_REPAIR_train_losses"][exp_idx][i] = train_loss_REPAIR
                metrics["only_expand_layer_i_REPAIR_test_accs"][exp_idx][i] = test_acc_REPAIR
                metrics["only_expand_layer_i_REPAIR_test_losses"][exp_idx][i] = test_loss_REPAIR

                metrics["only_expand_layer_i_num_params"][exp_idx][i] = num_params

                print(f"Layer set {i+1}, expansion {exp}: {test_acc=}, {test_acc_REPAIR=}")
                save_evaluation_checkpoint(metrics, filepath, csv=False)
                metrics = load_file(filepath)  # necessary because of a safetensors bug

    return metrics


def experiment_c(model_name_a: str, model_name_b: str = None, threshold=1.1):
    """
    Expands the model from front to back
    """
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="experiment_c")
    filepath = os.path.join(evaluations_dir, f"experiment-c-{model_name_a}{variant_b}.safetensors")

    if os.path.exists(filepath):
        metrics = load_file(filepath)
        print(f"📤 Loaded saved leave-one-out metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        model_a = load_model(model_name_a).cuda()
        model_b = load_model(model_name_b).cuda()

        train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

        default_num_params = get_num_params(model_a)

        metrics = {}

        all_expansions = [1.2, 1.4, 1.6, 1.8, 2.0]

        if model_type_a == "ResNet":
            indices = {
                0: [1],
                1: [3],
                2: [5],
                3: [7],
                4: [9],
                5: [11],
                6: [13],
                7: [15],
                8: [0, 2, 4],
                9: [6, 8],
                10: [10, 12],
                11: [14, 16],
            }
            sorted_layers = [8, 0, 1, 2, 9, 3, 4, 10, 5, 6, 11, 7]
        else:
            indices = {x: [x] for x in range(model_a.num_layers)}
            sorted_layers = range(len(indices))

        metrics["default_num_params"] = torch.zeros(len(all_expansions), len(sorted_layers)) * default_num_params

        metrics["only_expand_layer_i_train_accs"] = torch.zeros(len(all_expansions), len(sorted_layers))
        metrics["only_expand_layer_i_train_losses"] = torch.zeros(len(all_expansions), len(sorted_layers))
        metrics["only_expand_layer_i_test_accs"] = torch.zeros(len(all_expansions), len(sorted_layers))
        metrics["only_expand_layer_i_test_losses"] = torch.zeros(len(all_expansions), len(sorted_layers))

        metrics["only_expand_layer_i_REPAIR_train_accs"] = torch.zeros(len(all_expansions), len(sorted_layers))
        metrics["only_expand_layer_i_REPAIR_train_losses"] = torch.zeros(len(all_expansions), len(sorted_layers))
        metrics["only_expand_layer_i_REPAIR_test_accs"] = torch.zeros(len(all_expansions), len(sorted_layers))
        metrics["only_expand_layer_i_REPAIR_test_losses"] = torch.zeros(len(all_expansions), len(sorted_layers))

        metrics["only_expand_layer_i_num_params"] = torch.zeros(len(all_expansions), len(sorted_layers))

        expansions = torch.ones(model_a.num_layers)

        for layer_idx, i in enumerate(sorted_layers):
            for exp_idx, exp in enumerate(all_expansions):
                print(
                    f"Doing expansion {layer_idx+1} of 12 on layers {','.join([str(x) for x in indices[i]])} with factor {exp}"
                )
                for index in indices[i]:
                    expansions[index] = exp
                model_a_exp = expand_model(model_a, expansions).cuda()
                model_b_exp = expand_model(model_b, expansions).cuda()
                model_b_exp_perm = permute_model(model_a_exp, model_b_exp, train_aug_loader, threshold=threshold).cuda()
                train_acc, train_loss = evaluate_two_models_merging(
                    model_a_exp, model_b_exp_perm, train_noaug_loader, 1
                )
                test_acc, test_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, test_loader, 1)
                train_acc_REPAIR, train_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                    model_a_exp, model_b_exp_perm, train_noaug_loader, train_aug_loader, 1
                )
                test_acc_REPAIR, test_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                    model_a_exp, model_b_exp_perm, test_loader, train_aug_loader, 1
                )
                num_params = get_num_params(smart_interpolate_models(model_a_exp, model_b_exp_perm), ignore_zeros=True)

                metrics["only_expand_layer_i_train_accs"][exp_idx][i] = train_acc
                metrics["only_expand_layer_i_train_losses"][exp_idx][i] = train_loss
                metrics["only_expand_layer_i_test_accs"][exp_idx][i] = test_acc
                metrics["only_expand_layer_i_test_losses"][exp_idx][i] = test_loss

                metrics["only_expand_layer_i_REPAIR_train_accs"][exp_idx][i] = train_acc_REPAIR
                metrics["only_expand_layer_i_REPAIR_train_losses"][exp_idx][i] = train_loss_REPAIR
                metrics["only_expand_layer_i_REPAIR_test_accs"][exp_idx][i] = test_acc_REPAIR
                metrics["only_expand_layer_i_REPAIR_test_losses"][exp_idx][i] = test_loss_REPAIR

                metrics["only_expand_layer_i_num_params"][exp_idx][i] = num_params

                print(f"Layer set {i+1}, expansion {exp}: {test_acc=}, {test_acc_REPAIR=}")
                save_evaluation_checkpoint(metrics, filepath, csv=False)
                metrics = load_file(filepath)  # necessary because of a safetensors bug

    return metrics


def experiment_d(model_name_a: str, model_name_b: str = None, interpolation_steps=21, thresholds=None):
    """
    Performs adaptive buffer assignment from 0.1 to 1.0
    """
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="adaptive")
    filepath = os.path.join(evaluations_dir, f"adaptive-{model_name_a}{variant_b}.safetensors")

    if os.path.exists(filepath):
        metrics = load_file(filepath)
        print(f"📤 Loaded saved metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        metrics = {"alphas": torch.linspace(0.0, 1.0, interpolation_steps)}

    train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0]

    for threshold in thresholds:
        if f"adaptive_merging_REPAIR_{threshold:g}_test_accs" not in metrics.keys():
            print(f"Collecting adaptive merging metrics ({threshold:g}) ...")
            model_a = load_model(model_name_a).cuda()
            model_b = load_model(model_name_b).cuda()

            num_params_default = get_num_params(model_a)

            model_a = expand_model(model_a, 2).cuda()
            model_b = expand_model(model_b, 2).cuda()
            model_b_perm = permute_model(
                reference_model=model_a, model=model_b, loader=train_aug_loader, threshold=threshold
            )
            (
                metrics[f"adaptive_merging_{threshold:g}_used_neurons_absolute"],
                metrics[f"adaptive_merging_{threshold:g}_used_neurons_relative"],
            ) = get_used_buffer_neurons(model_b_perm)
            (
                metrics[f"adaptive_merging_{threshold:g}_train_accs"],
                metrics[f"adaptive_merging_{threshold:g}_train_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, train_noaug_loader, interpolation_steps)
            (
                metrics[f"adaptive_merging_{threshold:g}_test_accs"],
                metrics[f"adaptive_merging_{threshold:g}_test_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, test_loader, interpolation_steps)
            print(f"Midpoint test acc: {metrics[f'adaptive_merging_{threshold:g}_test_accs'][10]}")

            print(f"Collecting adaptive merging + REPAIR metrics ({threshold:g}) ...")
            (
                metrics[f"adaptive_merging_REPAIR_{threshold:g}_train_accs"],
                metrics[f"adaptive_merging_REPAIR_{threshold:g}_train_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, train_noaug_loader, train_aug_loader, interpolation_steps
            )
            (
                metrics[f"adaptive_merging_REPAIR_{threshold:g}_test_accs"],
                metrics[f"adaptive_merging_REPAIR_{threshold:g}_test_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, test_loader, train_aug_loader, interpolation_steps
            )
            print(f"Midpoint test acc: {metrics[f'adaptive_merging_REPAIR_{threshold:g}_test_accs'][10]}")

            num_params_adaptive = get_num_params(interpolate_models(model_a, model_b_perm), ignore_zeros=True)
            metrics[f"adaptive_merging_{threshold:g}_param_increase"] = torch.Tensor(
                [num_params_adaptive / num_params_default]
            )

            save_evaluation_checkpoint(metrics, filepath)
            metrics = load_file(filepath)  # necessary because of a safetensors bug

    return metrics


def get_used_buffer_neurons(model):
    """
    From a model permuted with adaptive buffer assignment, fetches the number of
    buffer neurons per layer that were actually used.
    :param model:
    :return:
    """
    num_layers = model.num_layers
    result_absolute = torch.zeros(num_layers)
    result_relative = torch.zeros(num_layers)

    if isinstance(model, MLP):
        i = 0
        for layer in model.classifier:
            if isinstance(layer, torch.nn.BatchNorm1d):
                # width = len(layer.is_buffer)
                original_width = (~layer.is_buffer).sum().item()
                used_neurons_absolute = layer.is_buffer[:original_width].sum().item()
                used_neurons_relative = used_neurons_absolute / original_width
                result_absolute[i] = used_neurons_absolute
                result_relative[i] = used_neurons_relative
                i += 1
    elif isinstance(model, VGG):
        i = 0
        for layer in model.features:
            if isinstance(layer, torch.nn.BatchNorm2d):
                # width = len(layer.is_buffer)
                original_width = (~layer.is_buffer).sum().item()
                used_neurons_absolute = layer.is_buffer[:original_width].sum().item()
                used_neurons_relative = used_neurons_absolute / original_width
                result_absolute[i] = used_neurons_absolute
                result_relative[i] = used_neurons_relative
                i += 1
    elif isinstance(model, ResNet18):
        layers = [
            model.bn1,
            model.layer1[0].bn1,
            model.layer1[0].bn2,
            model.layer1[1].bn1,
            model.layer1[1].bn2,
            model.layer2[0].bn1,
            model.layer2[0].bn2,
            model.layer2[1].bn1,
            model.layer2[1].bn2,
            model.layer3[0].bn1,
            model.layer3[0].bn2,
            model.layer3[1].bn1,
            model.layer3[1].bn2,
            model.layer4[0].bn1,
            model.layer4[0].bn2,
            model.layer4[1].bn1,
            model.layer4[1].bn2,
        ]
        for i, layer in enumerate(layers):
            original_width = (~layer.is_buffer).sum().item()
            used_neurons_absolute = layer.is_buffer[:original_width].sum().item()
            used_neurons_relative = used_neurons_absolute / original_width
            result_absolute[i] = used_neurons_absolute
            result_relative[i] = used_neurons_relative

    print(result_absolute)
    print(result_relative)
    return result_absolute, result_relative


def experiment_r(model_name_a: str, model_name_b: str = None, interpolation_steps=21, thresholds=None):
    """
    Pulls apart neurons randomly or pulls apart lowest selected corrs
    """
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="random-or-smallest")
    filepath = os.path.join(evaluations_dir, f"random-or-smallest-{model_name_a}{variant_b}.safetensors")

    if os.path.exists(filepath):
        metrics = load_file(filepath)
        print(f"📤 Loaded saved metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        metrics = {"alphas": torch.linspace(0.0, 1.0, interpolation_steps)}

    train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

    for k in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
        if f"pull_apart_randomly_merging_REPAIR_{k:g}_test_accs" not in metrics.keys():
            print(f"Collecting pulling apart merging metrics ({k:g}) ...")
            model_a = load_model(model_name_a).cuda()
            model_b = load_model(model_name_b).cuda()

            model_a = expand_model(model_a, k).cuda()
            model_b = expand_model(model_b, k).cuda()
            model_b_perm = permute_model(reference_model=model_a, model=model_b, loader=train_aug_loader, threshold=-1)
            (
                metrics[f"pull_apart_randomly_merging_{k:g}_train_accs"],
                metrics[f"pull_apart_randomly_merging_{k:g}_train_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, train_noaug_loader, interpolation_steps)
            (
                metrics[f"pull_apart_randomly_merging_{k:g}_test_accs"],
                metrics[f"pull_apart_randomly_merging_{k:g}_test_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, test_loader, interpolation_steps)
            print(f"Midpoint test acc: {metrics[f'pull_apart_randomly_merging_{k:g}_test_accs'][10]}")

            print(f"Collecting pull_apart_randomly merging + REPAIR metrics ({k:g}) ...")
            (
                metrics[f"pull_apart_randomly_merging_REPAIR_{k:g}_train_accs"],
                metrics[f"pull_apart_randomly_merging_REPAIR_{k:g}_train_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, train_noaug_loader, train_aug_loader, interpolation_steps
            )
            (
                metrics[f"pull_apart_randomly_merging_REPAIR_{k:g}_test_accs"],
                metrics[f"pull_apart_randomly_merging_REPAIR_{k:g}_test_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, test_loader, train_aug_loader, interpolation_steps
            )
            print(f"Midpoint test acc: {metrics[f'pull_apart_randomly_merging_REPAIR_{k:g}_test_accs'][10]}")

            save_evaluation_checkpoint(metrics, filepath)
            metrics = load_file(filepath)  # necessary because of a safetensors bug

            model_b_perm = permute_model(reference_model=model_a, model=model_b, loader=train_aug_loader, threshold=-2)
            (
                metrics[f"pull_apart_smallest_merging_{k:g}_train_accs"],
                metrics[f"pull_apart_smallest_merging_{k:g}_train_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, train_noaug_loader, interpolation_steps)
            (
                metrics[f"pull_apart_smallest_merging_{k:g}_test_accs"],
                metrics[f"pull_apart_smallest_merging_{k:g}_test_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, test_loader, interpolation_steps)
            print(f"Midpoint test acc: {metrics[f'pull_apart_smallest_merging_{k:g}_test_accs'][10]}")

            print(f"Collecting pull_apart_smallest merging + REPAIR metrics ({k:g}) ...")
            (
                metrics[f"pull_apart_smallest_merging_REPAIR_{k:g}_train_accs"],
                metrics[f"pull_apart_smallest_merging_REPAIR_{k:g}_train_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, train_noaug_loader, train_aug_loader, interpolation_steps
            )
            (
                metrics[f"pull_apart_smallest_merging_REPAIR_{k:g}_test_accs"],
                metrics[f"pull_apart_smallest_merging_REPAIR_{k:g}_test_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, test_loader, train_aug_loader, interpolation_steps
            )
            print(f"Midpoint test acc: {metrics[f'pull_apart_smallest_merging_REPAIR_{k:g}_test_accs'][10]}")

            save_evaluation_checkpoint(metrics, filepath)
            metrics = load_file(filepath)  # necessary because of a safetensors bug

    return metrics


def experiment_k(
    model_name_a: str, model_name_b: str = None, interpolation_steps=21, after_repair=False, just_use_accs=False
):
    """
    Ensembles parts of the model on an order determined by experiment b
    """
    if model_name_b is None:
        model_name_b = f"{model_name_a}-b"
        model_name_a = f"{model_name_a}-a"

    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="smart_order")
    filepath = os.path.join(
        evaluations_dir,
        f"smart_order_{'after' if after_repair else 'before'}{'_justaccs' if just_use_accs else ''}-{model_name_a}{variant_b}.safetensors",
    )

    if os.path.exists(filepath):
        metrics = load_file(filepath)
        print(f"📤 Loaded saved metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        metrics = {"alphas": torch.linspace(0.0, 1.0, interpolation_steps)}

    train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

    order = get_order(
        dataset_a,
        model_type_a,
        size_a,
        width_a,
        f"{variant_a}{variant_b}",
        after_repair=after_repair,
        just_use_accs=just_use_accs,
    )
    print(f"{order=}")

    indices = {
        0: [1],
        1: [3],
        2: [5],
        3: [7],
        4: [9],
        5: [11],
        6: [13],
        7: [15],
        8: [0, 2, 4],
        9: [6, 8],
        10: [10, 12],
        11: [14, 16],
    }

    model_a = load_model(model_name_a).cuda()
    num_params_default = get_num_params(model_a)
    width_expansion = torch.ones(model_a.num_layers)
    for k, idxs in enumerate(order):
        if f"smart_order_merging_REPAIR_{k:g}_test_accs" not in metrics.keys():
            print(f"Collecting smart_order metrics ({k:g}) ...")
            model_a = load_model(model_name_a).cuda()
            model_b = load_model(model_name_b).cuda()

            if model_type_a == "ResNet":
                idxs = indices[idxs.item()]
            else:
                idxs = [idxs]

            for idx in idxs:
                width_expansion[idx] = 2

            print(width_expansion)
            model_a = expand_model(model_a, width_expansion).cuda()
            model_b = expand_model(model_b, width_expansion).cuda()
            model_b_perm = permute_model(reference_model=model_a, model=model_b, loader=train_aug_loader)
            (
                metrics[f"smart_order_merging_{k:g}_train_accs"],
                metrics[f"smart_order_merging_{k:g}_train_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, train_noaug_loader, interpolation_steps)
            (
                metrics[f"smart_order_merging_{k:g}_test_accs"],
                metrics[f"smart_order_merging_{k:g}_test_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, test_loader, interpolation_steps)
            print(f"Midpoint test acc: {metrics[f'smart_order_merging_{k:g}_test_accs'][10]}")

            print(f"Collecting smart_order merging + REPAIR metrics ({k:g}) ...")
            (
                metrics[f"smart_order_merging_REPAIR_{k:g}_train_accs"],
                metrics[f"smart_order_merging_REPAIR_{k:g}_train_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, train_noaug_loader, train_aug_loader, interpolation_steps
            )
            (
                metrics[f"smart_order_merging_REPAIR_{k:g}_test_accs"],
                metrics[f"smart_order_merging_REPAIR_{k:g}_test_losses"],
            ) = evaluate_two_models_merging_REPAIR(
                model_a, model_b_perm, test_loader, train_aug_loader, interpolation_steps
            )
            print(f"Midpoint test acc: {metrics[f'smart_order_merging_REPAIR_{k:g}_test_accs'][10]}")

            num_params = get_num_params(interpolate_models(model_a, model_b_perm), ignore_zeros=True)
            metrics[f"smart_order_merging_{k:g}_param_increase"] = torch.Tensor([num_params / num_params_default])
            print(f"Param increase: {num_params / num_params_default}")

            save_evaluation_checkpoint(metrics, filepath)
            metrics = load_file(filepath)  # necessary because of a safetensors bug

    return metrics


def get_order(dataset, architecture, size, width, variants, after_repair, just_use_accs=False):
    """
    Fetches the order of expansion based on experiment b results
    """

    metrics = load_file(
        os.path.join(
            get_evaluations_dir("experiment_b"),
            f"experiment-b-{dataset}-{architecture}{size}-bn-{width}x-{variants}.safetensors",
        )
    )
    metrics_default = load_file(
        os.path.join(get_evaluations_dir("two_models"), f"{dataset}-{architecture}{size}-bn-{width}x-ab.safetensors")
    )

    accs = torch.flip(metrics[f"only_expand_layer_i{'_REPAIR' if after_repair else ''}_test_accs"], dims=[0])[:, :12]
    # losses = torch.flip(metrics[f"only_expand_layer_i{'_REPAIR' if after_repair else ''}_test_losses"], dims=[0])[:, :12]
    params = (torch.flip(metrics["only_expand_layer_i_num_params"], dims=[0]) / metrics["default_num_params"])[
        :, :12
    ] - 1

    acc_endpoints = metrics_default["ensembling_test_accs"][[0, -1]].mean()
    acc_merging = metrics_default[f"merging{'_REPAIR' if after_repair else ''}_test_accs"][10]
    acc_barrier_reduction = (accs - acc_merging) / (acc_endpoints - acc_merging)

    # loss_endpoints = metrics_default["ensembling_test_losses"][[0, -1]].mean()
    # loss_merging = metrics_default[f"merging{'_REPAIR' if repair else ''}_test_losses"][10]
    # loss_barrier_reduction = (losses - loss_merging) / (loss_endpoints - loss_merging)

    benefits = acc_barrier_reduction[0] if just_use_accs else (acc_barrier_reduction / params)[0]
    _, order = torch.sort(benefits, descending=True)

    # we enforce ResNet layer group 15/17 to be the last one - it's just not worth expanding it ...
    if architecture == "ResNet":
        idx_position = (order == 11).nonzero(as_tuple=True)[0]
        order = torch.cat((order[:idx_position], order[idx_position + 1 :]))
        order = torch.cat((order, torch.tensor([11])))

    return order
