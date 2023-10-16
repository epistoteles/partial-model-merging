import torch
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
import os
from codecarbon import track_emissions
from itertools import product
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
    get_num_params,
)


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
def evaluate_two_models(model_name_a: str, model_name_b: str, interpolation_steps: int = 21):
    """
    Evaluates two models in terms of accuracy and loss with different combination techniques (and saves the result)
    :param model_name_a: the name of the first (reference) model checkpoint
    :param model_name_b: the name of the second model checkpoint.
    :param interpolation_steps: number of interpolation steps between the models, i.e. 21 = 1.0, 0.95, 0.9, 0.85, ...
    :return: TODO
    """
    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert dataset_a == dataset_b
    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="two_models")
    filepath = os.path.join(evaluations_dir, f"{model_name_a}{variant_b}.csv")

    if os.path.exists(filepath.replace(".csv", ".safetensors")):
        metrics = load_file(filepath.replace(".csv", ".safetensors"))
        print(f"📤 Loaded saved metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        model_a = load_model(model_name_a).cuda()
        model_b = load_model(model_name_b).cuda()

        train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

        metrics = {"alphas": torch.linspace(0.0, 1.0, interpolation_steps)}

        print("Collecting ensembling metrics ...")
        metrics["ensembling_train_accs"], metrics["ensembling_train_losses"] = evaluate_two_models_ensembling(
            model_a, model_b, train_noaug_loader, interpolation_steps
        )
        metrics["ensembling_test_accs"], metrics["ensembling_test_losses"] = evaluate_two_models_ensembling(
            model_a, model_b, test_loader, interpolation_steps
        )

        print("Collecting naive merging metrics ...")
        metrics["naive_train_accs"], metrics["naive_train_losses"] = evaluate_two_models_merging(
            model_a, model_b, train_noaug_loader, interpolation_steps
        )
        metrics["naive_test_accs"], metrics["naive_test_losses"] = evaluate_two_models_merging(
            model_a, model_b, test_loader, interpolation_steps
        )

        print("Collecting permuted merging metrics ...")
        model_b_perm = permute_model(reference_model=model_a, model=model_b, loader=train_aug_loader)
        metrics["merging_train_accs"], metrics["merging_train_losses"] = evaluate_two_models_merging(
            model_a, model_b_perm, train_noaug_loader, interpolation_steps
        )
        metrics["merging_test_accs"], metrics["merging_test_losses"] = evaluate_two_models_merging(
            model_a, model_b_perm, test_loader, interpolation_steps
        )

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

        save_evaluation_checkpoint(metrics, filepath)

        for k in [1.1, 1.5, 1.8, 2.0]:
            print(f"Collecting partial merging metrics ({k}) ...")
            model_a = load_model(model_name_a).cuda()
            model_b = load_model(model_name_b).cuda()

            model_a = expand_model(model_a, k).cuda()
            model_b = expand_model(model_b, k).cuda()
            model_b_perm = permute_model(reference_model=model_a, model=model_b, loader=train_aug_loader)
            (
                metrics[f"partial_merging_{k}_train_accs"],
                metrics[f"partial_merging_{k}_train_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, train_noaug_loader, interpolation_steps)
            (
                metrics[f"partial_merging_{k}_test_accs"],
                metrics[f"partial_merging_{k}_test_losses"],
            ) = evaluate_two_models_merging(model_a, model_b_perm, test_loader, interpolation_steps)

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

            save_evaluation_checkpoint(metrics, filepath)
            print(f"📥 Metrics saved for {model_name_a}{variant_b} as .csv and .safetensors")

    return metrics


def save_evaluation_checkpoint(metrics, filepath):
    save_file(metrics, filename=filepath.replace(".csv", ".safetensors"))
    np.savetxt(
        filepath,
        np.asarray([list(metrics.keys()), *list(zip(*[ensure_numpy(x) for x in metrics.values()]))]),
        delimiter=",",
        fmt="%s",
    )
    print("📥 Metrics checkpointed as .csv and .safetensors")


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
    model_a.eval()
    model_b.eval()

    accs = []
    losses = []

    for alpha in torch.linspace(0.0, 1.0, interpolation_steps) if interpolation_steps > 1 else [0.5]:
        model_merged = smart_interpolate_models(model_a, model_b, alpha)
        acc, loss = get_acc_and_loss(model_merged, loader)
        accs.append(acc)
        losses.append(loss)

    return torch.FloatTensor(accs), torch.FloatTensor(losses)


def evaluate_two_models_merging_REPAIR(
    model_a: torch.nn.Module, model_b: torch.nn.Module, loader, repair_loader, interpolation_steps: int = 21
):
    model_a.eval()
    model_b.eval()

    accs = []
    losses = []

    for alpha in torch.linspace(0.0, 1.0, interpolation_steps) if interpolation_steps > 1 else [0.5]:
        model_merged = smart_interpolate_models(model_a, model_b, alpha)
        if model_a.bn:  # there is no special REPAIR for models with bn (is there?)
            reset_bn_stats(model_merged.cuda(), repair_loader)
            model_repaired = model_merged
        else:
            model_repaired = repair(model_merged, model_a, model_b, repair_loader, alpha)
        acc, loss = get_acc_and_loss(model_repaired, loader)
        accs.append(acc)
        losses.append(loss)

    return torch.FloatTensor(accs), torch.FloatTensor(losses)


def experiment_a(model_name_a: str, model_name_b: str):
    """
    Conducts leave-one-out experiments with full merging vs. ensembling and saves the results
    """
    dataset_a, model_type_a, size_a, batch_norm_a, width_a, variant_a = parse_model_name(model_name_a)
    dataset_b, model_type_b, size_b, batch_norm_b, width_b, variant_b = parse_model_name(model_name_b)

    assert dataset_a == dataset_b
    assert model_type_a == model_type_b
    assert size_a == size_b
    assert batch_norm_a == batch_norm_b
    assert width_a == width_b  # not strictly necessary, but always the case in our experiments

    evaluations_dir = get_evaluations_dir(subdir="experiment_a")
    filepath = os.path.join(evaluations_dir, f"experiment-a-{model_name_a}{variant_b}.csv")

    if os.path.exists(filepath.replace(".csv", ".safetensors")):
        metrics = load_file(filepath.replace(".csv", ".safetensors"))
        print(f"📤 Loaded saved leave-one-out metrics for {model_name_a}{variant_b} from .safetensors")
    else:
        model_a = load_model(model_name_a).cuda()
        model_b = load_model(model_name_b).cuda()

        train_aug_loader, train_noaug_loader, test_loader = get_loaders(dataset_a)

        num_layers = model_a.num_layers
        default_num_params = get_num_params(model_a)

        metrics = {"layers": torch.range(1, num_layers)}

        duo_metrics = evaluate_two_models(model_name_a, model_name_b)
        midway_index = ((duo_metrics["alphas"] == 0.5).nonzero(as_tuple=True)[0]).item()

        for metric, split in product(["accs", "losses"], ["train", "test"]):
            full_ensembling = duo_metrics[f"ensembling_{split}_{metric}"][midway_index].item()
            metrics[f"full_ensembling_{split}_{metric}"] = torch.Tensor([full_ensembling]).repeat(num_layers)
            full_merging = duo_metrics[f"merging_{split}_{metric}"][midway_index].item()
            metrics[f"full_merging_{split}_{metric}"] = torch.Tensor([full_merging]).repeat(num_layers)
            full_ensembling_REPAIR = duo_metrics[f"ensembling_REPAIR_{split}_{metric}"][midway_index].item()
            metrics[f"full_ensembling_REPAIR_{split}_{metric}"] = torch.Tensor([full_ensembling_REPAIR]).repeat(
                num_layers
            )
            full_merging_REPAIR = duo_metrics[f"merging_REPAIR_{split}_{metric}"][midway_index].item()
            metrics[f"full_merging_REPAIR_{split}_{metric}"] = torch.Tensor([full_merging_REPAIR]).repeat(num_layers)

        metrics["default_num_params"] = torch.Tensor([default_num_params]).repeat(num_layers)

        # only ensembling layer i

        train_accs = []
        train_losses = []
        test_accs = []
        test_losses = []

        train_accs_REPAIR = []
        train_losses_REPAIR = []
        test_accs_REPAIR = []
        test_losses_REPAIR = []

        nums_params = []

        for i in range(model_a.num_layers):
            print(f"Only ensembling layer {i+1} of {num_layers}")
            expansions = torch.ones(model_a.num_layers)
            expansions[i] = 2.0
            model_a_exp = expand_model(model_a, expansions).cuda()
            model_b_exp = expand_model(model_b, expansions).cuda()
            model_b_exp_perm = permute_model(model_a_exp, model_b_exp, train_aug_loader).cuda()
            train_acc, train_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, train_noaug_loader, 1)
            test_acc, test_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, test_loader, 1)
            train_acc_REPAIR, train_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                model_a_exp, model_b_exp_perm, train_noaug_loader, train_aug_loader, 1
            )
            test_acc_REPAIR, test_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                model_a_exp, model_b_exp_perm, test_loader, train_aug_loader, 1
            )
            num_params = get_num_params(model_a_exp)

            train_accs.append(train_acc.item())
            train_losses.append(train_loss.item())
            test_accs.append(test_acc.item())
            test_losses.append(test_loss.item())

            train_accs_REPAIR.append(train_acc_REPAIR.item())
            train_losses_REPAIR.append(train_loss_REPAIR.item())
            test_accs_REPAIR.append(test_acc_REPAIR.item())
            test_losses_REPAIR.append(test_loss_REPAIR.item())

            nums_params.append(num_params)

        metrics["only_ensemble_i_train_accs"] = torch.FloatTensor(train_accs)
        metrics["only_ensemble_i_train_losses"] = torch.FloatTensor(train_losses)
        metrics["only_ensemble_i_test_accs"] = torch.FloatTensor(test_accs)
        metrics["only_ensemble_i_test_losses"] = torch.FloatTensor(test_losses)

        metrics["only_ensemble_i_REPAIR_train_accs"] = torch.FloatTensor(train_accs_REPAIR)
        metrics["only_ensemble_i_REPAIR_train_losses"] = torch.FloatTensor(train_losses_REPAIR)
        metrics["only_ensemble_i_REPAIR_test_accs"] = torch.FloatTensor(test_accs_REPAIR)
        metrics["only_ensemble_i_REPAIR_test_losses"] = torch.FloatTensor(test_losses_REPAIR)

        metrics["only_ensemble_i_num_params"] = torch.FloatTensor(nums_params)

        # only merging layer i
        train_accs = []
        train_losses = []
        test_accs = []
        test_losses = []

        train_accs_REPAIR = []
        train_losses_REPAIR = []
        test_accs_REPAIR = []
        test_losses_REPAIR = []

        nums_params = []

        for i in range(model_a.num_layers):
            print(f"Only merging layer {i+1} of {num_layers}")
            expansions = torch.ones(model_a.num_layers) * 2
            expansions[i] = 1.0
            model_a_exp = expand_model(model_a, expansions).cuda()
            model_b_exp = expand_model(model_b, expansions).cuda()
            model_b_exp_perm = permute_model(model_a_exp, model_b_exp, train_aug_loader).cuda()
            train_acc, train_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, train_noaug_loader, 1)
            test_acc, test_loss = evaluate_two_models_merging(model_a_exp, model_b_exp_perm, test_loader, 1)
            train_acc_REPAIR, train_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                model_a_exp, model_b_exp_perm, train_noaug_loader, train_aug_loader, 1
            )
            test_acc_REPAIR, test_loss_REPAIR = evaluate_two_models_merging_REPAIR(
                model_a_exp, model_b_exp_perm, test_loader, train_aug_loader, 1
            )
            num_params = get_num_params(model_a_exp)

            train_accs.append(train_acc.item())
            train_losses.append(train_loss.item())
            test_accs.append(test_acc.item())
            test_losses.append(test_loss.item())

            train_accs_REPAIR.append(train_acc_REPAIR.item())
            train_losses_REPAIR.append(train_loss_REPAIR.item())
            test_accs_REPAIR.append(test_acc_REPAIR.item())
            test_losses_REPAIR.append(test_loss_REPAIR.item())

            nums_params.append(num_params)

        metrics["only_merge_i_train_accs"] = torch.FloatTensor(train_accs)
        metrics["only_merge_i_train_losses"] = torch.FloatTensor(train_losses)
        metrics["only_merge_i_test_accs"] = torch.FloatTensor(test_accs)
        metrics["only_merge_i_test_losses"] = torch.FloatTensor(test_losses)

        metrics["only_merge_i_REPAIR_train_accs"] = torch.FloatTensor(train_accs_REPAIR)
        metrics["only_merge_i_REPAIR_train_losses"] = torch.FloatTensor(train_losses_REPAIR)
        metrics["only_merge_i_REPAIR_test_accs"] = torch.FloatTensor(test_accs_REPAIR)
        metrics["only_merge_i_REPAIR_test_losses"] = torch.FloatTensor(test_losses_REPAIR)

        metrics["only_merge_i_num_params"] = torch.FloatTensor(nums_params)

        save_evaluation_checkpoint(metrics, filepath)
        print(f"📥 Saved leave-ont-out for {model_name_a}{variant_b} as .csv and .safetensors")

    return metrics
