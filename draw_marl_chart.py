import os
import json
import re
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from scipy.ndimage import gaussian_filter1d


def discover_summary_files(summary_dir: str) -> List[Tuple[str, str]]:
    pattern = re.compile(r"^summary_(.+)\.json$")
    discovered_files = []

    for filename in os.listdir(summary_dir):
        match = pattern.match(filename)
        if match:
            model_name = match.group(1)
            file_path = os.path.join(summary_dir, filename)
            discovered_files.append((model_name, file_path))

    return discovered_files


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_metric_key_index(summary_payload: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    full_to_short = {}
    short_to_full = {}

    for full_metric_key in summary_payload:
        short_metric_key = os.path.basename(full_metric_key)
        full_to_short[full_metric_key] = short_metric_key
        short_to_full[short_metric_key] = full_metric_key

    return full_to_short, short_to_full


def extract_metric_series(summary_file: str, metric_name: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    summary_payload = load_json_file(summary_file)
    _, short_to_full = build_metric_key_index(summary_payload)

    if metric_name not in short_to_full:
        return None, None

    full_metric_key = short_to_full[metric_name]
    metric_entries = sorted(summary_payload[full_metric_key], key=lambda item: item[1])

    steps = [entry[1] for entry in metric_entries]
    values = [entry[2] for entry in metric_entries]
    return steps, values


def normalize_experiment_group_name(model_name: str) -> str:
    normalized_name = model_name.strip()
    normalized_name = re.sub(r"(?:[\-_\s]?)\d{1,3}$", "", normalized_name)
    return normalized_name


def aggregate_series_with_dispersion(
    steps_collection: List[List[float]],
    values_collection: List[np.ndarray],
    aggregation: str = "mean",
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    按 step 对多条曲线聚合，并返回：
    - 聚合后的 step 列表
    - 中心曲线（mean 或 median）
    - 标准差 std（用于阴影带）
    """
    step_to_values: Dict[float, List[float]] = {}

    for steps, values in zip(steps_collection, values_collection):
        for step, value in zip(steps, values):
            step_to_values.setdefault(step, []).append(float(value))

    merged_steps = sorted(step_to_values.keys())

    center_values = []
    std_values = []

    for step in merged_steps:
        current_values = np.array(step_to_values[step], dtype=float)

        if aggregation == "median":
            center = float(np.median(current_values))
        else:
            center = float(np.mean(current_values))

        std = float(np.std(current_values))

        center_values.append(center)
        std_values.append(std)

    return merged_steps, np.array(center_values, dtype=float), np.array(std_values, dtype=float)


def list_available_metrics(summary_files: List[Tuple[str, str]]) -> None:
    print("未找到指定指标，可用指标如下：")
    for model_name, summary_file in summary_files:
        summary_payload = load_json_file(summary_file)
        full_to_short, _ = build_metric_key_index(summary_payload)
        print(f"{model_name}: {list(full_to_short.values())}")


def apply_optional_smoothing(values: np.ndarray, enabled: bool, sigma: float) -> np.ndarray:
    if not enabled:
        return values
    return gaussian_filter1d(values, sigma=sigma)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marl-metric-plotter",
        description=(
            "Interactive visualization tool for MARL summary files with optional "
            "smoothing, grouping, and variance shading."
        ),
    )

    parser.add_argument(
        "--summary-dir",
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing summary_<MODEL>.json files.",
    )
    parser.add_argument(
        "--metric-name",
        "--metric",
        "-m",
        required=True,
        help="Metric name to visualize, e.g. eval_win_rate.",
    )
    parser.add_argument(
        "--initial-smoothing-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma. Default: 1.5",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=11.0,
        help="Figure width. Default: 11.0",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=7.0,
        help="Figure height. Default: 7.0",
    )
    parser.add_argument(
        "--group-band-alpha",
        type=float,
        default=0.18,
        help="Alpha of grouping variance shading. Default: 0.18",
    )

    return parser


def run_interactive_metric_dashboard(
    summary_dir: str,
    metric_name: str,
    fixed_color_mapping: Optional[Dict[str, str]] = None,
    smoothing_sigma: float = 1.5,
    figure_size: Tuple[float, float] = (11.0, 7.0),
    group_band_alpha: float = 0.18,
) -> None:
    summary_files = discover_summary_files(summary_dir)
    if not summary_files:
        print(f"在路径 '{summary_dir}' 下未找到任何 'summary_<MODEL>.json' 文件。")
        return

    series_by_model: Dict[str, np.ndarray] = {}
    steps_by_model: Dict[str, List[float]] = {}

    for model_name, summary_file in summary_files:
        steps, values = extract_metric_series(summary_file, metric_name)
        if steps is not None and values is not None:
            steps_by_model[model_name] = steps
            series_by_model[model_name] = np.array(values, dtype=float)

    if not series_by_model:
        list_available_metrics(summary_files)
        return

    grouped_models: Dict[str, List[str]] = {}
    for model_name in series_by_model.keys():
        group_name = normalize_experiment_group_name(model_name)
        grouped_models.setdefault(group_name, []).append(model_name)

    fig, axis = plt.subplots(figsize=figure_size)
    plt.subplots_adjust(left=0.28, right=0.98, bottom=0.10, top=0.92)

    raw_line_handles: Dict[str, plt.Line2D] = {}
    for model_name in series_by_model:
        line_color = fixed_color_mapping.get(model_name) if fixed_color_mapping else None
        line_handle, = axis.plot(
            steps_by_model[model_name],
            series_by_model[model_name],
            label=model_name,
            color=line_color,
            alpha=0.9,
        )
        raw_line_handles[model_name] = line_handle

    group_line_handles: Dict[str, plt.Line2D] = {}
    group_band_handles: Dict[str, object] = {}

    for group_name, member_models in grouped_models.items():
        if len(member_models) == 1:
            group_line_handles[group_name] = raw_line_handles[member_models[0]]
            group_band_handles[group_name] = None
        else:
            aggregated_steps, aggregated_center, _ = aggregate_series_with_dispersion(
                [steps_by_model[m] for m in member_models],
                [series_by_model[m] for m in member_models],
                aggregation="mean",
            )

            group_line, = axis.plot(
                aggregated_steps,
                aggregated_center,
                label=f"{group_name} (mean)",
                linestyle="-",
                linewidth=2.0,
                alpha=0.95,
                visible=False,
            )
            group_line_handles[group_name] = group_line

            group_band = axis.fill_between(
                aggregated_steps,
                aggregated_center,
                aggregated_center,
                alpha=group_band_alpha,
                visible=False,
                color=group_line.get_color(),
            )
            group_band_handles[group_name] = group_band

    axis.set_title(metric_name, pad=10)
    axis.set_xlabel("Environment steps")
    axis.set_ylabel(metric_name)
    axis.grid(True, linestyle="--", alpha=0.5)

    plot_state = {
        "smoothing_enabled": False,
        "smoothing_sigma": smoothing_sigma,
        "group_aggregation_mode": "off",  # off / mean / median
    }

    model_labels = list(series_by_model.keys())
    default_visibility = [True] * len(model_labels)

    checkbox_axis = plt.axes([0.02, 0.35, 0.22, 0.55])
    visibility_checkboxes = CheckButtons(checkbox_axis, model_labels, default_visibility)

    def remove_existing_group_band(group_name: str) -> None:
        existing_band = group_band_handles.get(group_name)
        if existing_band is not None:
            try:
                existing_band.remove()
            except Exception:
                pass
            group_band_handles[group_name] = None

    def create_group_band(group_name: str, x: List[float], y_center: np.ndarray, y_std: np.ndarray) -> None:
        remove_existing_group_band(group_name)

        lower = y_center - y_std
        upper = y_center + y_std

        line_handle = group_line_handles[group_name]
        band = axis.fill_between(
            x,
            lower,
            upper,
            alpha=group_band_alpha,
            visible=True,
            color=line_handle.get_color(),
        )
        group_band_handles[group_name] = band

    def hide_group_band(group_name: str) -> None:
        existing_band = group_band_handles.get(group_name)
        if existing_band is not None:
            try:
                existing_band.set_visible(False)
            except Exception:
                pass

    def refresh_legend() -> None:
        legend_handles = []
        legend_labels = []

        for model_name, line_handle in raw_line_handles.items():
            if line_handle.get_visible():
                legend_handles.append(line_handle)
                legend_labels.append(model_name)

        for group_name, group_line_handle in group_line_handles.items():
            if group_line_handle.get_visible():
                legend_handles.append(group_line_handle)
                legend_labels.append(group_line_handle.get_label())

        if legend_handles:
            axis.legend(legend_handles, legend_labels, loc="upper left", fontsize=9)
        else:
            axis.legend([], [], loc="upper left")

        fig.canvas.draw_idle()

    def update_raw_curves() -> None:
        for model_name in series_by_model:
            original_values = series_by_model[model_name]
            displayed_values = apply_optional_smoothing(
                original_values,
                enabled=plot_state["smoothing_enabled"],
                sigma=plot_state["smoothing_sigma"],
            )
            raw_line_handles[model_name].set_ydata(displayed_values)

    def recompute_grouped_curves() -> None:
        mode = plot_state["group_aggregation_mode"]

        if mode == "off":
            for group_name, group_line_handle in group_line_handles.items():
                members = grouped_models[group_name]
                if len(members) > 1:
                    group_line_handle.set_visible(False)
                    hide_group_band(group_name)
                else:
                    single_model = members[0]
                    group_line_handle.set_visible(raw_line_handles[single_model].get_visible())
                    group_line_handle.set_xdata(raw_line_handles[single_model].get_xdata())
                    group_line_handle.set_ydata(raw_line_handles[single_model].get_ydata())
                    hide_group_band(group_name)

            for index, model_name in enumerate(model_labels):
                raw_line_handles[model_name].set_visible(visibility_checkboxes.get_status()[index])

            refresh_legend()
            return

        for model_name in model_labels:
            raw_line_handles[model_name].set_visible(False)

        for group_name, member_models in grouped_models.items():
            selected_models = [
                model_name
                for model_name in member_models
                if visibility_checkboxes.get_status()[model_labels.index(model_name)]
            ]

            if len(selected_models) == 0:
                group_line_handles[group_name].set_visible(False)
                hide_group_band(group_name)
                continue

            if len(selected_models) == 1:
                selected_model = selected_models[0]
                aggregated_steps = steps_by_model[selected_model]
                aggregated_center = series_by_model[selected_model].copy()
                aggregated_std = np.zeros_like(aggregated_center, dtype=float)
            else:
                aggregated_steps, aggregated_center, aggregated_std = aggregate_series_with_dispersion(
                    [steps_by_model[m] for m in selected_models],
                    [series_by_model[m] for m in selected_models],
                    aggregation=mode,
                )

            aggregated_center = np.array(aggregated_center, dtype=float)
            aggregated_std = np.array(aggregated_std, dtype=float)

            if plot_state["smoothing_enabled"]:
                aggregated_center = gaussian_filter1d(aggregated_center, sigma=plot_state["smoothing_sigma"])
                aggregated_std = gaussian_filter1d(aggregated_std, sigma=plot_state["smoothing_sigma"])

            group_line_handle = group_line_handles[group_name]
            group_line_handle.set_xdata(aggregated_steps)
            group_line_handle.set_ydata(aggregated_center)
            group_line_handle.set_label(f"{group_name} ({mode})")
            group_line_handle.set_visible(True)

            if len(selected_models) >= 2:
                create_group_band(group_name, aggregated_steps, aggregated_center, aggregated_std)
            else:
                hide_group_band(group_name)

        refresh_legend()

    def handle_checkbox_click(clicked_label: str) -> None:
        if plot_state["group_aggregation_mode"] == "off":
            current_visibility = raw_line_handles[clicked_label].get_visible()
            raw_line_handles[clicked_label].set_visible(not current_visibility)
        else:
            recompute_grouped_curves()

        refresh_legend()
        plt.draw()

    visibility_checkboxes.on_clicked(handle_checkbox_click)

    select_all_axis = plt.axes([0.02, 0.27, 0.105, 0.06])
    clear_all_axis = plt.axes([0.135, 0.27, 0.105, 0.06])
    select_all_button = Button(select_all_axis, "select all")
    clear_all_button = Button(clear_all_axis, "clear all")

    def handle_select_all(_event) -> None:
        for index, checked in enumerate(visibility_checkboxes.get_status()):
            if not checked:
                visibility_checkboxes.set_active(index)

        for line_handle in raw_line_handles.values():
            line_handle.set_visible(True)

        if plot_state["group_aggregation_mode"] == "off":
            for group_name, group_line_handle in group_line_handles.items():
                if len(grouped_models[group_name]) > 1:
                    group_line_handle.set_visible(False)
                    hide_group_band(group_name)

        refresh_legend()

    def handle_clear_all(_event) -> None:
        for index, checked in enumerate(visibility_checkboxes.get_status()):
            if checked:
                visibility_checkboxes.set_active(index)

        for line_handle in raw_line_handles.values():
            line_handle.set_visible(False)

        for group_name, group_line_handle in group_line_handles.items():
            if len(grouped_models[group_name]) > 1:
                group_line_handle.set_visible(False)
            hide_group_band(group_name)

        refresh_legend()

    select_all_button.on_clicked(handle_select_all)
    clear_all_button.on_clicked(handle_clear_all)

    smoothing_axis = plt.axes([0.02, 0.17, 0.22, 0.06])
    smoothing_button = Button(smoothing_axis, "smoothing: off")

    def handle_smoothing_toggle(_event) -> None:
        plot_state["smoothing_enabled"] = not plot_state["smoothing_enabled"]
        smoothing_button.label.set_text(
            f"smoothing: {'on' if plot_state['smoothing_enabled'] else 'off'}"
        )

        update_raw_curves()
        recompute_grouped_curves()
        plt.draw()

    smoothing_button.on_clicked(handle_smoothing_toggle)

    aggregation_axis = plt.axes([0.02, 0.08, 0.22, 0.06])
    aggregation_button = Button(aggregation_axis, "grouping: off")

    def handle_grouping_toggle(_event) -> None:
        current_mode = plot_state["group_aggregation_mode"]

        if current_mode == "off":
            plot_state["group_aggregation_mode"] = "mean"
        elif current_mode == "mean":
            plot_state["group_aggregation_mode"] = "median"
        else:
            plot_state["group_aggregation_mode"] = "off"

        aggregation_button.label.set_text(
            f"grouping: {plot_state['group_aggregation_mode']}"
        )
        recompute_grouped_curves()
        plt.draw()

    aggregation_button.on_clicked(handle_grouping_toggle)

    refresh_legend()
    plt.show()


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    fixed_color_mapping: Dict[str, str] = {
        # "MAT": "tab:blue",
        # "TMO": "tab:orange",
    }

    run_interactive_metric_dashboard(
        summary_dir=args.summary_dir,
        metric_name=args.metric_name,
        fixed_color_mapping=fixed_color_mapping,
        smoothing_sigma=args.initial_smoothing_sigma,
        figure_size=(args.figure_width, args.figure_height),
        group_band_alpha=args.group_band_alpha,
    )


if __name__ == "__main__":
    main()