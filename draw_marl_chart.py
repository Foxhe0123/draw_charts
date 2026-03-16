
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from scipy.ndimage import gaussian_filter1d
import re
import argparse

def find_summary_files(dir_path):
    pattern = re.compile(r'^summary_(.+)\.json$')
    files = []
    for fname in os.listdir(dir_path):
        m = pattern.match(fname)
        if m:
            model_name = m.group(1)
            full_path = os.path.join(dir_path, fname)
            files.append((model_name, full_path))
    return files


def load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_stripped_key_map(summary_dict):
    full_to_stripped = {}
    stripped_to_full = {}
    for full_key in summary_dict:
        stripped = os.path.basename(full_key)
        full_to_stripped[full_key] = stripped
        stripped_to_full[stripped] = full_key
    return full_to_stripped, stripped_to_full


def load_metric_from_summary(path, metric_key):
    data = load_summary(path)
    _, stripped_to_full = build_stripped_key_map(data)
    if metric_key not in stripped_to_full:
        return None, None
    full_key = stripped_to_full[metric_key]
    entries = sorted(data[full_key], key=lambda x: x[1])
    steps = [e[1] for e in entries]
    values = [e[2] for e in entries]
    return steps, values



def base_name(name: str) -> str:

    s = name.strip()

    s = re.sub(r'(?:[\-_\s]?)\d{1,3}$', '', s)
    return s

def aggregate_group(steps_list, values_list, mode='mean'):

    step_to_vals = {}
    for steps, vals in zip(steps_list, values_list):
        for x, y in zip(steps, vals):
            step_to_vals.setdefault(x, []).append(y)
    all_steps = sorted(step_to_vals.keys())
    if mode == 'median':
        agg_vals = [float(np.median(step_to_vals[s])) for s in all_steps]
    else:

        agg_vals = [float(np.mean(step_to_vals[s])) for s in all_steps]
    return all_steps, agg_vals


def interactive_plot(dir_path, metric_key, fixed_colors=None):
    summaries = find_summary_files(dir_path)
    if not summaries:
        print(f"在路径 '{dir_path}' 下未找到任何 'summary_<MODEL>.json' 文件。")
        return

    data = {}
    x_vals = {}
    for model_name, path in summaries:
        x, y = load_metric_from_summary(path, metric_key)
        if x is not None and y is not None:
            x_vals[model_name] = x
            data[model_name] = np.array(y, dtype=float)

    if not data:
        print(f"所有 summary 文件都未找到指标 '{metric_key}'。可用指标如下：")
        for model_name, path in summaries:
            summary = load_summary(path)
            full_to_stripped, _ = build_stripped_key_map(summary)
            print(f"{model_name}: {[v for v in full_to_stripped.values()]}")
        return


    groups = {}
    for model in data.keys():
        b = base_name(model)
        groups.setdefault(b, []).append(model)

    fig, ax = plt.subplots(figsize=(11, 7))
    plt.subplots_adjust(left=0.28, right=0.98, bottom=0.1, top=0.92)


    lines = {}
    for model in data:
        color = fixed_colors.get(model) if fixed_colors else None
        line, = ax.plot(x_vals[model], data[model], label=model, color=color, alpha=0.9)
        lines[model] = line


    group_lines = {}
    for gname, members in groups.items():
        if len(members) == 1:

            group_lines[gname] = lines[members[0]]
        else:
            steps_list = [x_vals[m] for m in members]
            values_list = [data[m] for m in members]

            gx, gy = aggregate_group(steps_list, values_list, mode='mean')
            color = None
            gline, = ax.plot(gx, gy, label=f"{gname} (group)",
                             linestyle='-', linewidth=2.0, alpha=0.95, visible=False, color=color)
            group_lines[gname] = gline

    ax.set_title(f"{metric_key}", pad=10)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric_key)
    ax.grid(True, linestyle='--', alpha=0.5)

    def refresh_legend():
        handles, labels = [], []

        for m, ln in lines.items():
            if ln.get_visible():
                handles.append(ln)
                labels.append(m)

        for g, gl in group_lines.items():
            if gl.get_visible():
                handles.append(gl)
                labels.append(gl.get_label())
        if handles:
            ax.legend(handles, labels, loc="upper left", fontsize=9)
        else:
            ax.legend([], [], loc="upper left")
        fig.canvas.draw_idle()


    labels = list(data.keys())
    visibility = [True] * len(labels)
    rax = plt.axes([0.02, 0.35, 0.22, 0.55])
    check = CheckButtons(rax, labels, visibility)

    def set_all_visibility(flag: bool):
        for i, label in enumerate(labels):
            lines[label].set_visible(flag)
            check.set_active(i) if (check.get_status()[i] != flag) else None
        if group_state['mode'] == 'off':
            for gl in group_lines.values():
                gl.set_visible(False)
        refresh_legend()
        plt.draw()

    def on_check_clicked(label):
        if group_state['mode'] == 'off':
            lines[label].set_visible(not lines[label].get_visible())
        else:
            pass
        refresh_legend()
        plt.draw()

    check.on_clicked(on_check_clicked)

    btn_all_ax = plt.axes([0.02, 0.27, 0.105, 0.06])
    btn_none_ax = plt.axes([0.135, 0.27, 0.105, 0.06])
    btn_all = Button(btn_all_ax, 'all')
    btn_none = Button(btn_none_ax, 'all not')

    def on_select_all(event):
        for i, state in enumerate(check.get_status()):
            if not state:
                check.set_active(i)
        for ln in lines.values():
            ln.set_visible(True)
        if group_state['mode'] == 'off':
            for gl in group_lines.values():
                gl.set_visible(False)
        refresh_legend()

    def on_select_none(event):
        for i, state in enumerate(check.get_status()):
            if state:
                check.set_active(i)
        for ln in lines.values():
            ln.set_visible(False)
        if group_state['mode'] == 'off':
            for gl in group_lines.values():
                gl.set_visible(False)
        refresh_legend()

    btn_all.on_clicked(on_select_all)
    btn_none.on_clicked(on_select_none)

    smooth_ax = plt.axes([0.02, 0.17, 0.22, 0.06])
    smooth_btn = Button(smooth_ax, 'smooth ON/OFF')
    smooth_state = {'on': False}

    def toggle_smooth(event):
        smooth_state['on'] = not smooth_state['on']
        def maybe_smooth(y):
            return gaussian_filter1d(y, sigma=1.5) if smooth_state['on'] else y

        for m in data:
            y = data[m]
            lines[m].set_ydata(maybe_smooth(y))

        if group_state['mode'] in ('mean', 'median'):
            recompute_groups()
        plt.draw()

    smooth_btn.on_clicked(toggle_smooth)

    group_ax = plt.axes([0.02, 0.08, 0.22, 0.06])
    group_btn = Button(group_ax, 'merge')

    group_state = {'mode': 'off'}
    def recompute_groups():
        mode = group_state['mode']
        if mode == 'off':

            for g, gl in group_lines.items():
                members = groups[g]
                if len(members) > 1:
                    gl.set_visible(False)
                else:
                    gl.set_visible(lines[members[0]].get_visible())
                    gl.set_ydata(lines[members[0]].get_ydata())
                    gl.set_xdata(lines[members[0]].get_xdata())
            for i, label in enumerate(labels):
                lines[label].set_visible(check.get_status()[i])
            refresh_legend()
            return

        for i, label in enumerate(labels):
            lines[label].set_visible(False)

        for gname, members in groups.items():
            selected = [m for m in members if check.get_status()[labels.index(m)]]
            if len(selected) == 0:
                group_lines[gname].set_visible(False)
                continue

            if len(selected) == 1:
                m = selected[0]
                gx, gy = x_vals[m], data[m]
            else:
                steps_list = [x_vals[m] for m in selected]
                vals_list = [data[m] for m in selected]
                gx, gy = aggregate_group(steps_list, vals_list, mode=mode)

            if smooth_state['on']:
                gy = gaussian_filter1d(np.array(gy, dtype=float), sigma=1.5)

            gl = group_lines[gname]
            gl.set_xdata(gx)
            gl.set_ydata(gy)
            gl.set_label(f"{gname} ({'average' if mode=='average' else 'median'})")
            gl.set_visible(True)

        refresh_legend()

    def on_group_toggle(event):
        if group_state['mode'] == 'off':
            group_state['mode'] = 'average'
            group_btn.label.set_text('merge: average')
        elif group_state['mode'] == 'average':
            group_state['mode'] = 'median'
            group_btn.label.set_text('merge: median')
        else:
            group_state['mode'] = 'off'
            group_btn.label.set_text('merge: off')
        recompute_groups()
        plt.draw()

    group_btn.on_clicked(on_group_toggle)


    refresh_legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument('--dir', '-d', required=True, help="")
    parser.add_argument('--metric', '-m', required=True, help="")
    args = parser.parse_args()

    fixed_color_dict = {

    }

    interactive_plot(
        dir_path=args.dir,
        metric_key=args.metric,
        fixed_colors=fixed_color_dict
    )


if __name__ == '__main__':
    main()
