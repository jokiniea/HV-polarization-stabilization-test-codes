import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_FILE = "HV_stablization_run_data.csv"
TIMING_FILE = "run_time_information.csv"
MAX_POINTS_PER_PLOT = 100_000
OUTPUT_DIR = Path(__file__).with_name("continuous_run_plots")


def save_plot(output_dir, plot_number, time_s, h_counts, v_counts, trigger_counts):
    hv_sum = h_counts + v_counts
    h_normalized = np.divide(
        h_counts,
        hv_sum,
        out=np.full(h_counts.shape, np.nan, dtype=float),
        where=hv_sum != 0,
    )
    v_normalized = np.divide(
        v_counts,
        hv_sum,
        out=np.full(v_counts.shape, np.nan, dtype=float),
        where=hv_sum != 0,
    )

    fig, ratio_ax = plt.subplots(figsize=(14, 8))
    trigger_ax = ratio_ax.twinx()
    h_line, = ratio_ax.plot(time_s, h_normalized, label="H/(H+V)")
    v_line, = ratio_ax.plot(time_s, v_normalized, label="V/(H+V)")
    trigger_line, = trigger_ax.plot(
        time_s, trigger_counts, color="tab:red", alpha=0.7, label="trigger_channel"
    )
    ratio_ax.set_ylabel("Fraction")
    ratio_ax.set_ylim(-0.05, 1.05)
    ratio_ax.grid(True)
    trigger_ax.set_ylabel("Trigger-laskurit")
    ratio_ax.set_xlabel("Ajallinen mittaus (s)")
    ratio_ax.legend(
        [h_line, v_line, trigger_line],
        [h_line.get_label(), v_line.get_label(), trigger_line.get_label()],
    )

    fig.suptitle(f"Part {plot_number:03d}")
    fig.tight_layout()
    output_path = output_dir / f"run_data_{plot_number:03d}.png"
    plt.show()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Tallennettu kuvaaja: {output_path}")


def load_timing_rows(timing_path):
    if not timing_path.exists():
        print(
            f"Varoitus: {timing_path} ei ole olemassa; "
            "käytetään mittausdataan perustuvaa aikajanaa."
        )
        return []

    with timing_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"round", "phase", "subphase", "duration_s"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{timing_path} ei sisällä vaadittuja sarakkeita: "
                f"{', '.join(sorted(required_columns))}"
            )
        return list(reader)


def build_phase_offsets(timing_rows):
    """Map sweep and send phase occurrences to a shared wall-clock timeline."""
    offsets = {}
    durations = {}

    setup_duration = 0.0
    for row in timing_rows:
        if row["phase"] == "setup" and row["subphase"] == "fpga_overlay_initialization":
            setup_duration += float(row["duration_s"])

    timeline_end = setup_duration
    sweep_count = 0
    send_count = 0
    has_sweep = False

    for row in timing_rows:
        if row["phase"] == "sweep" and row["subphase"] == "total_sweep_phase":
            sweep_count += 1
            sweep_round = int(row["round"])
            between_duration = 0.0
            for timing_row in timing_rows:
                if (
                    timing_row["phase"] == "between"
                    and int(timing_row["round"]) == sweep_round
                    and timing_row["subphase"] == "previous_phase_to_sweep"
                ):
                    between_duration = float(timing_row["duration_s"])
                    break

            if not has_sweep:
                sweep_start = setup_duration + between_duration
            else:
                sweep_start = timeline_end + between_duration

            sweep_duration = float(row["duration_s"])
            sweep_preparation = 0.0
            for timing_row in timing_rows:
                if (
                    timing_row["phase"] == "sweep"
                    and int(timing_row["round"]) == sweep_round
                    and timing_row["subphase"] == "data_preparation"
                ):
                    sweep_preparation = float(timing_row["duration_s"])
                    break
            offsets[f"sweep:{sweep_count}"] = sweep_start + sweep_preparation
            durations[f"sweep:{sweep_count}"] = sweep_duration
            timeline_end = sweep_start + sweep_duration
            has_sweep = True

            send_round = sweep_round
            send_duration = 0.0
            send_between = 0.0
            for timing_row in timing_rows:
                if (
                    timing_row["phase"] == "send"
                    and int(timing_row["round"]) == send_round
                    and timing_row["subphase"] == "total_send_phase"
                ):
                    send_duration = float(timing_row["duration_s"])
                if (
                    timing_row["phase"] == "between"
                    and int(timing_row["round"]) == send_round
                    and timing_row["subphase"] == "previous_phase_to_send"
                ):
                    send_between = float(timing_row["duration_s"])

            if send_duration > 0:
                send_count += 1
                send_start = sweep_start + sweep_duration + send_between
                send_preparation = 0.0
                for timing_row in timing_rows:
                    if (
                        timing_row["phase"] == "send"
                        and int(timing_row["round"]) == send_round
                        and timing_row["subphase"] == "data_preparation"
                    ):
                        send_preparation = float(timing_row["duration_s"])
                        break
                offsets[f"measurement:{send_count}"] = send_start + send_preparation
                durations[f"measurement:{send_count}"] = send_duration
                timeline_end = send_start + send_duration

    return offsets, durations


def plot_run_data(csv_path, output_dir, max_points_per_plot):
    if max_points_per_plot <= 0:
        raise ValueError("max_points_per_plot-arvon täytyy olla positiivinen")

    output_dir.mkdir(parents=True, exist_ok=True)
    timing_rows = load_timing_rows(csv_path.with_name(TIMING_FILE))
    timing_offsets, _ = build_phase_offsets(timing_rows)
    use_timing_offsets = bool(timing_offsets)

    chunk = []
    plot_number = 0
    current_phase = None
    event_time_offset = 0.0
    last_event_time = 0.0
    last_event_step = 0.0
    previous_time = None
    sweep_count = 0
    measurement_count = 0

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "phase",
            "time_s",
            "H",
            "V",
            "trigger_channel",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} ei sisällä vaadittuja sarakkeita: "
                f"{', '.join(sorted(required_columns))}"
            )

        for row in reader:
            phase = row["phase"]
            time_s = float(row["time_s"])

            if phase != current_phase or (
                previous_time is not None and time_s < previous_time
            ):
                if use_timing_offsets:
                    if phase == "sweep":
                        sweep_count += 1
                        event_time_offset = timing_offsets.get(
                            f"sweep:{sweep_count}",
                            event_time_offset,
                        )
                    elif phase == "measurement":
                        measurement_count += 1
                        event_time_offset = timing_offsets.get(
                            f"measurement:{measurement_count}",
                            event_time_offset,
                        )
                elif current_phase is not None:
                    event_time_offset += last_event_time + last_event_step
                current_phase = phase
                last_event_time = 0.0
                last_event_step = 0.0
                previous_time = None

            if previous_time is not None and time_s > previous_time:
                last_event_step = time_s - previous_time
            previous_time = time_s
            last_event_time = time_s

            chunk.append(
                (
                    event_time_offset + time_s,
                    float(row["H"]),
                    float(row["V"]),
                    float(row["trigger_channel"]),
                )
            )

            if len(chunk) == max_points_per_plot:
                plot_number += 1
                chunk_array = np.asarray(chunk, dtype=float)
                save_plot(
                    output_dir,
                    plot_number,
                    chunk_array[:, 0],
                    chunk_array[:, 1],
                    chunk_array[:, 2],
                    chunk_array[:, 3],
                )
                chunk.clear()

    if chunk:
        plot_number += 1
        chunk_array = np.asarray(chunk, dtype=float)
        save_plot(
            output_dir,
            plot_number,
            chunk_array[:, 0],
            chunk_array[:, 1],
            chunk_array[:, 2],
            chunk_array[:, 3],
        )

    if plot_number == 0:
        raise ValueError(f"{csv_path} ei sisällä mittausrivejä")


if __name__ == "__main__":
    plot_run_data(
        Path(__file__).with_name(RUN_FILE),
        OUTPUT_DIR,
        MAX_POINTS_PER_PLOT,
    )