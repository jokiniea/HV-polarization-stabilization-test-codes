import numpy as np
from Swabian import TimeTagger
import csv
import json, uuid
import requests
import time
import websocket
from pathlib import Path

"""
Changing variables
"""
t_FPGA_ps = 62500
t_detector_ps = 1e9
t_sweep_ps = 10e9
N_H = 7
N_V = 7
N_send = 20
t_send = 10e9
margin = 0.06
sleep_time = 0.07

channel_H = 7
channel_V = 4

run_file = "HV_stablization_run_data.csv"
timing_file = "run_time_information.csv"
timing_round = 0
timing_last_end = None
program_start = time.perf_counter()

# Unchanging variables

HOST = "192.168.2.99:9090"
PASSWORD = "xilinx"

SETUP = r'''
import gc
import time
import numpy as np
from pynq import Overlay, allocate
import xrfclk

xrfclk.set_ref_clks(lmk_freq=122.88, lmx_freq=409.6)
ol = Overlay("/home/xilinx/jupyter_notebooks/dma_to_dac_16mhz_32mA_SG_enabled_dynamic_clk.xsa")
dma = ol.axi_dma_0
'''

SEND = r'''
dma.sendchannel.transfer(tx_buffer)
dma.sendchannel.wait()
tx_buffer.freebuffer()
'''

REPEAT_COUNT_SWEEP = int(t_sweep_ps/t_FPGA_ps)
REPEAT_COUNT_SEND = int(t_send/t_FPGA_ps)
trigger_num = int(700/1e9*t_detector_ps)

def initialize_run_file():
    run_path = Path(__file__).with_name(run_file)
    with run_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["phase", "time_s", "H", "V", "trigger_channel"])

def initialize_timing_file():
    timing_path = Path(__file__).with_name(timing_file)
    with timing_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "round",
            "phase",
            "subphase",
            "duration_s",
            "cpu_elapsed_s",
        ])

def save_timing(round_number, phase, subphase, duration_s, cpu_elapsed_s=None):
    timing_path = Path(__file__).with_name(timing_file)
    if cpu_elapsed_s is None:
        cpu_elapsed_s = time.perf_counter() - program_start
    with timing_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            round_number,
            phase,
            subphase,
            f"{duration_s:.6f}",
            f"{cpu_elapsed_s:.6f}",
        ])

def save_run_data(phase, t, data):
    run_path = Path(__file__).with_name(run_file)
    with run_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(
            (phase, time_s, h, v, trigger)
            for time_s, h, v, trigger in zip(t, data[0], data[1], data[2])
        )

def wait_for_run(msg_id):
    """Print output for a submitted board command until it finishes."""
    first_output = None
    execution_error = None
    while True:
        m = json.loads(ws.recv())
        if m["parent_header"].get("msg_id") != msg_id:
            continue
        if m["msg_type"] == "stream":
            if first_output is None:
                first_output = time.time()
            print(m["content"]["text"], end="")
        elif m["msg_type"] == "error":
            execution_error = "\n".join(m["content"]["traceback"])
        elif m["msg_type"] == "status" and m["content"]["execution_state"] == "idle":
            if execution_error is not None:
                raise RuntimeError(execution_error)
            return first_output

def run(code, wait=True):
    """Submit code to the board; optionally wait for kernel completion."""
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "me", "session": "s",
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell",
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False},
    }))
    if wait:
        return wait_for_run(msg_id)
    return msg_id

def extract_triggered(data, n_data, binwidth_ps, trig_row=2):
    """
    Palauttaa H, V ja ikkunan indeksit.
    data: Counterin data, muoto [kanava, bini], viimeinen rivi on triggerikanava
    n_data: odotettu datapisteiden määrä triggerin jälkeen
    """

    triggered = data[trig_row] > trigger_num
    if not np.any(triggered):
        raise RuntimeError("Kanavalla 8 ei havaittu liipaisua")
    trig_idx = int(np.argmax(triggered))
    first_idx = trig_idx
    last_idx = first_idx + n_data - 1         

    if last_idx >= data.shape[1]:
        raise RuntimeError(
            f"Kasvata margin_s"
        )

    H = data[0, first_idx:last_idx + 1].astype(float)
    V = data[1, first_idx:last_idx + 1].astype(float)
    return H, V, first_idx, last_idx

def sweep_and_find_best_values(SH,SV):
    global timing_round, timing_last_end
    sweep_start = time.perf_counter()
    timing_round += 1
    if timing_last_end is not None:
        save_timing(
            timing_round,
            "between",
            "previous_phase_to_sweep",
            sweep_start - timing_last_end,
        )

    if SH == None and SV == None:
        sweep_values_h = np.linspace(-12000, 12000, N_H, dtype=np.int16)
        sweep_values_v = np.linspace(-12000, 12000, N_V, dtype=np.int16)
    else:
        n_h_low = N_H // 2
        n_h_high = N_H - n_h_low
        n_v_low = N_V // 2
        n_v_high = N_V - n_v_low
        sweep_values_h = np.unique(np.concatenate([
            np.linspace(SH[0] - 4000, SH[0] + 4000, n_h_low, dtype=np.int16),
            np.linspace(SH[1] - 4000, SH[1] + 4000, n_h_high, dtype=np.int16),
        ]))
        sweep_values_v = np.unique(np.concatenate([
            np.linspace(SV[0] - 4000, SV[0] + 4000, n_v_low, dtype=np.int16),
            np.linspace(SV[1] - 4000, SV[1] + 4000, n_v_high, dtype=np.int16),
        ]))

    h_grid, v_grid = np.meshgrid(sweep_values_h, sweep_values_v)
    sweep_values = np.column_stack((h_grid.ravel(), v_grid.ravel())).astype(np.int16)
    sweep_values_flat = sweep_values.ravel().tolist()
    
    # Luo sweep data 
    GET_READY = r'''
    sweep_pairs = np.array({sweep_values_flat}, dtype=np.int16).reshape(-1, 2)
    sweep_data = np.repeat(sweep_pairs, {REPEAT_COUNT_SWEEP}, axis=0).reshape(-1)

    N = len(sweep_data)
    tx_buffer = allocate(shape=(N,),dtype=np.int16)
    tx_buffer[:] = sweep_data

    del sweep_data
    gc.collect()
    '''.format(
        sweep_values_flat=sweep_values_flat,
        REPEAT_COUNT_SWEEP=REPEAT_COUNT_SWEEP,
    )

    # Käske FPGA:ta valmistautumaan datan lähetykseen
    preparation_start = time.perf_counter()
    run(GET_READY)
    save_timing(
        timing_round,
        "sweep",
        "data_preparation",
        time.perf_counter() - preparation_start,
    )

    bins_per_setting = int(t_sweep_ps / t_detector_ps)
    n_data = len(sweep_values) * bins_per_setting
    margin_bins = int(np.ceil(margin * 1e12 / t_detector_ps))
    n_total = n_data + margin_bins

    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[channel_H, channel_V, 8],
        binwidth=int(t_detector_ps),
        n_values=n_total,
    )

    capture_start = time.perf_counter()
    counter.startFor(capture_duration=int(t_detector_ps * n_total))
    sleep_start = time.perf_counter()
    time.sleep(sleep_time)
    save_timing(
        timing_round,
        "sweep",
        "pre_send_sleep",
        time.perf_counter() - sleep_start,
    )
    dma_send_start = time.perf_counter()
    send_msg_id = run(SEND, wait=False)
    counter.waitUntilFinished()
    wait_for_run(send_msg_id)
    capture_end = time.perf_counter()
    save_timing(
        timing_round,
        "sweep",
        "dma_send_and_wait",
        capture_end - dma_send_start,
    )
    save_timing(
        timing_round,
        "sweep",
        "counter_capture",
        capture_end - capture_start,
    )

    processing_start = time.perf_counter()
    data = counter.getData(rolling=False)

    t = counter.getIndex() / 1e12
    t = t - t[0]

    save_run_data("sweep", t, data)

    H, V, first_idx, last_idx = extract_triggered(data, n_data, int(t_detector_ps))
    hv_sum = H + V
    hv_sum[hv_sum == 0] = np.nan
    H_norm = H / hv_sum 
    V_norm = V / hv_sum
    t = t[first_idx:last_idx + 1]

    H_segments = H_norm.reshape(len(sweep_values), bins_per_setting)
    V_segments = V_norm.reshape(len(sweep_values), bins_per_setting)
    valid = ~np.isnan(H_segments) & ~np.isnan(V_segments)
    counts = valid.sum(axis=1)
    H_segments = np.where(valid, H_segments, 0.0)
    V_segments = np.where(valid, V_segments, 0.0)
    H_means = H_segments.sum(axis=1) / np.maximum(counts, 1)
    V_means = V_segments.sum(axis=1) / np.maximum(counts, 1)
    H_std = np.sqrt(((H_segments - H_means[:, None]) ** 2 * valid).sum(axis=1) / np.maximum(counts - 1, 1))
    V_std = np.sqrt(((V_segments - V_means[:, None]) ** 2 * valid).sum(axis=1) / np.maximum(counts - 1, 1))
    H_stderr = H_std / np.sqrt(np.maximum(counts, 1))
    V_stderr = V_std / np.sqrt(np.maximum(counts, 1))

    usable = counts > 1
    z2_H = np.full(len(sweep_values), np.inf)
    z2_V = np.full(len(sweep_values), np.inf)
    z2_H[usable] = ((H_means[usable] - 1) / H_stderr[usable]) ** 2 + (V_means[usable] / V_stderr[usable]) ** 2
    z2_V[usable] = (H_means[usable] / H_stderr[usable]) ** 2 + ((V_means[usable] - 1) / V_stderr[usable]) ** 2

    SH_best = sweep_values[int(np.argmin(z2_H))]
    SV_best = sweep_values[int(np.argmin(z2_V))]

    print("Best H: ",SH_best, "best V: ",SV_best)
    SH = [int(SH_best[0]), int(SH_best[1])]
    SV = [int(SV_best[0]), int(SV_best[1])]
    timing_last_end = time.perf_counter()
    save_timing(
        timing_round,
        "sweep",
        "data_processing_and_analysis",
        timing_last_end - processing_start,
    )
    save_timing(
        timing_round,
        "sweep",
        "total_sweep_phase",
        timing_last_end - sweep_start,
    )
    return SH, SV

def send_data(SH, SV):
    global timing_round, timing_last_end
    send_start = time.perf_counter()
    if timing_last_end is not None:
        save_timing(
            timing_round,
            "between",
            "previous_phase_to_send",
            send_start - timing_last_end,
        )

    GET_READY_SEND = r'''
    values = np.concatenate([
    np.tile({SH}, {REPEAT_COUNT_SEND}),
    np.tile({SV}, {REPEAT_COUNT_SEND}),
    ])

    data = np.tile(values, {TEST_REPETITIONS})
    N = len(data)
    tx_buffer = allocate(shape=(N,), dtype=np.int16)
    tx_buffer[:] = data
    '''.format(
        SH=SH,SV=SV,
        REPEAT_COUNT_SEND=REPEAT_COUNT_SEND,
        TEST_REPETITIONS=N_send,
    )

    preparation_start = time.perf_counter()
    run(GET_READY_SEND)
    save_timing(
        timing_round,
        "send",
        "data_preparation",
        time.perf_counter() - preparation_start,
    )

    n_data = int(2*N_send*t_sweep_ps/t_detector_ps)
    margin_bins = int(np.ceil(margin * 1e12 / t_detector_ps))
    n_total = n_data + margin_bins

    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[channel_H, channel_V, 8],
        binwidth=int(t_detector_ps),
        n_values=n_total,
    )
    capture_start = time.perf_counter()
    counter.startFor(capture_duration=int(t_detector_ps * n_total))
    sleep_start = time.perf_counter()
    time.sleep(sleep_time)
    save_timing(
        timing_round,
        "send",
        "pre_send_sleep",
        time.perf_counter() - sleep_start,
    )
    dma_send_start = time.perf_counter()
    send_msg_id = run(SEND, wait=False)
    counter.waitUntilFinished()
    wait_for_run(send_msg_id)
    capture_end = time.perf_counter()
    save_timing(
        timing_round,
        "send",
        "dma_send_and_wait",
        capture_end - dma_send_start,
    )
    save_timing(
        timing_round,
        "send",
        "counter_capture",
        capture_end - capture_start,
    )

    processing_start = time.perf_counter()
    data = counter.getData(rolling=False)
    t = counter.getIndex() / 1e12
    t = t - t[0]
    
    save_run_data("measurement", t, data)
    timing_last_end = time.perf_counter()
    save_timing(
        timing_round,
        "send",
        "data_processing_and_save",
        timing_last_end - processing_start,
    )
    save_timing(
        timing_round,
        "send",
        "total_send_phase",
        timing_last_end - send_start,
    )

# Yhteys FPGA:n
s = requests.Session()
s.get(f"http://{HOST}/login")
s.post(f"http://{HOST}/login",
       data={"password": PASSWORD, "_xsrf": s.cookies["_xsrf"]})
h = {"X-XSRFToken": s.cookies["_xsrf"]}
kid = s.post(f"http://{HOST}/api/kernels", json={"name": "python3"}, headers=h).json()["id"]
cookie = "; ".join(f"{k}={v}" for k, v in s.cookies.items())
ws = websocket.create_connection(f"ws://{HOST}/api/kernels/{kid}/channels", cookie=cookie)

# Yhteys TimeTaggeriin
tagger = TimeTagger.createTimeTaggerNetwork('130.230.124.213')
tagger.setInputDelay(channel_H, 0)
tagger.setInputDelay(channel_V, 0)
tagger.setInputDelay(8, 0)
tagger.setTriggerLevel(8, 0.15)
tagger.setEventDivider(8, 16)
           
try:
    initialize_run_file()
    initialize_timing_file()
    setup_start = time.perf_counter()
    run(SETUP)
    save_timing(
        0,
        "setup",
        "fpga_overlay_initialization",
        time.perf_counter() - setup_start,
    )
    timing_last_end = time.perf_counter()
    SH, SV = sweep_and_find_best_values(None, None)

    for i in range(4):
        send_data(SH, SV)
        SH, SV = sweep_and_find_best_values(SH, SV)

finally:
    ws.close()
    s.delete(f"http://{HOST}/api/kernels/{kid}", headers=h)


