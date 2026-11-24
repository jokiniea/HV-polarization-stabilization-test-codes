from matplotlib import pyplot as plt
import numpy as np
from Swabian import TimeTagger
import json, uuid
import requests
import time
import websocket
from itertools import product

"""
Muunneltavat vakiot
"""
t_FPGA_ps = 62500
t_detector_ps = 1e9
t_sweep_ps = 10e9
N_H = 9
N_V = 9
test_repetitions = 20
test_time = 1e10

channel_H = 7
channel_V = 4

sweep_file = "HV_stablization_code_test_sweep_data.csv"
test_file = "HV_stabilization_code_test_data.csv"

trigger_num = int(700/1e9*t_detector_ps)

HOST = "192.168.2.99:9090"
PASSWORD = "xilinx"

SETUP = r'''
import numpy as np
from pynq import Overlay, allocate
import xrfclk

xrfclk.set_ref_clks(lmk_freq=122.88, lmx_freq=409.6)
ol = Overlay("/home/xilinx/jupyter_notebooks/dma_to_dac_16mhz_32mA_SG_enabled_dynamic_clk.xsa")
dma = ol.axi_dma_0
'''

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
    if not np.any(data[trig_row] > trigger_num):
        raise RuntimeError("Kanavalla 8 ei havaittu liipaisua")

    trig_idx = int(np.argmax(data[trig_row] > trigger_num))
    first_idx = trig_idx
    last_idx = first_idx + n_data - 1          # viimeinen dataan kuuluva bini (mukaan lukien)

    if last_idx > data.shape[1]:
        raise RuntimeError(
            f"Kasvata margin_s"
        )

    to_ms = binwidth_ps * 1e-12 * 1e3
    print(f"Liipaisu: bini {trig_idx} ({trig_idx * to_ms:.1f} ms)")
    print(f"Ensimmäinen dataan kuuluva indeksi: {first_idx}")
    print(f"Viimeinen dataan kuuluva indeksi:   {last_idx}")
    print(f"Datapisteitä: {n_data}, ikkuna {first_idx * to_ms:.1f}...{(last_idx + 1) * to_ms:.1f} ms")

    tail = data[:trig_row, last_idx + 1:]
    n_tail_bins = tail.shape[1]
    n_tail_counts = int(tail.sum())
    print(f"Ikkunan jälkeen {n_tail_bins} binia, joissa yhteensä {n_tail_counts} tapahtumaa (hylätty)")

    H = data[0, first_idx:last_idx + 1].astype(float)
    V = data[1, first_idx:last_idx + 1].astype(float)
    return H, V, first_idx, last_idx


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

    # Setup FPGA
    run(SETUP)
    sweep_values_h = np.linspace(-12000, 12000, N_H, dtype=np.int16)
    sweep_values_v = np.linspace(-12000, 12000, N_V, dtype=np.int16)
    sweep_values = list(product(sweep_values_h, sweep_values_v))
    sweep_values_flat = np.asarray(sweep_values, dtype=np.int16).reshape(-1).tolist()
    REPEAT_COUNT = t_sweep_ps/t_FPGA_ps

    # Luo sweep data 
    GET_READY = r'''
    import numpy as np
    from pynq import allocate
    import gc
    import time

    sweep_pairs = np.array({sweep_values_flat}, dtype=np.int16).reshape(-1, 2)
    sweep_data = np.repeat(sweep_pairs, {REPEAT_COUNT}, axis=0).reshape(-1)

    N = len(sweep_data)
    tx_buffer = allocate(shape=(N,),dtype=np.int16)
    tx_buffer[:] = sweep_data

    del sweep_data
    gc.collect()
    '''.format(sweep_values_flat = sweep_values_flat, REPEAT_COUNT=REPEAT_COUNT)

    SEND = r'''
    dma.sendchannel.transfer(tx_buffer)
    dma.sendchannel.wait()
    tx_buffer.freebuffer()
    '''

    # Käske FPGA:ta valmistautumaan datan lähetykseen
    run(GET_READY)

    n_data = int(N_H*N_V*t_sweep_ps/t_detector_ps)
    margin_s = 0.5
    margin_bins = int(np.ceil(margin_s * 1e12 / t_detector_ps))
    n_total = n_data + margin_bins

    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[channel_H, channel_V, 8],
        binwidth=int(t_detector_ps),
        n_values=n_total,
    )

    counter.startFor(capture_duration=int(t_detector_ps * n_total))
    time.sleep(0.1)
    send_msg_id = run(SEND, wait=False)
    counter.waitUntilFinished()
    wait_for_run(send_msg_id)

    data = counter.getData(rolling=False)

    t = counter.getIndex() / 1e12
    t = t - t[0]

    output = np.column_stack([t, data[0], data[1], data[2]])
    np.savetxt(
        sweep_file,
        output,
        delimiter=",",
        header="time_s,H,V,trigger_channel",
        comments="",
    )

    H, V, first_idx, last_idx = extract_triggered(data, n_data, int(t_detector_ps))
    hv_sum = H + V
    hv_sum[hv_sum == 0] = np.nan
    H_norm = H / hv_sum 
    V_norm = V / hv_sum
    t = t[first_idx:last_idx + 1]

    best_z2_H = np.inf
    best_z2_V = np.inf
    SH_best = 0
    SV_best = 0
    idx = int(t_sweep_ps/t_detector_ps)
    for i in range(N_H*N_V):
        H_segment = H_norm[i*idx:(i+1)*idx]
        V_segment = V_norm[i*idx:(i+1)*idx]

        H_segment = H_norm[i * idx:(i + 1) * idx]
        H_segment = H_segment[~np.isnan(H_segment)]
        V_segment = V_norm[i * idx:(i + 1) * idx]
        V_segment = V_segment[~np.isnan(V_segment)]

        if len(H_segment) == 0 or len(V_segment) == 0:
            continue

        H_mean = np.mean(H_segment)
        stderr_H = np.std(H_segment) / np.sqrt(len(H_segment))
        V_mean = np.mean(V_segment)
        stderr_V = np.std(V_segment) / np.sqrt(len(V_segment))

        z2_H = ((H_mean-1)/stderr_H)**2 + (V_mean/stderr_V)**2
        z2_V = (H_mean/stderr_H)**2 + ((V_mean-1)/stderr_V)**2

        if z2_H < best_z2_H:
            SH_best = sweep_values[i]
            best_z2_H = z2_H

        if z2_V < best_z2_V:
            SV_best = sweep_values[i]
            best_z2_V = z2_V

    print("Best H: ",SH_best, "best V: ",SV_best)

    """
    Testataan saatuja arvoja
    """

    REPEAT_COUNT2 = int(test_time/t_FPGA_ps)
    SH = [int(SH_best[0]), int(SH_best[1])]
    SV = [int(SV_best[0]), int(SV_best[1])]

    GET_READY2 = r'''
    values = np.concatenate([
    np.tile({SH}, {REPEAT_COUNT2}),
    np.tile({SV}, {REPEAT_COUNT2}),
    ])

    data = np.tile(values, {TEST_REPETITIONS})
    N = len(data)
    tx_buffer = allocate(shape=(N,), dtype=np.int16)
    tx_buffer[:] = data
    '''.format(
        SH=SH,SV=SV,
        REPEAT_COUNT2=REPEAT_COUNT2,
        TEST_REPETITIONS=test_repetitions,
    )

    # Käske FPGA:ta valmistautumaan datan lähetykseen
    run(GET_READY2)

    SEND2 = r'''
    dma.sendchannel.transfer(tx_buffer)
    dma.sendchannel.wait()
    tx_buffer.freebuffer()
    '''
    n_data = int(2*test_repetitions*t_sweep_ps/t_detector_ps)
    margin_s = 0.5
    margin_bins = int(np.ceil(margin_s * 1e12 / t_detector_ps))
    n_total = n_data + margin_bins

    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[channel_H, channel_V, 8],
        binwidth=int(t_detector_ps),
        n_values=n_total,
    )
    counter.startFor(capture_duration=int(t_detector_ps * n_total))
    time.sleep(0.1)
    send_msg_id = run(SEND2, wait=False)
    counter.waitUntilFinished()
    wait_for_run(send_msg_id)

    data = counter.getData(rolling=False)
    t = counter.getIndex() / 1e12
    t = t - t[0]
    
    np.savetxt(
        test_file,
        np.column_stack((t, data.T)),
        delimiter=",",
        header="time_s,channel_1,channel_2,channel_8",
        comments="",
    )

finally:
    ws.close()
    s.delete(f"http://{HOST}/api/kernels/{kid}", headers=h)


