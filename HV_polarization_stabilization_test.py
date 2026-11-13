from matplotlib import pyplot as plt
import numpy as np
from Swabian import TimeTagger
import json, uuid
import requests
import time
import websocket

"""
Muunneltavat vakiot
"""
delta_t_FPGA = 62500
delta_t_detecor = 1e9
delta_sweep_mittaus = 2e11
resolution = 1000
name_of_sweep_file = "HV_stablization_code_test_sweep_data.csv"
name_of_test_file = "HV_stabilization_code_test_data.csv"

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

def extract_triggered(data, n_data, binwidth_ps, trig_row=2, offset=0):
    """
    Palauttaa H, V ja ikkunan indeksit.
    data: Counterin data, muoto [kanava, bini], viimeinen rivi on triggerikanava
    n_data: odotettu datapisteiden määrä triggerin jälkeen
    offset: montako binia triggerin bini(n) jälkeen data alkaa (0 = samasta binistä)
    """
    if not np.any(data[trig_row] > 700):
        raise RuntimeError("Kanavalla 8 ei havaittu liipaisua")

    trig_idx = int(np.argmax(data[trig_row] > 700))
    first_idx = trig_idx + offset
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
tagger.setInputDelay(1, 0)
tagger.setInputDelay(2, 0)
tagger.setInputDelay(8, 0)
tagger.setTriggerLevel(8, 0.15)
tagger.setEventDivider(8, 16)
           
try:

    # Setup FPGA
    run(SETUP)

    # Luo sweep data 
    GET_READY = r'''
    import numpy as np
    from pynq import allocate
    import gc
    import time

    sweep_values = np.arange(-12000, 13000, {resolution}, dtype=np.int16)
    REPEAT_COUNT = 2 * 3_200_000

    N = len(sweep_values) * REPEAT_COUNT

    # Allokoidaan DMA-muisti kerran
    tx_buffer = allocate(shape=(N,),dtype=np.int16)

    # Täytetään DMA-puskuri arvo kerrallaan
    for i, value in enumerate(sweep_values):

        start = i * REPEAT_COUNT
        end = start + REPEAT_COUNT

        tx_buffer[start:end] = value

    del sweep_values
    gc.collect()
    '''.format(resolution=resolution)

    sweep_values = np.arange(-12000, 13000, resolution)
    N_S = 25000//resolution
    
    # Käske FPGA:ta valmistautumaan datan lähetykseen
    run(GET_READY)

    SEND = r'''
    dma.sendchannel.transfer(tx_buffer)
    dma.sendchannel.wait()
    tx_buffer.freebuffer()
    '''
    
    n_data = 5000
    margin_s = 2.0
    margin_bins = int(np.ceil(margin_s * 1e12 / delta_t_detecor))
    n_total = n_data + margin_bins

    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[1, 2, 8],
        binwidth=int(delta_t_detecor),
        n_values=n_total,
    )
    counter.startFor(capture_duration=int(delta_t_detecor * n_total))
    time.sleep(0.1)
    send_msg_id = run(SEND, wait=False)
    counter.waitUntilFinished()
    wait_for_run(send_msg_id)

    data = counter.getData(rolling=False)

    t = counter.getIndex() / 1e12
    t = t - t[0]

    output = np.column_stack([t, data[0], data[1], data[2]])
    np.savetxt(
        name_of_sweep_file,
        output,
        delimiter=",",
        header="time_s,H,V,trigger_channel",
        comments="",
    )
    H, V, first_idx, last_idx = extract_triggered(data, n_data, int(delta_t_detecor))
    hv_sum = H + V
    hv_sum[hv_sum == 0] = np.nan
    H_norm = H / hv_sum 
    V_norm = V / hv_sum
    t = t[first_idx:last_idx + 1]

    idx = int(delta_sweep_mittaus/delta_t_detecor)

    best_z2_H = np.inf
    best_z2_V = np.inf

    SH_best = 0
    SV_best = 0

    for i in range(N_S):
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

    GET_READY2 = r'''
    values = np.concatenate((np.full(2 * 16000, {SH_best}),np.full(2 * 16000, {SV_best})))
    data = np.tile(values, 100)
    N = len(data)
    tx_buffer = allocate(shape=(N,), dtype=np.int16)
    tx_buffer[:] = data
    '''.format(SH_best = SH_best, SV_best=SV_best)

    # Käske FPGA:ta valmistautumaan datan lähetykseen
    run(GET_READY2)

    SEND2 = r'''
    dma.sendchannel.transfer(tx_buffer)
    dma.sendchannel.wait()
    tx_buffer.freebuffer()
    '''
    n_data = 100
    margin_s = 1.0
    margin_bins = int(np.ceil(margin_s * 1e12 / delta_t_detecor))
    n_total = n_data + margin_bins

    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[1, 2, 8],
        binwidth=int(delta_t_detecor),
        n_values=n_total,
    )
    counter.startFor(capture_duration=int(delta_t_detecor * n_total))
    time.sleep(0.1)
    send_msg_id = run(SEND2, wait=False)
    counter.waitUntilFinished()
    wait_for_run(send_msg_id)

    data = counter.getData(rolling=False)
    t = counter.getIndex() / 1e12
    t = t - t[0]
    
    np.savetxt(
        name_of_test_file,
        np.column_stack((t, data.T)),
        delimiter=",",
        header="time_s,channel_1,channel_2,channel_8",
        comments="",
    )

finally:
    ws.close()
    s.delete(f"http://{HOST}/api/kernels/{kid}", headers=h)


