from matplotlib import pyplot as plt
import numpy as np
from time import sleep
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
scope_max_events = 1_000_000
trigger_level_ch8 = 0.15

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
tagger.setInputDelay(8, 0)
tagger.setTriggerLevel(8, trigger_level_ch8)
tagger.clearConditionalFilter()
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

    counts = counter.getData(rolling=False)
    time_s = counter.getIndex() / 1e12

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, channel, channel_counts in zip(axes, (1, 2, 8), counts):
        ax.plot(time_s, channel_counts)
        ax.set_ylabel(f"Ch {channel} counts / 1 ms")
        ax.set_title(f"Channel {channel}")
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Simultaneous measurement: channels 1, 2 and 8")
    fig.tight_layout()
    plt.show()

    # Keep one shared time axis so interruptions can be compared directly.

finally:
    ws.close()
    s.delete(f"http://{HOST}/api/kernels/{kid}", headers=h)






