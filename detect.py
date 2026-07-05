import os
import warnings
# Suppress TensorFlow and CUDA warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL']        = '3'
os.environ['CUDA_VISIBLE_DEVICES']         = ''
os.environ['TF_ENABLE_ONEDNN_OPTS']        = '0'
os.environ['ABSL_MIN_LOG_LEVEL']           = '3'
warnings.filterwarnings('ignore')
import time
import signal
import argparse
import numpy as np
import pandas as pd
import joblib
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
from datetime import datetime
# Suppress absl logging after import
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

MODEL_DIR = '/home/vboxuser/nids-project/models'
ALERT_LOG = '/home/vboxuser/nids-project/alerts.log'

# CLI arguments
parser = argparse.ArgumentParser(description='NIDS Real-time Detector')
parser.add_argument('--interface',    default=None)
parser.add_argument('--model',        choices=['nn', 'rf'], default='rf')
parser.add_argument('--flow-timeout', type=int, default=5)
parser.add_argument('--window',       type=int, default=100)
args = parser.parse_args()

# Load model artifacts
print("Loading model artifacts.")
scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

if args.model == 'rf':
    clf = joblib.load(os.path.join(MODEL_DIR, 'nids_model_rf.pkl'))
    clf.verbose = 0
    def predict(X):
        return clf.predict(X)[0]
else:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    nn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'nids_model_nn.h5'))
    def predict(X):
        probs = nn_model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)[0]

print(f"Model    : {'Random Forest' if args.model == 'rf' else 'Neural Network'}")
print(f"Features : {len(feature_names)}")
print(f"Log file : {ALERT_LOG}\n")

log_file = open(ALERT_LOG, 'a')

def write_log(message):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {message}"
    print(line)
    log_file.write(line + '\n')
    log_file.flush()


# Flow store
flows    = {}
running  = True   
def make_flow_record(initiator):
    return {
        'initiator'  : initiator,
        'fwd'        : [],
        'bwd'        : [],
        'start_time' : None,
        'last_seen'  : None,
    }
def get_flow_key_and_src(pkt):
    if IP not in pkt:
        return None, None
    proto    = 'TCP' if TCP in pkt else ('UDP' if UDP in pkt else 'OTHER')
    src_port = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
    dst_port = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
    src = (pkt[IP].src, src_port)
    dst = (pkt[IP].dst, dst_port)
    a, b = (src, dst) if src <= dst else (dst, src)
    return (*a, *b, proto), src


def safe(fn, arr, default=0.0):
    return float(fn(arr)) if len(arr) > 0 else default

def safe_std(arr):
    return float(np.std(arr)) if len(arr) > 1 else 0.0

def inter_arrival(timestamps):
    if len(timestamps) < 2:
        return np.array([0.0])
    return np.diff(sorted(timestamps))

FLAG_BITS = [
    (0x01,'FIN'),(0x02,'SYN'),(0x04,'RST'),
    (0x08,'PSH'),(0x10,'ACK'),(0x20,'URG'),
    (0x40,'CWE'),(0x80,'ECE'),
]

def count_flags(pkt_list):
    counts = defaultdict(int)
    for _, p in pkt_list:
        if TCP in p:
            f = int(p[TCP].flags)
            for bit, name in FLAG_BITS:
                if f & bit:
                    counts[name] += 1
    return counts

BULK_THRESH = 0.001
def bulk_stats(ts_list, len_list):
    if len(ts_list) < 2:
        return 0.0, 0.0, 0.0
    iats = np.diff(sorted(ts_list))
    b_bytes = b_dur = run_bytes = 0.0
    b_pkts = run_pkts = n_bulks = 0
    for i, iat in enumerate(iats):
        if iat < BULK_THRESH:
            run_bytes += len_list[i] if i < len(len_list) else 0
            run_pkts  += 1
            b_dur     += iat
        else:
            if run_pkts > 0:
                b_bytes += run_bytes; b_pkts += run_pkts; n_bulks += 1
            run_bytes = run_pkts = 0
    if run_pkts > 0:
        b_bytes += run_bytes; b_pkts += run_pkts; n_bulks += 1
    if n_bulks == 0:
        return 0.0, 0.0, 0.0
    return b_bytes/n_bulks, b_pkts/n_bulks, b_bytes/max(b_dur, 1e-9)

IDLE_THRESH = 1.0

def extract_features(flow):
    fwd = flow['fwd']
    bwd = flow['bwd']
    all_pkts = sorted(fwd + bwd, key=lambda x: x[0])
    if len(all_pkts) < 2:
        return None

    all_ts  = [x[0] for x in all_pkts]
    all_len = np.array([len(x[1]) for x in all_pkts], dtype=float)
    fwd_ts  = [x[0] for x in fwd]
    bwd_ts  = [x[0] for x in bwd]
    fwd_len = np.array([len(x[1]) for x in fwd], dtype=float)
    bwd_len = np.array([len(x[1]) for x in bwd], dtype=float)

    flow_duration = max((all_ts[-1] - all_ts[0]) * 1e6, 1.0)
    duration_s    = flow_duration / 1e6

    all_iats = inter_arrival(all_ts)
    fwd_iats = inter_arrival(fwd_ts)
    bwd_iats = inter_arrival(bwd_ts)

    fwd_flags = count_flags(fwd)
    bwd_flags = count_flags(bwd)
    all_flags = count_flags(all_pkts)

    fwd_init_win = int(fwd[0][1][TCP].window) if fwd and TCP in fwd[0][1] else 0
    bwd_init_win = int(bwd[0][1][TCP].window) if bwd and TCP in bwd[0][1] else 0

    active_arr = np.array([t for t in all_iats if t <= IDLE_THRESH]) if any(t <= IDLE_THRESH for t in all_iats) else np.array([0.0])
    idle_arr   = np.array([t for t in all_iats if t >  IDLE_THRESH]) if any(t >  IDLE_THRESH for t in all_iats) else np.array([0.0])

    fwd_bulk = bulk_stats(fwd_ts, fwd_len.tolist())
    bwd_bulk = bulk_stats(bwd_ts, bwd_len.tolist())

    return {
        'Flow Duration'               : flow_duration,
        'Total Fwd Packets'           : len(fwd),
        'Total Backward Packets'      : len(bwd),
        'Total Length of Fwd Packets' : safe(np.sum,  fwd_len),
        'Total Length of Bwd Packets' : safe(np.sum,  bwd_len),
        'Fwd Packet Length Max'       : safe(np.max,  fwd_len),
        'Fwd Packet Length Min'       : safe(np.min,  fwd_len),
        'Fwd Packet Length Mean'      : safe(np.mean, fwd_len),
        'Fwd Packet Length Std'       : safe_std(fwd_len),
        'Bwd Packet Length Max'       : safe(np.max,  bwd_len),
        'Bwd Packet Length Min'       : safe(np.min,  bwd_len),
        'Bwd Packet Length Mean'      : safe(np.mean, bwd_len),
        'Bwd Packet Length Std'       : safe_std(bwd_len),
        'Flow Bytes/s'                : safe(np.sum, all_len) / duration_s,
        'Flow Packets/s'              : len(all_pkts) / duration_s,
        'Flow IAT Mean'               : safe(np.mean, all_iats),
        'Flow IAT Std'                : safe_std(all_iats),
        'Flow IAT Max'                : safe(np.max,  all_iats),
        'Flow IAT Min'                : safe(np.min,  all_iats),
        'Fwd IAT Total'               : safe(np.sum,  fwd_iats),
        'Fwd IAT Mean'                : safe(np.mean, fwd_iats),
        'Fwd IAT Std'                 : safe_std(fwd_iats),
        'Fwd IAT Max'                 : safe(np.max,  fwd_iats),
        'Fwd IAT Min'                 : safe(np.min,  fwd_iats),
        'Bwd IAT Total'               : safe(np.sum,  bwd_iats),
        'Bwd IAT Mean'                : safe(np.mean, bwd_iats),
        'Bwd IAT Std'                 : safe_std(bwd_iats),
        'Bwd IAT Max'                 : safe(np.max,  bwd_iats),
        'Bwd IAT Min'                 : safe(np.min,  bwd_iats),
        'Fwd PSH Flags'               : fwd_flags['PSH'],
        'Bwd PSH Flags'               : bwd_flags['PSH'],
        'Fwd URG Flags'               : fwd_flags['URG'],
        'Bwd URG Flags'               : bwd_flags['URG'],
        'Fwd Header Length'           : 20 * len(fwd),
        'Bwd Header Length'           : 20 * len(bwd),
        'Fwd Packets/s'               : len(fwd) / duration_s,
        'Bwd Packets/s'               : len(bwd) / duration_s,
        'Min Packet Length'           : safe(np.min,  all_len),
        'Max Packet Length'           : safe(np.max,  all_len),
        'Packet Length Mean'          : safe(np.mean, all_len),
        'Packet Length Std'           : safe_std(all_len),
        'Packet Length Variance'      : float(np.var(all_len)) if len(all_len) > 1 else 0.0,
        'FIN Flag Count'              : all_flags['FIN'],
        'SYN Flag Count'              : all_flags['SYN'],
        'RST Flag Count'              : all_flags['RST'],
        'PSH Flag Count'              : all_flags['PSH'],
        'ACK Flag Count'              : all_flags['ACK'],
        'URG Flag Count'              : all_flags['URG'],
        'CWE Flag Count'              : all_flags['CWE'],
        'ECE Flag Count'              : all_flags['ECE'],
        'Down/Up Ratio'               : len(bwd) / max(len(fwd), 1),
        'Average Packet Size'         : safe(np.mean, all_len),
        'Avg Fwd Segment Size'        : safe(np.mean, fwd_len),
        'Avg Bwd Segment Size'        : safe(np.mean, bwd_len),
        'Fwd Avg Bytes/Bulk'          : fwd_bulk[0],
        'Fwd Avg Packets/Bulk'        : fwd_bulk[1],
        'Fwd Avg Bulk Rate'           : fwd_bulk[2],
        'Bwd Avg Bytes/Bulk'          : bwd_bulk[0],
        'Bwd Avg Packets/Bulk'        : bwd_bulk[1],
        'Bwd Avg Bulk Rate'           : bwd_bulk[2],
        'Subflow Fwd Packets'         : len(fwd),
        'Subflow Fwd Bytes'           : safe(np.sum, fwd_len),
        'Subflow Bwd Packets'         : len(bwd),
        'Subflow Bwd Bytes'           : safe(np.sum, bwd_len),
        'Init_Win_bytes_forward'      : fwd_init_win,
        'Init_Win_bytes_backward'     : bwd_init_win,
        'act_data_pkt_fwd'            : len(fwd),
        'min_seg_size_forward'        : 20,
        'Active Mean'                 : safe(np.mean, active_arr),
        'Active Std'                  : safe_std(active_arr),
        'Active Max'                  : safe(np.max,  active_arr),
        'Active Min'                  : safe(np.min,  active_arr),
        'Idle Mean'                   : safe(np.mean, idle_arr),
        'Idle Std'                    : safe_std(idle_arr),
        'Idle Max'                    : safe(np.max,  idle_arr),
        'Idle Min'                    : safe(np.min,  idle_arr),
    }

def classify_flow(flow_key, flow):
    features = extract_features(flow)
    if features is None:
        return
    row      = {col: features.get(col, 0.0) for col in feature_names}
    df       = pd.DataFrame([row])[feature_names]
    df       = df.clip(-1e9, 1e9).replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(df.values)
    pred_idx = predict(X_scaled)
    label    = label_encoder.inverse_transform([pred_idx])[0]

    ip_a, port_a, ip_b, port_b, proto = flow_key
    initiator = flow['initiator']
    n_fwd     = len(flow['fwd'])
    n_bwd     = len(flow['bwd'])

    if label != 'BENIGN':
        msg = (f"ALERT [{label}]  "
               f"{initiator[0]}:{initiator[1]} -> {ip_b}:{port_b}  "
               f"proto={proto}  fwd={n_fwd}  bwd={n_bwd}")
        write_log(msg)
    else:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}]    BENIGN  "
              f"{initiator[0]}:{initiator[1]} -> {ip_b}:{port_b}  "
              f"proto={proto}  fwd={n_fwd}  bwd={n_bwd}")

# Packet callback
packet_count = 0
def process_packet(pkt):
    global packet_count
    packet_count += 1
    key, src = get_flow_key_and_src(pkt)
    if key is None:
        return
    now = time.time()
    if key not in flows:
        flows[key] = make_flow_record(initiator=src)
    flow = flows[key]
    if flow['start_time'] is None:
        flow['start_time'] = now
    flow['last_seen'] = now
    direction = 'fwd' if src == flow['initiator'] else 'bwd'
    flow[direction].append((now, pkt))
    if TCP in pkt:
        flags = int(pkt[TCP].flags)
        if flags & 0x01 or flags & 0x04:
            classify_flow(key, flow)
            del flows[key]
            return
    if packet_count % args.window == 0:
        timed_out = [k for k, v in list(flows.items())
                     if now - v['last_seen'] > args.flow_timeout]
        for k in timed_out:
            classify_flow(k, flows[k])
            del flows[k]

def shutdown(signum=None, frame=None):
    print("\nStopping, classifying remaining open flows.")
    for k, v in list(flows.items()):
        try:
            if v['fwd'] or v['bwd']:
                classify_flow(k, v)
        except Exception:
            pass
    try:
        log_file.close()
    except Exception:
        pass
    print("Done.")
    os._exit(0)   


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# entry
if __name__ == '__main__':
    print(f"Bidirectional NIDS detector")
    print(f"Interface    : {args.interface or 'auto'}")
    print(f"Flow timeout : {args.flow_timeout}s")
    print(f"Model        : {'Random Forest' if args.model == 'rf' else 'Neural Network'}")
    print(f"Alert log    : {ALERT_LOG}")
    print("Press Ctrl+C to stop.\n")

    sniff(
        iface=args.interface,
        prn=process_packet,
        store=False,
        filter="ip"
    )
