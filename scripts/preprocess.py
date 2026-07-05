import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

dataset = os.path.expanduser("~/nids-project/datasets/CSE-CIC-IDS2018")
model = os.path.expanduser("~/nids-project/models")
data1 = os.path.expanduser("~/nids-project/data")

IDS_files = [
  'Brute Force -Web.csv',
  'Brute Force -XSS.csv',
  'DDOS attack-HOIC.csv',
  'DDOS attack-LOIC-UDP.csv',
  'DDoS attacks-LOIC-HTTP.csv',
  'DoS attacks-GoldenEye.csv',
  'DoS attacks-Hulk.csv',
  'DoS attacks-SlowHTTPTest.csv',
  'DoS attacks-Slowloris.csv',
  'FTP-BruteForce.csv',
  'Infilteration.csv',
  'SQL Injection.csv',
  'SSH-Bruteforce.csv',
]

LABEL_MAP = {
  'Benign'              : 'BENIGN',
  'Brute Force -XSS'    : 'BruteForce-XSS',
  'DDOS attack-HOIC'    : 'DDoS-HOIC',
  'DDOS attack-LOIC-UDP': 'DDoS-LOIC-UDP',
  'DDoS attacks-LOIC-HTTP': 'DDoS-LOIC-HTTP',
  'DoS attacks-GoldenEye': 'DoS-GoldenEye',
  'DoS attacks-Hulk'    : 'DoS-Hulk',
  'DoS attacks-SlowHTTPTest': 'DoS-SlowHTTPTest',
  'DoS attacks-Slowloris': 'DoS-Slowloris',
  'FTP-BruteForce'      : 'BruteForce-FTP',
  'Infiltration'        : 'Infiltration',
  'Infilteration'       : 'Infiltration',
  'SQL Injection'       : 'SQLInjection',
  'SSH-Bruteforce'      : 'BruteForce-SSH',
  'Bot'                 : 'Botnet',
  'Label'               : None,
}

def load_data():
   frames = []
   for filename in IDS_files:
       path = os.path.join(dataset, filename)
       if not os.path.exists(path):
           print(f"Not found, skipping: {filename}")
           continue
       print(f"Loading {filename}...")
       chunks = []
       for chunk in pd.read_csv(path, low_memory=False, chunksize=50000):
          chunks.append(chunk.sample(frac=0.3, random_state=42))
       df = pd.concat(chunks, ignore_index=True)

      
       df.columns = df.columns.str.strip()
       COLUMN_RENAME = { 'Tot Fwd Pkts' : 'Total Fwd Packets',
                       'Tot Bwd Pkts' : 'Total Backward Packets',
                       'TotLen Fwd Pkts' : 'Total Length of Fwd Packets',
                       'TotLen Bwd Pkts' : 'Total Length of Bwd Packets',
                       'Fwd Pkt Len Max' : 'Fwd Packet Length Max',
                       'Fwd Pkt Len Min' : 'Fwd Packet Length Min',
                       'Fwd Pkt Len Mean' : 'Fwd Packet Length Mean',
                       'Fwd Pkt Len Std' : 'Fwd Packet Length Std',
                       'Bwd Pkt Len Max' : 'Bwd Packet Length Max',
                       'Bwd Pkt Len Min' : 'Bwd Packet Length Min',
                       'Bwd Pkt Len Mean' : 'Bwd Packet Length Mean',
                       'Bwd Pkt Len Std' : 'Bwd Packet Length Std',
                       'Flow Byts/s' : 'Flow Bytes/s',
                       'Flow Pkts/s' : 'Flow Packets/s',
                       'Fwd IAT Tot' : 'Fwd IAT Total',
                       'Fwd Header Len' : 'Fwd Header Length',
                       'Bwd Header Len' : 'Bwd Header Length',
                       'Fwd Pkts/s' : 'Fwd Packets/s',
                       'Bwd Pkts/s' : 'Bwd Packets/s',
                       'Pkt Len Min' : 'Min Packet Length',
                       'Pkt Len Max' : 'Max Packet Length',
                       'Pkt Len Mean' : 'Packet Length Mean',
                       'Pkt Len Std' : 'Packet Length Std',
                       'Pkt Len Var' : 'Packet Length Variance',
                       'FIN Flag Cnt' : 'FIN Flag Count',
                       'SYN Flag Cnt' : 'SYN Flag Count',
                       'RST Flag Cnt' : 'RST Flag Count',
                       'PSH Flag Cnt' : 'PSH Flag Count',
                       'ACK Flag Cnt' : 'ACK Flag Count',
                       'URG Flag Cnt' : 'URG Flag Count',
                       'ECE Flag Cnt' : 'ECE Flag Count',
                       'Pkt Size Avg' : 'Average Packet Size',
                       'Fwd Seg Size Avg' : 'Avg Fwd Segment Size',
                       'Bwd Seg Size Avg' : 'Avg Bwd Segment Size',
                       'Fwd Byts/b Avg' : 'Fwd Avg Bytes/Bulk',
                       'Fwd Pkts/b Avg' : 'Fwd Avg Packets/Bulk',
                       'Fwd Blk Rate Avg' : 'Fwd Avg Bulk Rate', 
                       'Bwd Byts/b Avg' : 'Bwd Avg Bytes/Bulk', 
                       'Bwd Pkts/b Avg' : 'Bwd Avg Packets/Bulk', 
                       'Bwd Blk Rate Avg' : 'Bwd Avg Bulk Rate', 
                       'Subflow Fwd Pkts' : 'Subflow Fwd Packets', 
                       'Subflow Fwd Byts' : 'Subflow Fwd Bytes', 
                       'Subflow Bwd Pkts' : 'Subflow Bwd Packets', 
                       'Subflow Bwd Byts' : 'Subflow Bwd Bytes', 
                       'Init Fwd Win Byts' : 'Init_Win_bytes_forward', 
                       'Init Bwd Win Byts' : 'Init_Win_bytes_backward', 
                       'Fwd Act Data Pkts' : 'act_data_pkt_fwd', 
                       'Fwd Seg Size Min' : 'min_seg_size_forward', 
                       'Bwd IAT Tot' : 'Bwd IAT Total', }
       df.rename(columns=COLUMN_RENAME, inplace=True)


       label_col = next((c for c in df.columns if c.strip().lower() == 'label'), None)
       if label_col is None:
           print (f'No label column found, skipping {filename}')
           continue
       df = df[df[label_col]!='Label']
       df[label_col] = df[label_col].str.strip().map(
       lambda x:LABEL_MAP.get(x,x)
       )
       df = df[df[label_col].notna()] 
       df = df.rename(columns={label_col:'label'})
       frames.append(df)
      
   if not frames:
       raise FileNotFoundError(
           "Not dataset files were found.\n"
           f" Put the files in: {dataset}")
   combined = pd.concat(frames, ignore_index=True)
   print(f"\n Loaded {len(combined):,}rows from {len(frames)} files")
   print("Label distribution:")
   print(combined['label'].value_counts().to_string())
   return combined


def main():
   print("Starting preprocessing for IDS Files.\n")
   df = load_data()
   labels = df["label"].copy()
   df_numeric = df.select_dtypes(include=['number'])
   zero_cols = df_numeric.columns[df_numeric.std() < 1e-6]
   if len(zero_cols):
       print(f"\n Removing {len(zero_cols)} near zero variance columns")
       df_numeric = df_numeric.drop(columns=zero_cols)
   inf_cols = df_numeric.columns[np.isinf(df_numeric).any()]
   if len(inf_cols):
       print(f"Replacing inf values in: {inf_cols.tolist()}")
   df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
   df_numeric.fillna(0, inplace=True)
   if not np.isfinite(df_numeric.values).all():
       raise ValueError("Non-finite values remain after cleaning.")
   X=df_numeric
   label_encoder = LabelEncoder()
   y = label_encoder.fit_transform(labels)
   print(f"\n Feature matrix shape: {X.shape}")
   print(f"\n Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
   os.makedirs(model, exist_ok=True)
   os.makedirs(data1, exist_ok=True)
   scaler= MinMaxScaler()
   scaler.fit(X)
   joblib.dump(scaler, os.path.join(model,'scaler.pkl'))
   joblib.dump(label_encoder, os.path.join(model,'label_encoder.pkl'))
   joblib.dump((X,y), os.path.join(data1,'processed.pkl'))
   joblib.dump(list(X.columns), os.path.join(model,'feature_names.pkl'))
   print(f"Preprocessing complete. Data saved to:")
   print(f"\n- {data1}/processed.pkl")
   print(f"\n- {model}/scaler.pkl")
   print(f"\n- {model}/label_encoder.pkl")


if __name__ == "__main__":
   main()

