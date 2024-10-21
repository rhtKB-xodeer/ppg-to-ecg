# ppg-to-ecg
It implements the reconstruction of ECG signals from corresponding PPG signals. Here, the U-Net architecture is used to do the task, and in the encoder section, the Bi-LSTM layer is added to extract the signals' temporal characteristics.
For training and testing purposes, we have used the MIMIC III( https://physionet.org/content/mimiciii/1.4/ ) dataset, which contains the patients' simultaneous ECG and PPG signals.
