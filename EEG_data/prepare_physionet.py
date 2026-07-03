import argparse
import glob
import math
import ntpath
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from mne.io import read_raw_edf
import dhedfreader

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W, "N1": N1, "N2": N2, "N3": N3, "REM": REM, "UNKNOWN": UNKNOWN
}

ann2label = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
    "Sleep stage 4": 3, "Sleep stage R": 4, "Sleep stage ?": 5, "Movement time": 5
}

EPOCH_SEC_SIZE = 30

print(">>> [系统检查] 脚本已成功启动！")


def main():
    print(">>> [系统检查] 成功进入主函数，开始配置路径...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    auto_data_dir = os.path.join(current_dir, "physionet_sleep")
    auto_output_dir = os.path.join(current_dir, "eeg_eog_emg")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=auto_data_dir, help="数据目录")
    parser.add_argument("--output_dir", type=str, default=auto_output_dir, help="输出目录")
    parser.add_argument("--select_ch", type=str, nargs='+', default=["EEG Fpz-Cz", "EOG horizontal", "EMG submental"])
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    select_ch = args.select_ch

    # 【修复重点】：智能配对逻辑
    all_psg = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    all_psg.sort()

    psg_fnames = []
    ann_fnames = []

    print(">>> [文件匹配] 正在自动配对波形与标签文件...")
    for psg_path in all_psg:
        filename = os.path.basename(psg_path)
        subject_id = filename[:7]  # 提取类似 'SC4001E' 的特征码
        ann_pattern = os.path.join(args.data_dir, f"{subject_id}*-Hypnogram.edf")
        matching_ann = glob.glob(ann_pattern)

        if len(matching_ann) > 0:
            psg_fnames.append(psg_path)
            ann_fnames.append(matching_ann[0])
        else:
            print(f"  [!] 警告: 找不到 {filename} 对应的标签，已安全跳过。")

    print(f">>> [数据诊断] 成功完美匹配了 {len(psg_fnames)} 对数据！")

    if len(psg_fnames) == 0:
        print("❌ [致命错误] 没有找到任何成对的数据！程序终止。")
        return

    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    print("\n==================正式开始提取数据==================\n")

    for i in range(len(psg_fnames)):
        print(f"\n正在处理 [{i + 1}/{len(psg_fnames)}]: {os.path.basename(psg_fnames[i])}")

        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None,
                           verbose='ERROR')  # verbose='ERROR' 可以屏蔽那些烦人的红字警告
        sampling_rate = raw.info['sfreq']

        raw_ch_df = raw.to_data_frame()[select_ch]
        raw_ch_df.set_index(np.arange(len(raw_ch_df)), inplace=True)

        f = open(psg_fnames[i], 'rb')
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        f = open(ann_fnames[i], 'rb')
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
        f.close()
        ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

        assert raw_start_dt == ann_start_dt

        remove_idx = []
        labels = []
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                label_idx.append(idx)
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                remove_idx.append(idx)

        labels = np.hstack(labels)

        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))

        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)

        if len(label_idx) > len(select_idx):
            extra_idx = np.setdiff1d(label_idx, select_idx)
            if np.all(extra_idx > select_idx[-1]):
                n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                select_idx = select_idx[:-n_trims]
                labels = labels[:-n_label_trims]

        raw_ch = raw_ch_df.values[select_idx]

        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")

        n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))

        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)

        x = x[select_idx]
        y = y[select_idx]
        print(f"  -> 切片成功! 数据形状: x={x.shape}, y={y.shape}")

        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        print("  -> 已保存至:", filename)

    print("\n🎉 [大功告成] 所有多通道数据提取完毕！")


if __name__ == "__main__":
    main()