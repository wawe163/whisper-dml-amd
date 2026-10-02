# -*- coding: utf-8 -*-
"""并行批处理：3个进程各自加载DML模型，分头处理视频，吃满12GB显存
用法: python batch_parallel.py [科目] [并行数]
  科目: 政治|英语|高数|all (默认all)
  并行数: 默认3
"""
import os, sys, subprocess, glob, json

BASE = r"F:\工作\8.学习规定文件\4考试复习\2专升本\2成考"
TMP = r"C:\Users\ww\WorkBuddy\2026-09-10-22-53-30\.workbuddy\tmp\asr"
SRT_DIR = os.path.join(TMP, "srt")
PY = r"C:\Users\ww\.workbuddy\binaries\python\envs\dml\Scripts\python.exe"
TRANSCRIBE = r"C:\Users\ww\WorkBuddy\2026-09-10-22-53-30\whisper-dml-amd\scripts\transcribe.py"
VIDEO_EXT = (".mp4", ".mkv", ".flv", ".avi", ".mov")

def collect_videos(subject):
    vids = []
    for root, _, fs in os.walk(BASE):
        for f in fs:
            if not f.lower().endswith(VIDEO_EXT):
                continue
            p = os.path.join(root, f)
            if subject == "all":
                vids.append(p)
            elif subject == "政治" and "政治" in p:
                vids.append(p)
            elif subject == "英语" and "英语" in p:
                vids.append(p)
            elif subject == "高数" and "高等数学" in p:
                vids.append(p)
    return sorted(vids)

def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    vids = collect_videos(subject)
    # 过滤已完成的（有 srt 的跳过）
    todo = []
    for v in vids:
        base = os.path.splitext(v)[0]
        srt = os.path.join(SRT_DIR, os.path.basename(v).rsplit(".", 1)[0] + ".srt")
        if os.path.exists(srt) and os.path.getsize(srt) > 0:
            continue
        todo.append(v)

    print(f"科目={subject}, 并行数={n_workers}, 待处理={len(todo)} 个视频")

    if not todo:
        print("全部已完成")
        return

    # 把待处理视频轮流分给各 worker（轮询分配）
    queues = [[] for _ in range(n_workers)]
    for i, v in enumerate(todo):
        queues[i % n_workers].append(v)

    procs = []
    for i, q in enumerate(queues):
        if not q:
            continue
        # 每个 worker 跑一个子进程，处理自己的队列
        # 用 --device 1（都是同一张卡，靠多进程并行）
        args = [PY, TRANSCRIBE, "--device", "1"]
        # transcribe.py 接受目录或单文件；这里让每个worker处理一个"文件清单"临时文件
        listfile = os.path.join(TMP, f"worker_{i}.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            f.write("\n".join(q))
        # 用一个新的小驱动脚本处理清单
        driver = os.path.join(TMP, "worker_driver.py")
        p = subprocess.Popen([PY, driver, listfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((i, p, q))

    # 等所有完成
    for i, p, q in procs:
        p.wait()
        print(f"[worker {i}] 完成 {len(q)} 个视频, 退出码 {p.returncode}")

    print("\n[全部完成]")

if __name__ == "__main__":
    main()
