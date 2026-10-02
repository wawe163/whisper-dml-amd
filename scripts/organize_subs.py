# -*- coding: utf-8 -*-
"""字幕归位 + 清理：
1. 把识别产出的 .srt / .txt 复制到对应视频同目录（播放器可自动加载）
2. 清理临时 .wav 文件（识别用的中间产物）
"""
import os, shutil, glob

BASE = r"F:\工作\8.学习规定文件\4考试复习\2专升本\2成考"
SRT_POOL = r"C:\Users\ww\WorkBuddy\2026-09-10-22-53-30\.workbuddy\tmp\asr\srt"
VIDEO_EXT = (".mp4", ".mkv", ".flv", ".avi", ".mov")

def main():
    # 1. 建立视频名 -> 路径 的索引
    videos = {}
    for root, _, fs in os.walk(BASE):
        for f in fs:
            if f.lower().endswith(VIDEO_EXT):
                videos[os.path.splitext(f)[0]] = os.path.join(root, f)

    # 2. 把 srt 池里的字幕复制到视频同目录
    moved = 0
    for src_dir in [SRT_POOL]:
        if not os.path.isdir(src_dir):
            continue
        for f in os.listdir(src_dir):
            name, ext = os.path.splitext(f)
            if ext not in (".srt", ".txt"):
                continue
            if name in videos:
                vdir = os.path.dirname(videos[name])
                dst = os.path.join(vdir, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(src_dir, f), dst)
                    moved += 1
    print(f"复制字幕到视频目录: {moved} 个")

    # 3. 清理临时 wav
    removed = 0
    for root, _, fs in os.walk(BASE):
        for f in fs:
            if f.lower().endswith(".wav"):
                try:
                    os.remove(os.path.join(root, f))
                    removed += 1
                except Exception:
                    pass
    print(f"清理临时 wav: {removed} 个")

    # 4. 统计
    print("\n=== 各科字幕完成情况 ===")
    for subj in ["政治", "英语", "高等数学一"]:
        subj_dir = os.path.join(BASE, subj)
        if not os.path.isdir(subj_dir):
            continue
        mp4 = 0; srt = 0
        for root, _, fs in os.walk(subj_dir):
            for f in fs:
                if f.lower().endswith(".mp4"):
                    mp4 += 1
                elif f.lower().endswith(".srt"):
                    srt += 1
        print(f"  {subj}: {srt}/{mp4} 个视频有字幕")

if __name__ == "__main__":
    main()
