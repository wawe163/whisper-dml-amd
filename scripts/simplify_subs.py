# -*- coding: utf-8 -*-
"""繁转简后处理
Whisper 的中文识别默认偏向输出繁体（如「各位同學」「高等數學」），
本脚本把生成的 .srt / .txt 批量转换为简体中文。

依赖: pip install opencc-python-reimplemented

用法:
    python simplify_subs.py <目录>        # 递归处理目录下所有 srt/txt
    python simplify_subs.py <单个文件>
"""
import os, sys

def main():
    if len(sys.argv) < 2:
        print("用法: python simplify_subs.py <目录或文件>")
        return
    try:
        from opencc import OpenCC
    except ImportError:
        print("请先安装: pip install opencc-python-reimplemented")
        return

    cc = OpenCC("t2s")
    target = sys.argv[1]
    files = []
    if os.path.isdir(target):
        for root, _, fs in os.walk(target):
            for f in fs:
                if f.lower().endswith((".srt", ".txt")):
                    files.append(os.path.join(root, f))
    else:
        files = [target]

    n = 0
    for p in files:
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
            c = cc.convert(t)
            if c != t:
                open(p, "w", encoding="utf-8").write(c)
                n += 1
        except Exception as e:
            print(f"[跳过] {p}: {e}")
    print(f"繁转简完成，修改 {n} 个文件（共扫描 {len(files)} 个）")

if __name__ == "__main__":
    main()
