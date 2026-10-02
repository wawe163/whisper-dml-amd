# -*- coding: utf-8 -*-
"""worker 驱动：读文件清单，逐个转写（供 batch_parallel.py 的多进程调用）
"""
import sys, os, importlib.util

def main():
    listfile = sys.argv[1]
    with open(listfile, encoding="utf-8") as f:
        files = [l.strip() for l in f if l.strip()]

    # 加载 transcribe.py 的函数（复用同一套逻辑）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transcribe_path = os.path.join(script_dir, "transcribe.py")
    spec = importlib.util.spec_from_file_location("transcribe_mod", transcribe_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["transcribe_mod"] = mod
    # 让 transcribe.py 内的 mel import 生效
    sys.path.insert(0, script_dir)
    spec.loader.exec_module(mod)

    print(f"[worker {os.getpid()}] 加载模型...")
    enc, dec = mod.load_models(1)
    tok = mod.load_tokenizer()

    for fp in files:
        try:
            mod.process_one(enc, dec, tok, fp, 1)
        except Exception as e:
            print(f"  [错误] {os.path.basename(fp)}: {e}", flush=True)

if __name__ == "__main__":
    main()
