"""Capture the exact execution environment without network access."""
from __future__ import annotations
import argparse,json,platform,subprocess,sys
from pathlib import Path
from .io import atomic_json

def command(*args):
    try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as exc:return f"UNAVAILABLE: {exc}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    import torch
    payload={"python":sys.version,"platform":platform.platform(),"executable":sys.executable,"pip_freeze":command(sys.executable,"-m","pip","freeze").splitlines(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"cuda_available":torch.cuda.is_available(),"gpu_count":torch.cuda.device_count(),"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"nvidia_smi":command("nvidia-smi"),"git_commit":command("git","-C","/workspace/base_profiling","rev-parse","HEAD")}
    atomic_json(args.output,payload); print(json.dumps({k:v for k,v in payload.items() if k not in {"pip_freeze","nvidia_smi"}}))
if __name__=="__main__":main()
