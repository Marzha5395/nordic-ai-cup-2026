import argparse
import hashlib
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
LLAMA_VERSION = 'b10917'
LLAMA_SHA256 = 'e7d547fb0bbdc64960e58cb57b02ab2a5f754036eed482d1e6ddce54c6fcafe0'
MODEL_REVISIONS = {
    '4B': 'e87f176479d0855a907a41277aca2f8ee7a09523',
    '9B': '3885219b6810b007914f3a7950a8d1b469d598a5',
}
ASR_REVISIONS = {
    'dropbox-dash/faster-whisper-large-v3-turbo': '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf',
    'Systran/faster-whisper-small.en': 'd1d751a5f8271d482d14ca55d9e2deeebbae577f',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODEL_REVISIONS, default='4B')
    parser.add_argument('--asr', choices=ASR_REVISIONS, default='Systran/faster-whisper-small.en')
    runtime_options = parser.add_mutually_exclusive_group()
    runtime_options.add_argument('--cpu-runtime', action='store_true')
    runtime_options.add_argument('--build-cuda', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'models'
    directory.mkdir(exist_ok=True)
    hf_hub_download(
        f'unsloth/Qwen3.5-{args.size}-GGUF', f'Qwen3.5-{args.size}-Q4_K_M.gguf',
        revision=MODEL_REVISIONS[args.size], local_dir=directory,
    )
    snapshot_download(args.asr, revision=ASR_REVISIONS[args.asr],
                      local_dir=directory / args.asr.rsplit('/', 1)[-1],
                      allow_patterns=['*.json', '*.bin', '*.txt'])
    if args.cpu_runtime:
        runtime = ROOT / '.runtime'
        runtime.mkdir(exist_ok=True)
        archive = runtime / f'llama-{LLAMA_VERSION}-bin-ubuntu-x64.tar.gz'
        if not archive.exists():
            subprocess.run([
                'gh', 'release', 'download', LLAMA_VERSION, '--repo', 'ggml-org/llama.cpp',
                '--pattern', archive.name, '--dir', str(runtime),
            ], check=True)
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != LLAMA_SHA256:
            raise ValueError('llama.cpp archive checksum mismatch')
        with tarfile.open(archive) as bundle:
            bundle.extractall(runtime, filter='data')
    if args.build_cuda:
        runtime = ROOT / '.runtime'
        runtime.mkdir(exist_ok=True)
        source = runtime / 'llama.cpp'
        if not source.exists():
            subprocess.run(['gh', 'repo', 'clone', 'ggml-org/llama.cpp', str(source),
                            '--', '--depth', '1', '--branch', LLAMA_VERSION], check=True)
        build = source / 'build'
        subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-DGGML_CUDA=ON',
                        '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_OPENSSL=OFF',
                        '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
                        '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'llama-server', '-j', '4'], check=True)
        print(f'Use LLAMA_SERVER={build / "bin" / "llama-server"}')
    print('Models are ready. No model downloads are performed during prediction.')


if __name__ == '__main__':
    main()
