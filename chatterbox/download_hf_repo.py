import os

from huggingface_hub import snapshot_download

CHATTERBOX_PROJECT = "./chatterbox-project"


def download_hf_repo():
    repo_name = "ResembleAI/chatterbox"
    repo_home_weights = os.path.join(CHATTERBOX_PROJECT, "chatterbox_weights")
    snapshot_download(repo_name, local_dir_use_symlinks=False, local_dir=repo_home_weights)


if __name__ == "__main__":
    download_hf_repo.local()
