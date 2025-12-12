from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="nyu-visionx/RAE-collections",
  local_dir="models",
  proxies={"https": "http://localhost:7890"},
  max_workers=8,
  resume_download=True,
# endpoint='https://hf-mirror.com'
)