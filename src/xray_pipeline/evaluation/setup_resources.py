import os
import ssl
import argparse
from pathlib import Path

from xray_pipeline.core.config import PipelineConfig

def download_resources(config_path: Path):
    config = PipelineConfig.from_yaml(config_path)
    
    res_dir = Path(config.paths.eval_resources_dir).resolve()
    nltk_dir = res_dir / "nltk_data"
    stanza_dir = res_dir / "stanza_resources"
    
    nltk_dir.mkdir(parents=True, exist_ok=True)
    stanza_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading evaluation resources to: {res_dir}")

    # Fix SSL Context for certain restrictive networks
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download NLTK safely
    import nltk
    print("Downloading NLTK punkt...")
    nltk.download('punkt', download_dir=str(nltk_dir))

    # Download Stanza safely
    import stanza
    print("Downloading Stanza english model...")
    stanza.download('en', model_dir=str(stanza_dir), verbose=True)
    
    print("Setup Complete! Resources isolated safely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    download_resources(args.config)