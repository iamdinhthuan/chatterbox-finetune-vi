#!/usr/bin/env python3
"""
Repository cleanup script
Removes unnecessary files and prepares repo for GitHub
"""

import os
import shutil
import glob
from pathlib import Path

def remove_files_and_dirs(patterns):
    """Remove files and directories matching patterns"""
    removed = []
    
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            try:
                if os.path.isfile(match):
                    os.remove(match)
                    removed.append(f"File: {match}")
                elif os.path.isdir(match):
                    shutil.rmtree(match)
                    removed.append(f"Dir: {match}")
            except Exception as e:
                print(f"❌ Failed to remove {match}: {e}")
    
    return removed

def cleanup_repo():
    """Clean up repository for GitHub"""
    print("🧹 Cleaning up repository...")
    
    # Files and directories to remove
    cleanup_patterns = [
        # Python cache
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/.Python",
        "**/build",
        "**/develop-eggs",
        "**/dist",
        "**/downloads",
        "**/eggs",
        "**/.eggs",
        "**/lib",
        "**/lib64",
        "**/parts",
        "**/sdist",
        "**/var",
        "**/wheels",
        "**/*.egg-info",
        "**/.installed.cfg",
        "**/*.egg",
        "**/MANIFEST",
        
        # Jupyter
        "**/.ipynb_checkpoints",
        
        # Environment
        "**/.env",
        "**/.venv",
        "**/env",
        "**/venv",
        "**/ENV",
        "**/env.bak",
        "**/venv.bak",
        
        # IDE
        "**/.vscode",
        "**/.idea",
        "**/*.swp",
        "**/*.swo",
        "**/*~",
        
        # OS
        "**/.DS_Store",
        "**/.DS_Store?",
        "**/._*",
        "**/.Spotlight-V100",
        "**/.Trashes",
        "**/ehthumbs.db",
        "**/Thumbs.db",
        
        # Training outputs (keep structure but remove large files)
        "**/checkpoints",
        "**/logs",
        "**/runs",
        "**/wandb",
        "**/*.log",
        
        # Large model files (should use Git LFS)
        "**/*.safetensors",
        "**/chatterbox-project",
        "**/model.safetensors",
        "**/t3_cfg_vietnamese.safetensors",
        
        # Dataset files (too large for GitHub)
        "**/train.csv",
        "**/val.csv",
        "**/wavs",
        
        # Temporary files
        "**/temp",
        "**/tmp",
        "**/inference_output",
        "**/gradio_outputs",
        
        # Tokenizer intermediate files
        "**/vietnamese_text_corpus.txt",
        "**/tokenizer_vietnamese_new.json",
        
        # Specific temporary files
        "**/simple_test.py",
        "**/test_vietnamese_model.py",
        "**/fix_*.py",
        "**/create_model_config.py",
        "**/gradio_interface.py",
        "**/gradio_local.py",
        "**/gradio_tts_app.py",
        "**/inference_vietnamese_tts.py",
        "**/tokenizer_vi_expanded.json",
        "**/tokenizer_vietnamese.json",
        "**/vietnamese_text.txt",
        "**/beam_config.yaml",
        "**/finetune_t3_beam.py",
    ]
    
    # Remove files
    removed = remove_files_and_dirs(cleanup_patterns)
    
    # Create necessary directories
    directories_to_create = [
        "examples/outputs",
        "docs/images",
        "chatterbox/chatterbox-project",
    ]
    
    for directory in directories_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create placeholder files for empty directories
    placeholder_files = [
        "examples/outputs/.gitkeep",
        "docs/images/.gitkeep",
        "chatterbox/chatterbox-project/.gitkeep",
    ]
    
    for placeholder in placeholder_files:
        with open(placeholder, 'w') as f:
            f.write("# This file keeps the directory in Git\n")
        print(f"✅ Created placeholder: {placeholder}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"Removed {len(removed)} items:")
    for item in removed[:10]:  # Show first 10 items
        print(f"  - {item}")
    if len(removed) > 10:
        print(f"  ... and {len(removed) - 10} more items")
    
    print(f"\n✅ Repository cleanup completed!")

def create_sample_files():
    """Create sample files for documentation"""
    print("\n📝 Creating sample files...")
    
    # Sample reference audio info
    sample_info = """# Sample Files

This directory contains sample files for testing and documentation.

## reference_voice.wav
- Duration: 5-7 seconds
- Format: WAV, 24kHz, mono
- Content: Clear Vietnamese speech
- Usage: Voice cloning reference

To use voice cloning examples, place your reference audio here as 'reference_voice.wav'.

## sample_texts.txt
Contains Vietnamese text samples for testing TTS functionality.
"""
    
    with open("examples/sample_info.md", 'w', encoding='utf-8') as f:
        f.write(sample_info)
    
    # Sample texts
    sample_texts = """Xin chào, tôi là trợ lý AI tiếng Việt.
Hôm nay là một ngày đẹp trời.
Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh.
Việt Nam là một đất nước xinh đẹp với văn hóa phong phú.
Cảm ơn bạn đã sử dụng hệ thống TTS tiếng Việt.
Chúc bạn có một ngày tốt lành và nhiều niềm vui!
Tôi có thể nói tiếng Việt rất tự nhiên và trôi chảy.
Hãy thử nghiệm với nhiều câu khác nhau để test chất lượng.
Công nghệ voice cloning giúp tạo ra giọng nói tự nhiên.
Hệ thống này có thể học và bắt chước giọng nói của bạn."""
    
    with open("examples/sample_texts.txt", 'w', encoding='utf-8') as f:
        f.write(sample_texts)
    
    print("✅ Sample files created")

def main():
    """Main cleanup function"""
    print("🇻🇳 Vietnamese TTS Repository Cleanup")
    print("=" * 50)
    
    # Change to repository root
    repo_root = Path(__file__).parent
    os.chdir(repo_root)
    
    # Run cleanup
    cleanup_repo()
    create_sample_files()
    
    print(f"\n🎉 Repository is ready for GitHub!")
    print(f"\n📋 Next steps:")
    print(f"1. Review the cleaned repository")
    print(f"2. Update README.md with your GitHub username")
    print(f"3. Add your model files to Git LFS (if needed)")
    print(f"4. Commit and push to GitHub")
    print(f"\n🚀 Git commands:")
    print(f"git add .")
    print(f"git commit -m 'Initial commit: Vietnamese TTS Voice Cloning'")
    print(f"git push origin main")

if __name__ == "__main__":
    main()
