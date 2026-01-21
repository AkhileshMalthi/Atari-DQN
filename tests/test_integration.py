# test_final_integration.py
import os
import torch


def is_valid_pytorch_file(filepath):
    """Check if file is a valid PyTorch checkpoint, not a Git LFS pointer."""
    try:
        # Git LFS pointer files start with "version https://git-lfs.github.com"
        with open(filepath, 'rb') as f:
            header = f.read(50)
            if header.startswith(b'version https://git-lfs'):
                return False
        # Try to load it
        torch.load(filepath, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def test_evaluation_workflow():
    from evaluate import evaluate
    
    # 1. Check if model exists
    if not os.path.exists("final_model.pth"):
        print("Skipping: final_model.pth not found. Train first!")
        return
    
    # 2. Check if it's a valid PyTorch file (not LFS pointer)
    if not is_valid_pytorch_file("final_model.pth"):
        print("Skipping: final_model.pth is not a valid model (likely LFS pointer in CI)")
        return

    # 3. Run a mini-evaluation of 2 episodes
    avg = evaluate(num_episodes=2, record_video=False)
    
    assert isinstance(avg, float), "Evaluation should return a numerical average"
    assert os.path.exists("eval_results.txt"), "Results file was not created"
    print("✅ Final Evaluation Workflow Verified!")


if __name__ == "__main__":
    test_evaluation_workflow()