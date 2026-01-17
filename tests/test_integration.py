# test_final_integration.py
import os
from evaluate import evaluate

def test_evaluation_workflow():
    # 1. Check if model exists (requires a dummy file if not trained yet)
    if not os.path.exists("final_model.pth"):
        print("Skipping: final_model.pth not found. Train first!")
        return

    # 2. Run a mini-evaluation of 2 episodes
    avg = evaluate(num_episodes=2, record_video=False)
    
    assert isinstance(avg, float), "Evaluation should return a numerical average"
    assert os.path.exists("eval_results.txt"), "Results file was not created"
    print("✅ Final Evaluation Workflow Verified!")

if __name__ == "__main__":
    test_evaluation_workflow()