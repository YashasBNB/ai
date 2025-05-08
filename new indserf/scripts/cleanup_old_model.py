import os

MODEL_PATH = 'models/pattern_autoencoder.pth'
OLD_MODEL_PATH = 'models/pattern_autoencoder_old.pth'

def cleanup_old_checkpoint():
    if os.path.exists(MODEL_PATH):
        # If an old backup already exists, remove it
        if os.path.exists(OLD_MODEL_PATH):
            os.remove(OLD_MODEL_PATH)
        os.rename(MODEL_PATH, OLD_MODEL_PATH)
        print(f"Moved old checkpoint to {OLD_MODEL_PATH}")
    else:
        print(f"No existing checkpoint at {MODEL_PATH}; nothing to clean up.")

if __name__ == "__main__":
    cleanup_old_checkpoint()
