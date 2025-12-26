# LFM2-350M Z-Image Turbo Trainer

An optimized Supervised Fine-Tuning (SFT) toolkit for the **Liquid LFM2-350M** model, specifically designed to generate high-quality, descriptive prompts for **Z-Image Turbo**.
You can find a pretrained model using 60k samples here: https://huggingface.co/marduk191/lfm2-350m-dp-marduk191

<img width="1095" height="845" alt="image" src="https://github.com/user-attachments/assets/85d73920-87ef-425f-9319-a1095c7b85cf" />


## 🚀 Features

- **Liquid LFM2-350M Optimized**: Tailored for the latest 350M model from LiquidAI.
- **RTX 5090 Ready**: Leveraging BF16, TF32, and `torch.compile` for blistering speed.
- **Prompt Expansion Logic**: Fine-tuned to turn simple user descriptions into rich, technical image prompts.
- **Integrated GUI**: Modern dark-themed Tkinter interface for non-technical users.
- **GPU Presets**: One-click optimization for 5090, 4090, 3090, 4080, and 4070.
- **Massive Data Synthesis**: Built-in script to generate up to 60,000+ synthetic image prompts.

## 🛠️ Setup

This repository is optimized for **Windows** (PowerShell) and **RTX GPUs**.

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/marduk191/LFM2-350M_trainer_marduk191.git
   cd LFM2-350M_trainer_marduk191
   ```

2. **Run the Setup Script**:
   This will create a virtual environment, install a CUDA-enabled PyTorch (2.9+), Triton (for Windows), and pre-download the model.

   **Windows:**
   ```powershell
   .\setup.ps1
   ```

   **Linux / macOS:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## 🖥️ Usage

### Launching the GUI
The easiest way to use the trainer is via the interactive GUI:

**Windows (PowerShell):**
```powershell
.\run_gui.ps1
```

**Linux / macOS (Bash):**
```bash
chmod +x run_gui.sh
./run_gui.sh
```

### From the GUI, you can:
1. **Choose your GPU**: Select a preset to auto-configure batch sizes.
2. **Synthesize Data**: Generate thousands of training prompts with one click.
3. **Train**: Monitor real-time logs and stop the session at any time.

## 🧠 Data Format
The trainer expects a `.jsonl` file with the following format:
```json
{"instruction": "...", "input": "simple description", "output": "detailed technical prompt"}
```

## 📂 Project Structure
- **/tools**: Contains the core Python implementation (`train.py`, `gui.py`, `generate_data.py`, `inference.py`).
- **setup.ps1 / setup.sh**: Automated environment setup.
- **run_gui.ps1 / run_gui.sh**: Launch the interactive trainer interface.
- **run_trainer.ps1 / run_trainer.sh**: Command-line training benchmarks.

## 📈 Optimization Details
- **Triton for Windows**: Enables advanced kernel fusion via `torch.compile`.
- **SFTTrainer**: Uses the `trl` library for efficient supervised fine-tuning.
- **AdamW 8-bit**: Memory-efficient optimizer to keep VRAM usage low.

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgements
- [LiquidAI](https://huggingface.co/LiquidAI) for the LFM2 model.
- [Hugging Face](https://huggingface.co/) for the Transformers/TRL stack.
- [Triton Windows](https://github.com/triton-lang/triton) for enabling kernel compilation.
