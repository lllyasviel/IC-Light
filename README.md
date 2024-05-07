# IC-Light

*There Will Always Be Those Who Dare To Brave The Relighting's Glow!*

# Get Started

    git clone X
    cd iclight
    conda create -n iclight python=3.10
    conda activate iclight
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python gradio_demo.py

Or, to use background-conditioned demo

    python gradio_demo_bg.py

Model downloading is automatic.

# Model Notes

* iclight_sd15_fc.safetensors - The default relighting model, conditioned on text and foreground. You can use initial latent to influence the relighting.

* iclight_sd15_fcon.safetensors - Same as "iclight_sd15_fc.safetensors" but trained with offset noise. Note that the default "iclight_sd15_fc.safetensors" outperform this model slightly in a user study. And this is the reason why the default model is the model without offset noise.

* iclight_sd15_fbc.safetensors - Relighting model conditioned with text, foreground, and background.

# Cite


