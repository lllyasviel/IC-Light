# IC-Light

IC-Light is a project to manipulate the illumination of images.

The name "IC-Light" stands for **"Imposing Consistent Light"**. We briefly describe what is "Imposing Consistent Light" at the end of this page.

Currently, we release two types of models: text-conditioned relighting model and background-conditioned model. Both types take foreground images as inputs.

# Get Started

Below script will run the text-conditioned relighting model:

    git clone https://github.com/lllyasviel/IC-Light.git
    cd iclight
    conda create -n iclight python=3.10
    conda activate iclight
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python gradio_demo.py

Or, to use background-conditioned demo:

    python gradio_demo_bg.py

Model downloading is automatic.

# Screenshot

### Text-Conditioned Model

### Background-Conditioned Model

# Imposing Consistent Light

# Model Notes

* **iclight_sd15_fc.safetensors** - The default relighting model, conditioned on text and foreground. You can use initial latent to influence the relighting.

* **iclight_sd15_fcon.safetensors** - Same as "iclight_sd15_fc.safetensors" but trained with offset noise. Note that the default "iclight_sd15_fc.safetensors" outperform this model slightly in a user study. And this is the reason why the default model is the model without offset noise.

* **iclight_sd15_fbc.safetensors** - Relighting model conditioned with text, foreground, and background.

# Cite

    @Misc{iclight,
      author = {Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
      title  = {IC-Light GitHub Page},
      year   = {2024},
    }
