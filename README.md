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

(Note that the "Lighting Preference" are just initial latents - eg., if the Lighting Preference is "Left" then initial latent is left white right black.)

---

**Prompt: beautiful woman, detailed face, warm atmosphere, at home, bedroom**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/87265483-aa26-4d2e-897d-b58892f5fdd7)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/148c4a6d-82e7-4e3a-bf44-5c9a24538afc)

---

**beautiful woman, detailed face, neon light, city**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/70db6cf1-973b-4925-849e-0560a4a0e1ec)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/25d6ea24-a736-4a0b-b42d-700fe8b2101e)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dd30387b-0490-46ee-b688-2191fb752e68)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/6c9511ca-f97f-401a-85f3-92b4442000e3)

---

**Prompt: beautiful woman, detailed face, shadow from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/e73701d5-890e-4b15-91ee-97f16ea3c450)

---

**Prompt: beautiful woman, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/ff26ac3d-1b12-4447-b51f-73f7a5122a05)

---

**Prompt: handsome boy, detailed face, neon light, city**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/d7795e02-46f7-444f-93e7-4d6460840437)

---

**Prompt: beautiful woman, detailed face, light and shadow**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/0f376dd1-4339-4a27-9f33-95ae67b61bc4)

---

**Prompt: Buddha, detailed face, sci-fi RGB glowing, cyberpunk**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/68d60c68-ce23-4902-939e-11629ccaf39a)

---

**Prompt: Buddha, detailed face, natural lighting**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1841d23d-0a0d-420b-a5ab-302da9c47c17)

---

**Prompt: toy, detailed face, shadow from window**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dcb97439-ea6b-483e-8e68-cf5d320368c7)

---

**Prompt: toy, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4f78b897-621d-4527-afa7-78d62c576100)

---

**Prompt: dog, magic lit, sci-fi RGB glowing, studio lighting**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1db9cac9-8d3f-4f40-82e2-e3b0cafd8613)

---

**Prompt: mysteriou human, warm atmosphere, warm atmosphere, at home, bedroom**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5d5aa7e5-8cbd-4e1f-9f27-2ecc3c30563a)

### Background-Conditioned Model

The background conditioned model does not require careful prompting. One can just use simple prompts like "handsome man, cinematic lighting".

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/0b2a889f-682b-4393-b1ec-2cabaa182010)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/477ca348-bd47-46ff-81e6-0ffc3d05feb2)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5b0979b6-7027-4d7c-8760-7eb0d7c3b986)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4010052f-e81e-4512-917a-01e5bbeae1c7)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/a35e4c57-e199-40e2-893b-cb1c549612a9)


# Imposing Consistent Light

In HDR space, illumination has a property that all light transports are independent. 

As a result, the blending of appearances of different light sources is equivalent to the appearance with mixed light sources:

Using the above light stage as an example, the two images from the "appearance mixture" and "light source mixture" are consistent (mathematically equivalent in HDR space, ideally).

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
