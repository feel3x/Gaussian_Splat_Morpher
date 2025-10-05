# Gaussian Splatting Morphing Tool 

![Gaussian Morphing Demo](docs/demo.gif)

A **CLI and visualization tool** for smoothly interpolating between two or more **Gaussian Splatting Models**.  
It builds intelligent point correspondences and generates intermediate morphs that transition seamlessly.  
Includes both command-line and real-time interactive visualizer modes.

---

## 🚀 Features

- **Interpolate between multiple Gaussian Splatting models**
- **Automatic one-to-one point correspondences** based on spatial and color similarity
- **Spherical linear interpolation (SLERP)** for rotations
- **Optional real-time visualizer** with interactive morphing slider
- **GPU-accelerated (PyTorch)** processing
- **Save interpolated frames** as `.ply` files for animation sequences

---

## 🧰 Installation

Clone the repository and install the base dependencies:

```bash
git clone https://github.com/feel3x/Gaussian_Splat_Morpher.git
cd Gaussian_Splat_Morpher
pip install -r requirements.txt
```

If you want to use the **visualizer**, install the extra dependencies:

```bash
pip install -r requirements_visualizer.txt
```

---

## 🧑‍💻 Usage (CLI Mode)

You can interpolate between `.ply` Gaussian Splat models directly from the command line.

### Basic Example
```bash
python GaussianInterpolator.py -d ./models/ -o ./output/
```

This searches the `./models/` folder for `.ply` files and generates intermediate models in `./output/`.

---

### Specify Individual Models
```bash
python GaussianInterpolator.py -m model1.ply model2.ply model3.ply -o ./output/
```

---

### Adjust Interpolation Settings

| Option | Description | Default |
|--------|--------------|----------|
| `--models_to_create` | Number of intermediate models per pair | `10` |
| `--direct_interpolation_value` | Create a single interpolated model between two specific models (e.g. `1.3`) | `None` |
| `--spatial_weight` | Weight for spatial distance during correspondence | `0.7` |
| `--color_weight` | Weight for color difference during correspondence | `0.3` |
| `--distance_threshold` | Max allowed distance for point matching | `None` |
| `--batch_size` | Size of point batches for matching (reduce for lower VRAM) | `512` |
| `--recenter_models` | Recenter all models before interpolation | `False` |
| `--normalize_scales` | Normalize scales of models | `False` |

---

### Example: Generate 20 Intermediate Morphs

```bash
python GaussianInterpolator.py -m bunny_A.ply bunny_B.ply -o ./morph_output --models_to_create 20
```

---

### Example: Export a Single Interpolated Model at 0.4

```bash
python GaussianInterpolator.py -m face_1.ply face_2.ply -o ./output --direct_interpolation_value 0.4
```

This saves one `.ply` at 40% interpolation between the first and second models.

---

## 🎨 Real-Time Visualizer (Optional)

To explore morphs interactively:

```bash
python visualizer.py -m model1.ply model2.ply
```

This launches a GUI with a **slider** that lets you morph smoothly between loaded Gaussian Splat models in real time.

*(Visualizer requires extra dependencies from `requirements_visualizer.txt`.)*


---

## 🧠 How It Works

1. Loads two or more Gaussian Splat models (`.ply` format)  
2. Builds **point correspondences** between consecutive models  
3. Interpolates:
   - Positions, features, scales, and opacities (linearly)  
   - Rotations using **SLERP**  
4. Handles unmatched points via fade-in/out blending  
5. Outputs intermediate `.ply` models or visualizes them in real time

---

## 📜 License

This project is released under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

**Author:** Felix Hirt  
**Copyright © 2025**

---

## 🌟 Acknowledgements

This project builds upon Gaussian Splatting concepts.

The visualizer uses Nerfstudio's gSplat rasterizer: [GitHub](https://github.com/nerfstudio-project/gsplat)

---

**Enjoy morphing! 🧩**
