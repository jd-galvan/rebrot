![Logo](static/logo.png)

## ✨ Description

`rebrot` is a Python application designed to perform **image inpainting** on photographs using advanced artificial intelligence and image processing models. This project was developed at the **Polytechnic University of Valencia (UPV)** as part of the **Salvem Les Fotos** initiative.

It uses the following technologies:

* 🔍 **YoloV8** for automatic detection of stained or damaged regions.
* 🔬 **SAM2** (Segment Anything Model v2) for stain segmentation. (Alternative 1)
* 🧠 **UNet** for precise segmentation of affected regions. (Alternative 2)
* 🤖 **SegFormer** for precise segmentation of affected regions. (Alternative 3)
* 👤 **LangSAM** for face detection.
* 🎨 **Stable Diffusion Inpainting XL** for image restoration.
* 🏞️ **OpenCV** for image processing.
* 🌐 **Gradio** for building an accessible web interface.

## ⚙️ Requirements

* 🐍 Python >= 3.10
* 🚀 CUDA-compatible GPU (optional, but recommended for better performance)

## 📥 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/jd-galvan/rebrot.git
cd rebrot
```

### 2️⃣ Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 🛠️ Environment Variables Configuration

This project requires certain environment variables to function properly. An example file `.env-example` is provided as a reference.

### 📌 Steps:

1. Copy the `.env-example` file and rename it to `.env`:

   ```bash
   cp .env-example .env
   ```
2. Edit the `.env` file and fill in the following values:

   ```env
   CUDA_DEVICE=cuda:0  # You can set it to "cuda:0", "cuda:1", or whichever GPU you wish to use.
   HUGGINGFACE_HUB_TOKEN=your_token_here
   APP_USER=your_app_username
   APP_PASSWORD=your_app_password
   ```

## 🚀 Usage

To run the application, simply execute the following command:

```bash
python main_yolo.py
```

This will launch a **Gradio** web interface where you can upload and process images to remove stains.

## 🤝 Contributions

Contributions are welcome! If you'd like to improve the project, please:

1. Fork the repository.
2. Create a new branch (`git checkout -b my-new-feature`).
3. Make your changes and commit them.
4. Submit a pull request.

## 📜 License

This project is licensed under the MIT License. For more details, see the `LICENSE` file. 🚀
