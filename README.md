# ImageMania: Reverse Image Search with Streamlit & CLIP

This app lets you perform reverse image search on your own image collection using OpenAI's CLIP model. Upload a query image, and the app finds the most visually similar images from your dataset.

## Features
- Upload a query image via the web interface
- Embeds all images in the `images/` folder (including subfolders)
- Uses CLIP for high-quality image embeddings
- Displays the top 5 most similar images

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ImageMania
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your images**
   - Place your images (JPG, PNG, JPEG) inside the `images/` folder.
   - You can organize images in subfolders as well (e.g., `images/cats/`, `images/dogs/`).

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Use the app**
   - Open the provided local URL in your browser.
   - Upload a query image.
   - View the top 5 most similar images from your dataset.

## Notes
- The first run may take some time as it computes embeddings for all images.
- Embeddings are cached for faster subsequent searches.
- For best results, use a GPU (if available) by modifying the code to use `device="cuda"` in the CLIP model loading section.

## Sample Data

This repository includes a sample shoe image dataset to help you get started quickly:

> Yogesh Singh. (2021). Shoe Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2531274

The images are already placed in the `images/` folder. You can use these for testing or replace them with your own images.

## License
MIT 