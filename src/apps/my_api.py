from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel, PositiveInt
from src.utils import my_utils
from src import config
import numpy as np
from fastapi.responses import JSONResponse
from PIL import Image
import io
import imagehash
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

config_ = config.Config()


# `PredictOut` class
class PredictOut(BaseModel):
    success: bool
    message: str
    recommendation: str


app = FastAPI()
loaded_model = my_utils.load_registered_model(
    model_name=config_.train["model_name"], tag="champion"
)
get_scene_embed = loaded_model.get_scene_embed
product_embeddings, index_to_key = my_utils.load_product_embedding(
    config_.data["product_embed_path"]
)
cache = {}


# Health check endpoint
@app.get("/healthy_check", status_code=200)
def healthy_check():
    return {"message": "Fashion recommender is healthy!"}


@app.post("/recsys", response_model=PredictOut)
async def ocr(
    num_recommendations: PositiveInt = Form(default=10, gt=0),  # Default and validation
    file: UploadFile = File(...),  # File upload
):
    response = {}
    f = ""  # Initialize f to avoid NameError in case of failure
    try:
        # Read image from the uploaded file
        logging.info("Read image from the uploaded file")
        scene_bytes = await file.read()

        # Check if the file is a valid image
        logging.info("Check if the file is a valid image")
        try:
            img = Image.open(io.BytesIO(scene_bytes))
            img.verify()
        except Exception:
            return JSONResponse(
                content={"error": "Invalid image file"}, status_code=400
            )
        logging.info("Get the hash of the image")
        img = Image.open(io.BytesIO(scene_bytes))
        img_hash = imagehash.average_hash(img)
        
        logging.info(f"hash: {img_hash}")
        # Get the recommendation from the cache if available, otherwise compute it and store it in the cache
        logging.info(
            "Get the recommendation from the cache if available, otherwise compute it and store it in the cache"
        )
        if img_hash in cache:
            logging.info("Getting result from cache!")
            f = cache[img_hash]
        else:
            logging.info("Making a new recommendation.....")
            scene = np.array([scene_bytes])
            scene_dict = my_utils.generate_embeddings(
                scene, get_scene_embed, 16, "scene", file_type="byte"
            )

            # Get the file format (MIME type)
            mime_type = file.content_type

            # Validate file format
            valid_formats = ["image/jpeg", "image/png"]
            if mime_type not in valid_formats:
                return JSONResponse(
                    content={"error": "Unsupported file format"}, status_code=400
                )

            # Find top-k products for the scene embeddings
            for index, (scene_path, scene_vec) in enumerate(scene_dict.items()):
                scene_embed = np.expand_dims(np.array(scene_vec), axis=0)
                scores_and_indices = my_utils.find_top_k(
                    scene_embed,
                    product_embeddings,
                    num_recommendations,  # Form parameter
                )

                # Save results as HTML (set save=False if you want the HTML returned)
                f = my_utils.export_recommendation_to_html(
                    scene_bytes, mime_type, scores_and_indices, index_to_key, save=False
                )
                cache[img_hash] = f  # Store the result in the cache for future use
                logging.info("Successfully made recommendation")
        response["success"] = True
        response["message"] = "Successfully made recommendation"

    except Exception as e:
        response["success"] = False
        response["message"] = str(e)
    response["recommendation"] = f  # Set to None in case of error

    return response
