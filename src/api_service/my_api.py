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
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

config_ = config.Config()

class PredictOut(BaseModel):
    success: bool
    message: str
    recommendation: dict

# Set up tracer
set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: "fashion-recsys-service"}))
)
tracer = get_tracer_provider().get_tracer("fashion_recsys", "0.0.1")
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
get_tracer_provider().add_span_processor(span_processor)

app = FastAPI()

# Add tracing for model loading
with tracer.start_as_current_span("load_model") as span:
    span.set_attribute("model.name", config_.train["model_name"])
    loaded_model = my_utils.load_registered_model(
        model_name=config_.train["model_name"], tag="champion"
    )
    get_scene_embed = loaded_model.get_scene_embed

with tracer.start_as_current_span("load_product_embeddings") as span:
    span.set_attribute("embeddings.path", config_.data["product_embed_path"])
    product_embeddings, index_to_key = my_utils.load_product_embedding(
        config_.data["product_embed_path"]
    )

cache = {}

@app.get("/healthy_check", status_code=200)
def healthy_check():
    return {"message": "Fashion recommender is healthy!"}

@app.post("/recsys", response_model=PredictOut)
async def ocr(
    num_recommendations: PositiveInt = Form(default=10, gt=0),
    file: UploadFile = File(...),
):
    with tracer.start_as_current_span("recsys_endpoint") as parent_span:
        response = {}
        recommendation_dict = {}
        try:
            # Trace image loading and validation
            with tracer.start_as_current_span("load_and_validate_image") as span:
                span.set_attribute("file.name", file.filename)
                span.set_attribute("file.content_type", file.content_type)
                
                scene_bytes = await file.read()
                
                # Validate image
                try:
                    img = Image.open(io.BytesIO(scene_bytes))
                    img.verify()
                except Exception as e:
                    span.set_attribute("validation.error", str(e))
                    return JSONResponse(
                        content={"error": "Invalid image file"}, status_code=400
                    )
                
                img = Image.open(io.BytesIO(scene_bytes))
                img_hash = imagehash.average_hash(img)
                cache_key = f"{img_hash}_{num_recommendations}"
                span.set_attribute("image.hash", str(img_hash))

            # Check cache
            with tracer.start_as_current_span("check_cache") as span:
                span.set_attribute("cache.key", cache_key)
                cache_hit = cache_key in cache
                span.set_attribute("cache.hit", cache_hit)
                
                if cache_hit:
                    logging.info("Getting result from cache!")
                    recommendation_dict = cache[cache_key]
                else:
                    # Generate embeddings
                    with tracer.start_as_current_span("generate_embeddings") as span:
                        logging.info("Making a new recommendation.....")
                        scene = np.array([scene_bytes])
                        scene_dict = my_utils.generate_embeddings(
                            scene, get_scene_embed, 16, "scene", file_type="byte"
                        )
                        span.set_attribute("embeddings.count", len(scene_dict))

                    # Validate file format
                    mime_type = file.content_type
                    valid_formats = ["image/jpeg", "image/png"]
                    if mime_type not in valid_formats:
                        return JSONResponse(
                            content={"error": "Unsupported file format"}, status_code=400
                        )

                    # Find recommendations
                    with tracer.start_as_current_span("find_recommendations") as span:
                        span.set_attribute("recommendations.count", num_recommendations)
                        for index, (scene_path, scene_vec) in enumerate(scene_dict.items()):
                            scene_embed = np.expand_dims(np.array(scene_vec), axis=0)
                            scores_and_indices = my_utils.find_top_k(
                                scene_embed,
                                product_embeddings,
                                num_recommendations,
                            )

                            recommendation_dict = my_utils.export_recommendation(
                                scene_bytes, mime_type, scores_and_indices, index_to_key, to_html=False
                            )
                            cache[cache_key] = recommendation_dict
                            span.set_attribute("recommendations.success", True)

            response["success"] = True
            response["message"] = "Successfully made recommendation"
            parent_span.set_attribute("response.success", True)

        except Exception as e:
            response["success"] = False
            response["message"] = str(e)
            parent_span.set_attribute("response.success", False)
            parent_span.set_attribute("error.message", str(e))
            
        response["recommendation"] = recommendation_dict

        return response