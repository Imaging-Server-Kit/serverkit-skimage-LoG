from typing import List, Type
from pathlib import Path
import skimage.io
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import imaging_server_kit as serverkit
import skimage.feature
from skimage.exposure import rescale_intensity


class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    image: str = Field(
        ...,
        title="Image",
        description="The input image (2D, 3D).",
        json_schema_extra={"widget_type": "image"},
    )
    min_sigma: float = Field(
        default=5.0,
        title="Min sigma",
        description="Minimum standard deviation of the Gaussian kernel, in pixels.",
        ge=0.1,
        le=100.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.1,
        },
    )
    max_sigma: float = Field(
        default=10.0,
        title="Max sigma",
        description="Maximum standard deviation of the Gaussian kernel, in pixels.",
        ge=0.1,
        le=100.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.1,
        },
    )
    num_sigma: int = Field(
        default=10,
        title="Num sigma",
        description="Number of intermediate sigma values to compute between the min_sigma and max_sigma.",
        ge=1,
        le=100,
        json_schema_extra={
            "widget_type": "int",
            "step": 1,
        },
    )
    threshold: float = Field(
        default=0.1,
        title="Threshold",
        description="Lower bound for scale space maxima.",
        ge=0.01,
        le=1.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.01,
        },
    )
    invert_image: bool = Field(
        default=False,
        title="Dark blobs",
        description="Whether to invert the image before computing the LoG filter.",
        json_schema_extra={"widget_type": "bool"},
    )
    time_dim: bool = Field(
        default=True,
        title="Frame by frame",
        description="Only applicable to 3D images. If set, the first dimension is considered time and the LoG is computed independently for every frame.",
        json_schema_extra={"widget_type": "bool"},
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array


class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str = "skimage-LoG",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        image: np.ndarray,
        max_sigma: float,
        num_sigma: int,
        threshold: float,
        invert_image: bool,
        time_dim: bool,
        min_sigma: float,
        **kwargs,
    ) -> List[tuple]:
        """Runs a LoG detector for blob detection."""
        if invert_image:
            image = -image

        image = rescale_intensity(image, out_range=(0, 1))

        if (image.ndim == 3) & time_dim:
            # Handle a time-series
            points = np.empty((0, 3))
            sigmas = []
            for frame_id, frame in enumerate(image):
                frame_results = skimage.feature.blob_log(
                    frame,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold,
                )
                frame_points = frame_results[:, :2]  # Shape (N, 2)
                frame_sigmas = list(frame_results[:, 2])  # Shape (N,)
                sigmas.extend(frame_sigmas)
                frame_points = np.hstack(
                    (np.array([frame_id] * len(frame_points))[..., None], frame_points)
                )  # Shape (N, 3)
                points = np.vstack((points, frame_points))
            sigmas = np.array(sigmas)
        else:
            results = skimage.feature.blob_log(
                image,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=threshold,
            )
            points = results[:, :2]
            sigmas = results[:, 2]

        points_params = {
            "name": "Detections",
            "opacity": 0.7,
            "face_color": 'sigma',
            "features": {'sigma': sigmas}  # sigmas = numpy array representing the point size
        }

        return [
            (points, points_params, "points"),
        ]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Load one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images


server = Server()
app = server.app


if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
