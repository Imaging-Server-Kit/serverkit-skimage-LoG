![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# Scikit-image LoG detector API in docker

Implementation of a web API server for [Scikit Image's LoG detector](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log).

## Installation with `pip`

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Running tests:

```
pytest
```

## Installation with `docker`

Build the docker image:

```
docker build -t serverkit-skimage-log .
```

Run the server in a container:

```
docker run -it --rm -p 8000:8000 serverkit-skimage-log:latest
```

Running tests:

```
docker run --rm serverkit-skimage-log:latest pytest
```

Pushing the image to `registry.rcp.epfl.ch`:

```
docker tag serverkit-skimage-log registry.rcp.epfl.ch/imaging-server-kit/serverkit-skimage-log
docker push registry.rcp.epfl.ch/imaging-server-kit/serverkit-skimage-log
```