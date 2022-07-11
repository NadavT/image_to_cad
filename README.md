# Image to cad

## Goals

Scripts to convert a given image to format a cad could use.

## Resources

Find resources [here](https://drive.google.com/drive/folders/1ql_MQ4TBghVFClZZAGk84Ai-Pe-QEuif?usp=sharing).
Put them in the resources dir.

## How to run

It is recommended to use a venv

### Installation

```sh
pip install -r requirements.txt
```

### Running

run (if an image file is not provided the `xPhys.ppm` file in the resources will be picked)

```sh
# Optional: activate the venv
python image_to_cad.py
```

press any key to continue after image loading, press escape to exit after the drawing is done.

### Help

run `python image_to_cad.py -h` to get a helpful message on the parameters.
