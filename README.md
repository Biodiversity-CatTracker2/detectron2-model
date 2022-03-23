# detectron2-model

[source](https://github.com/facebookresearch/detectron2)

## Installation

```
pip show -r requirements.txt
```

```python
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ",
      CUDA_VERSION)  # Install detectron2 that matches this pytorch version
```

```
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
```
