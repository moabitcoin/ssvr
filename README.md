# ssvr: self-supervised visual representations

Quoting the great machine learning researcher D. Ross in her foundational prior art

```
Upside down
Boy, you turn me
Inside out
And round and round
```


# Use Case

We record tens of thousand hours of drive video data and have to extract their visual semantics.
In this project we explore if we can use the vast amount of unlabeled data for self-supervised visual representation learning.


# Implementation Sketch

Following the work in [https://arxiv.org/abs/1803.07728](https://arxiv.org/abs/1803.07728) we apply 2d rotations by 0, 90, 180, or 270 degrees to images from our vast unlabeled dataset.
We then set up a classification task predicting the rotation as one of four categories (applied 2d rotation).
This surprisingly simple task requires the model to learn semantic feature representations from our vast unlabeled dataset.


# References

- [Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)
- [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)
- [D. Ross - Upside Down](https://en.wikipedia.org/wiki/Upside_Down_(Diana_Ross_song))
