# StableDiffusionStuff

Loopback and Superimpose
Mixes output of img2img with original input image at strength alpha. The result is fed into img2img again (at loop>=2), and this procedure repeats.
First image is result of img2img with no looping.
Small alpha means small changes to the first image, large alpha means large changes.
Tends to sharpen the image, improve consistency, reduce creativity and reduce fine detail.
Does not work so well with the ancestral samplers (euler a).
