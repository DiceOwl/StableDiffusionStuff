# Loopback and Superimpose

Mixes output of img2img with original input image at strength alpha. The result is fed into img2img again (at loop>=2), and this procedure repeats.
First image is result of img2img with no looping.
Small alpha means small changes to the first image, large alpha means large changes.
Tends to sharpen the image, improve consistency, reduce creativity and reduce fine detail.
Does not work so well with the ancestral samplers (euler a).

# Interpolate
Overview: An img2img script to produce in-between images. To that end one defines the interpolation ratio as number between 0 and 1. There are two main applications:

a) Upload a second input image in the area the script provides for that purpose in addition to the primary input image of img2img. Then the script will blend the two input images at the interpolation ratio to base the actual input to img2img on. This way you can transition smoothly(relatively) from one input image to another. It can be useful to put a noise image as image 2. I have provided some interesting noise files for that purpose.

b) The script will search prompt and negative prompt for "<number a>\~<number b>" (that's a tilde between the numbers), and replace this by the linear interpolation of <number a> and <number b> according to the interpolation ratio. So for example "0.5\~1.5" will be replaced by 0.6 at interpolation ratio 0.1, and 1.0 at interpolation ratio 0.5. You can also use negative numbers, and use an arbitrary number of these statements, assuming they do not overlap. The main purpose is to go smoothly from one prompt to another, via "<prompt a>:1\~0 AND <prompt b>:0\~1", so for example "a cat, highly detailed, by greg rutkowski:1\~0 AND a dog, sharp focus, award-winning phot:0\~1" will interpolate from a cat painting to a dog photo.
Both a) and b) can be combined. In both cases all other settings remain the same, in particular the seed does not change (for an eception see extra)

In the interpolate field, you can put a comma seperated list of

  a) numbers ("0, 0.25, 0.5")

  b) ranges ("0-0.5[3]" will produce three numbers, evenly from 0 to 0.5)

Extra:
Var Seed Interpolation: When ticked, the script will interpolate the varseed strength from the Seed.Extra settings from 0 at interpolation ratio 0 to UI value at interpolation ratio 1. If you want to interpolate between two seeds, you can do so by putting the second seed as Var Seed and setting the strength to 1.

Loopback: 
The script can do a second(and third and ...) pass over the produced output. Lets call the basic blend of input image 1 and input image 2 the level0 images. Without loopback, it will simply feed the level0 images into img2img to produce the level1 images, which are the output. With loopback the script will continue for a second round. For that, the level1 images are temporaly blended in range determined by the stride setting. At stride 1, the 4th image will be mixed with the 3rd and 5th image, because these are in range 1. This blended image is then further mixed with the 4th level0 image at ratio alpha. For alpha=0, only the level0 image is used, for alpha=1, only the blended level1 images are used. This is then fed into img2img again to produce the 4th level2 output image. There is one exception: the first and last image are not temporaly blended. Instead, the first level1 image is mixed with the first level0 image at proportion given by the border alpha parameter. The same for the last image. This is so that it is possible to chain multiple interpolations. The seed used for the loopback pass will be the same as the first pass, unless the reuse seed box is unticked. Increase the loopback value for even more passes. Note: This multiplies the number of images which need to be processed with a corresponding compute time cost. See normal.jpg for interpolating a train to a car, with 4 loopbacks, alpha 0.15, border_alpha 0.1. See latent.jpg for the same with interpoplation in latent space.

Paste on mask: Useful for inpainting.
When unticked, all of input image 1 is mixed with all of input image 2. When ticked, instead input image 2 is rescaled to the rectangle incribed by the mask, and then only blended with image 1 inside the mask. Combine with using the cropping tool of the image 2 UI to do a very targeted blending. Hint: Use denoising 0 and interpolate 0.5 to quickly see if you selected a good cropping area.

Interpolate in latent: When ticked, the image interpolation will be performed on the latent representations instead of the pixel space blending. This is somewhat experimental, put seems to lead to better images. More experimentation needed. 
