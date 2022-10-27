import numpy as np
from tqdm import trange
import types

from copy import copy
import modules.scripts as scripts
import gradio as gr
import torch
import numpy as np

import random
from PIL import Image, ImageFilter, ImageOps

from modules import processing, shared, sd_samplers, images, masking
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
import modules.images as images

import re


#re_int = re.compile(r'([+-]?[0-9]*[.]?[0-9]+)[~]([+-]?[0-9]*[.]?[0-9]+)', flags=re.MULTILINE)
re_int = re.compile(r'([+-]?[0-9]*[.]?[0-9]+)([@][0-9]*[.][0-9]+)?[~]([+-]?[0-9]*[.]?[0-9]+)([@][0-9]*[.][0-9]+)?', flags=re.MULTILINE)

re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*")

def apply_prompt_interpolate(p, x):
    def f_interpolate(matchobj):
        y1 = float(matchobj[1])
        #y2 = float(matchobj[2])
        if matchobj[2]:
            a = float(matchobj[2][1:])
        else:
            a = 0
        y2 = float(matchobj[3])
        if matchobj[4]:
            b = float(matchobj[4][1:])
        else:
            b = 1
        if x<= a:
            i = y1
        elif x>= b:
            i = y2
        else:
            i = y1 + (x-a)*(y2-y1)/(b-a)
        return f'{i:.3f}'
    p.prompt = re_int.sub(f_interpolate, p.prompt)
    p.negative_prompt = re_int.sub(f_interpolate, p.negative_prompt)
    pass


def process_int(vals):
    valslist = [x.strip() for x in vals.split(",")]

    valslist_ext = []

    for val in valslist:
        m = re_range_float.fullmatch(val)
        mc = re_range_count_float.fullmatch(val)
        if m is not None:
            start = float(m.group(1))
            end = float(m.group(2))
            step = float(m.group(3)) if m.group(3) is not None else 1

            valslist_ext += np.arange(start, end + step, step).tolist()
        elif mc is not None:
            start = float(mc.group(1))
            end   = float(mc.group(2))
            num   = int(mc.group(3)) if mc.group(3) is not None else 1
            
            valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
        else:
            valslist_ext.append(val)

        valslist = valslist_ext

    valslist = [float(x) for x in valslist]

    return valslist




def hijack_init(self, all_prompts, all_seeds, all_subseeds):
    image_mask_bak = self.image_mask
    self.old_init(all_prompts, all_seeds, all_subseeds)
    
    # imgs = []
    # for img in self.init_images:
        # image = img.convert("RGB")

        # if crop_region is None:
            # image = images.resize_image(self.resize_mode, image, self.width, self.height)

        # if self.image_mask is not None:
            # image_masked = Image.new('RGBa', (image.width, image.height))
            # image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

            # self.overlay_images.append(image_masked.convert('RGBA'))

        # if crop_region is not None:
            # image = image.crop(crop_region)
            # image = images.resize_image(2, image, self.width, self.height)

        # if self.image_mask is not None:
            # if self.inpainting_fill != 1:
                # image = masking.fill(image, latent_mask)

        # if add_color_corrections:
            # self.color_corrections.append(setup_color_correction(image))

        # image = np.array(image).astype(np.float32) / 255.0
        # image = np.moveaxis(image, 2, 0)

        # imgs.append(image)
        
    def toLatent(img):
        res = img.convert("RGB") #should be already, but whatever
        #res = images.resize_image(self.resize_mode, image2, self.width, self.height)
        res = np.array(res).astype(np.float32) / 255.0
        res = np.moveaxis(res, 2, 0)

        batch_res = np.expand_dims(res, axis=0).repeat(self.batch_size, axis=0)
        
        res = torch.from_numpy(batch_res)
        res = 2. * res - 1.

        res = res.to(shared.device)
        res = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(res))
        
        return res
    
    if self.image_mask and self.inpaint_full_res:
        x,y,w,h = self.paste_to
        crop_region = (x,y,x+w,y+h)
        # image_mask_bak = image_mask_bak.convert('L')

        # if self.inpainting_mask_invert:
            # image_mask_bak = ImageOps.invert(image_mask_bak)

        # if self.mask_blur > 0:
            # image_mask_bak = image_mask_bak.filter(ImageFilter.GaussianBlur(self.mask_blur))
        # mask = image_mask_bak.convert('L')
        # crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
        # crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
        latent2 = toLatent(images.resize_image(2, self.init_img2.crop(crop_region), self.width, self.height))
    else:
        latent2 = toLatent(self.init_img2)
    self.init_latent = self.init_latent*(1.-self.interpolate_ratio) + latent2*self.interpolate_ratio
    del latent2
    if self.mixin_img:
        self.init_latent = self.init_latent*(1.-self.mixin_ratio)
        self.mixin_ratio /= len(self.mixin_img)
        for i in self.mixin_img:
            if self.image_mask and self.inpaint_full_res:
                latent_i = toLatent(images.resize_image(2, i.crop(crop_region), self.width, self.height))
            else:
                latent_i = toLatent(i)
            self.init_latent = self.init_latent + latent_i*self.mixin_ratio
            del latent_i
        #latent_mixin = toLatent(self.mixin_img)
        #self.init_latent = self.init_latent*(1.-self.mixin_ratio) + latent_mixin*self.mixin_ratio
        #del latent_mixin



class Script(scripts.Script):
    def title(self):
        return "Interpolate"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        init_img2 = gr.Image(label="alternate img2img imgage", elem_id="img2img_image_alternate", show_label=False, source="upload", interactive=True, type="pil")     
        
        i_values = gr.Textbox(label="interpolation values", lines=1)
        
        def gr_show(visible=True):
            return {"visible": visible, "__type__": "update"}

        def change_visibility(show):
            return {comp: gr_show(show) for comp in loopback_vis}
            
        loopback_vis = []
        
        loopback_toggle = gr.Checkbox(label='Loopback', value=False)
        
        with gr.Box(visible=False) as loopback_box:           
            loopback_vis.append(loopback_box)
            loopback_loops = gr.Slider(minimum=1, maximum=32, step=1, label='Refinement loops', value=1)
            loopback_alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.005, label='Loopback alpha', value=0.2)
            border_alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.005, label='Border alpha', value=0.1)
            blend_strides = gr.Slider(minimum=0, maximum=32, step=1, label='Blending strides', value=1)
            reuse_seed = gr.Checkbox(label='Reuse Seed', value=True)
        
        with gr.Row() as settings_row:
            one_grid = gr.Checkbox(label='One grid', value=True)
            interpolate_varseed = gr.Checkbox(label='Interpolate VarSeed', value=False)
            paste_on_mask = gr.Checkbox(label='Paste on mask', value=False)
            inpaint_all = gr.Checkbox(label='Inpaint all', value=False)
            interpolate_latent = gr.Checkbox(label='Interpolate in latent', value=False)
            
        loopback_toggle.change(change_visibility, show_progress=False, inputs=[loopback_toggle], outputs=loopback_vis) 

            
        return [init_img2, i_values, loopback_alpha, border_alpha, loopback_loops, blend_strides, loopback_toggle, reuse_seed, one_grid, interpolate_varseed, paste_on_mask, inpaint_all, interpolate_latent]

    def run(self, p, init_img2, i_values, loopback_alpha, border_alpha, loopback_loops, blend_strides, loopback_toggle, reuse_seed, one_grid, interpolate_varseed, paste_on_mask, inpaint_all, interpolate_latent):
        processing.fix_seed(p)
        init_seed = p.seed
        tick_seed = init_seed + 1
        batch_count = p.n_iter
        
        p.extra_generation_params = {}

        p.batch_size = 1
        #batch_count = 1
        n = 0
        p.n_iter = 1

        output_images, info = None, "info test"
        initial_seed = p.seed
        initial_info = None
        initial_prompt = p.prompt
        #initial_info = create_infotext(p, p.prompt, [p.seed], [p.subseed], [])
        var_seed_strength = p.subseed_strength

        grids = []
        all_images = []
        all_images_grid = []

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
        
        base_prompt = p.prompt
        base_negative_prompt = p.negative_prompt
        
        init_img = p.init_images[0]
        init_mask = None
        
        paste_full_res = True
        
        #cropping not supported?
        crop_region = None
        if p.image_mask is None:
            init_img = images.resize_image(p.resize_mode, p.init_images[0], p.width, p.height)
            if init_img2:
                init_img2 = images.resize_image(p.resize_mode, init_img2, p.width, p.height)
        else: 
            init_mask = p.image_mask.convert('L')

            if p.inpainting_mask_invert:
                init_mask = ImageOps.invert(init_mask)

            init_unblurred_mask = init_mask

            if p.mask_blur > 0:
                init_mask = init_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
            
            if p.inpaint_full_res:
                #init_img = p.init_images[0].crop(crop_region)
                #init_img = images.resize_image(2, init_img, p.width, p.height)
                init_img = p.init_images[0]
                if init_img2:
                    if paste_on_mask:
                        init_img_copy = init_img.copy()
                        x1,y1,x2,y2 = masking.get_crop_region(np.array(init_mask),0)
                        init_img_copy.paste(images.resize_image(p.resize_mode, init_img2, x2-x1, y2-y1), (x1,y1))
                        init_img2 = init_img_copy
                    else:
                        w, h = init_img.size
                        init_img2 = images.resize_image(p.resize_mode, init_img2, w, h)
            else:
                init_img = images.resize_image(p.resize_mode, p.init_images[0], p.width, p.height)
                init_mask = images.resize_image(p.resize_mode, init_mask, p.width, p.height)
                if init_img2:
                    if paste_on_mask:
                        init_img_copy = init_img.copy()
                        x1,y1,x2,y2 = masking.get_crop_region(np.array(init_mask),0)
                        init_img_copy.paste(images.resize_image(p.resize_mode, init_img2, x2-x1, y2-y1), (x1,y1))
                        init_img2 = init_img_copy
                    else:
                        init_img2 = images.resize_image(p.resize_mode, init_img2, p.width, p.height)
                
                p.image_mask = init_mask
        if not init_img2:
            init_img2 = init_img
        
        
        history = []
        
        x = process_int(i_values)
        
        state.job_count = len(x) * batch_count * ( loopback_loops + 1 if loopback_toggle else 1 )
    
        

        
        def process_list(img_in):
            res = []
            nonlocal initial_info
            
            for i in range(len(x)):
            
                pc = copy(p)
                apply_prompt_interpolate(pc, x[i])
                
                if interpolate_varseed:
                    pc.subseed_strength = var_seed_strength*x[i]
                    
                    
                   
                if interpolate_latent:
                    pc.old_init = pc.init
                    pc.init = types.MethodType(hijack_init,pc)
                    pc.init_img2 = init_img2
                    if img_in and i!=0 and i!=len(x)-1:  #mix with previous level and neighbors
                        pc.mixin_ratio = loopback_alpha
                        pc.mixin_img = [ img_in[j] for j in set(range(len(img_in))) & set(range(i-blend_strides,i+blend_strides+1)) ]
                    elif img_in:
                        pc.mixin_ratio = border_alpha
                        pc.mixin_img = [ img_in[i] ]    #no sideways blending
                    else:
                        pc.mixin_img = None
                    pc.interpolate_ratio = x[i]
                    pc.init_images = [init_img]
                else:
                    pc.init_images = [img_in[i]]
                    
                pc.n_iter = 1
                pc.batch_size = 1
                pc.do_not_save_grid = True
                if inpaint_all:
                    pc.image_mask = None

                if opts.img2img_color_correction:
                    pc.color_corrections = initial_color_corrections

                state.job = f"Iteration {i + 1}/{len(x)}, batch {n + 1}/{batch_count}"

                processed = processing.process_images(pc)
                
                if not initial_info:
                    initial_info = processed.info
                
                if init_mask and not inpaint_all:#test
                    res.append( Image.composite(processed.images[0], init_img, init_mask) )
                else:
                    res.append(processed.images[0])
            
            if not one_grid:
                grid = images.image_grid(res, rows=1)
                if opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

                grids.append(grid)
            
            return res
            
        def blend_images(base, pre, alpha, border_alpha, strides):
            res = []
            for i in range(len(base)):
                if i == 0 or i == len(base)-1 or strides == 0:
                    res.append(Image.blend(base[i], pre[i], border_alpha))
                elif 0:
                    temp = Image.blend(pre[i-1], pre[i+1], 0.5)
                    blend = Image.blend(pre[i], temp, 0.5)
                    res.append(Image.blend(base[i], blend, alpha))
                elif 1:
                    l = min(min(i,strides),min(len(base)-i-1,strides))
                    #indices = list(set(range(i-l,i+l+1)) & set(range(len(base))) - set([i]))
                    indices = list(set(range(i-l,i+l+1)) & set(range(len(base))))
                    temp = np.asarray(pre[i])
                    blend = np.zeros_like(temp)
                    a = 1/len(indices)
                    for j in indices:
                        blend = blend + np.asarray(pre[j])*a
                    #blend = temp*0.5 + blend*0.5
                    res.append(Image.blend(base[i], Image.fromarray(blend.astype(np.uint8)), alpha))    
                else:   #get 0.5 pre[i] plus an even mixture of all other images in range, has rounding issues
                    indices = list(set(range(i-strides,i+strides+1)) & set(range(len(base))) - set([i]))
                    temp = pre[i]
                    a = 0.5/len(indices)
                    for n,j in enumerate(indices):
                        temp = Image.blend(temp, pre[j], a * (1-a)**(n+1-len(indices)))
                    res.append(Image.blend(base[i], temp, alpha))    
            return res
                  
            
        for n in range(batch_count):
            if interpolate_latent:
                level0 = [init_img for i in x]
            elif init_mask:
                level0 = [Image.composite(Image.blend(init_img, init_img2, min(1,max(0,i))), init_img, init_mask) for i in x]
            else:
                level0 = [Image.blend(init_img, init_img2, min(1,max(0,i))) for i in x]
            
            if interpolate_latent:
                level1 = process_list(None) #blending done in hijack_init
            else:
                level1 = process_list(level0)
            all_images_grid += level1
            all_images += level1
     
            if loopback_toggle:
                cur_level = level1
                for i in range(loopback_loops):
                    if not reuse_seed:
                        p.seed = p.seed + 1
                        p.subseed = p.subseed + 1
                    # if interpolate_latent:
                        # if init_mask:
                            # if crop_region is None:
                                # cur_level_resized = [ images.resize_image(p.resize_mode, j, p.width, p.height) for j in cur_level ]
                            # else:
                                # cur_level_resized = [ j.crop(crop_region) for j in cur_level ]
                                # cur_level_resized = [ images.resize_image(p.resize_mode, j, p.width, p.height) for j in cur_level_resized ]
                        # cur_level = process_list( cur_level_resized )   
                    # else:
                    cur_level = process_list( blend_images(level0, cur_level, loopback_alpha, border_alpha, blend_strides) )
                    all_images += cur_level
                    all_images_grid += cur_level
            
            p.seed = p.seed + 1
            p.subseed = p.subseed + 1
            

        if one_grid:
            grid = images.image_grid(all_images_grid, rows=batch_count*(loopback_loops+1 if loopback_toggle else 1))
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

                grids.append(grid)
                
        if opts.return_grid:
            all_images = grids + all_images
            
        all_seeds = [initial_seed  for i in all_images]
        all_infos = [initial_info  for i in all_images]
        all_prompts = [initial_prompt for i in all_images]
        processed = Processed(p, all_images, seed=initial_seed, info=initial_info, all_seeds=all_seeds, all_prompts=all_prompts, infotexts=all_infos )

        return processed
