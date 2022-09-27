import numpy as np
from tqdm import trange

import modules.scripts as scripts
import gradio as gr

from PIL import Image

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
import modules.images as images

class Script(scripts.Script):
    def title(self):
        return "Loopback and Superimpose"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        def gr_show(visible=True):
            return {"visible": visible, "__type__": "update"}

        def change_visibility(show):
            return {comp: gr_show(show) for comp in superimpose_extra}
            
        superimpose_extra = []
    
        loops = gr.Slider(minimum=1, maximum=32, step=1, label='Loops', value=4)
        superimpose = gr.Slider(minimum=0.0, maximum=0.3, step=0.005, label='Superimpose alpha', value=0.09)
        
        superimpose_toggle_extra = gr.Checkbox(label='Show extra settings', value=False)

        with gr.Row(visible=False) as superimpose_extra_row:
            superimpose_extra.append(superimpose_extra_row)
            with gr.Box() as superimpose_extra_box1:
                reuse_seed = gr.Checkbox(label='Reuse seed', value=True, visible=True)
                one_grid = gr.Checkbox(label='One grid', value=False, visible=True)
            with gr.Box() as superimpose_extra_box2:
                denoising_strength_change_factor = gr.Slider(minimum=0.2, maximum=1.5, step=0.01, label='Denoising strength change factor', value=1)
                cfg_change_factor = gr.Slider(minimum=0, maximum=1, step=0.01, label='CFG decay factor', value=0)
                cfg_change_target = gr.Slider(minimum=1, maximum=30, step=0.5, label='CFG target', value=7)

        superimpose_toggle_extra.change(change_visibility, show_progress=False, inputs=[superimpose_toggle_extra], outputs=superimpose_extra)        

        return [loops, superimpose, reuse_seed, denoising_strength_change_factor, cfg_change_factor, cfg_change_target, one_grid, superimpose_toggle_extra]

    def run(self, p, loops, superimpose, reuse_seed, denoising_strength_change_factor, cfg_change_factor, cfg_change_target, one_grid, superimpose_toggle_extra):
        processing.fix_seed(p)
        batch_count = p.n_iter
        #batch_count = 1
        p.extra_generation_params = {
            "Superimpose alpha": superimpose,
            "Loop count": loops
        }

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        grids = []
        all_images = []
        state.job_count = loops * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
        
        cfg_start = p.cfg_scale
        cfg_target = cfg_change_target
        
        init_img_bak = p.init_images
        
        history = []

        for n in range(batch_count):
            history = []
            p.init_images = init_img_bak
            
            #cropping not supported
            crop_region = None
            if crop_region is None:
                base_img = images.resize_image(p.resize_mode, p.init_images[0], p.width, p.height)
            if crop_region is not None:
                base_img = p.init_images[0].crop(crop_region)
                base_img = images.resize_image(2, base_img, p.width, p.height)
                
            p.cfg_scale = cfg_start

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True

                if opts.img2img_color_correction:
                    p.color_corrections = initial_color_corrections

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"

                processed = processing.process_images(p)

                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                init_img = Image.blend(processed.images[0], base_img, 1-superimpose)

                p.init_images = [init_img]
                if not reuse_seed:
                    p.seed = processed.seed + 1
                #p.extra_generation_params["Loop count"] = i+1
                    
                p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
                p.cfg_scale = cfg_target + (p.cfg_scale-cfg_target)*(1-cfg_change_factor)
                history.append(processed.images[0])

            if not one_grid:
                grid = images.image_grid(history, rows=1)
                if opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

                grids.append(grid)
            all_images += history
            p.seed = p.seed + 1

        if one_grid:
            grid = images.image_grid(all_images, rows=batch_count)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

                grids.append(grid)

        if opts.return_grid:
            all_images = grids + all_images
            
        p.init_images = init_img_bak

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
