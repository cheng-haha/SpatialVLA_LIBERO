import os
import argparse
from pathlib import Path
import shutil
import os
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser("Huggingface AutoModel Tesing")
parser.add_argument("--model_name_or_path", default="", help="pretrained model name or path.")
parser.add_argument("--test", action="store_true", help="run test.")
parser.add_argument("--replace", action="store_true", help="replace imports.")
parser.add_argument("--num_images", type=int, default=1, help="num_images for testing.")


args = parser.parse_args()
if __name__ == "__main__":
    model_name_or_path = Path(args.model_name_or_path)
    # replace imports
    if args.replace:
        # copy file
        copy_files = list(model_name_or_path.parent.glob("*.py")) + list(model_name_or_path.parent.glob("*.json"))
        copy_files = [
            "action_tokenizer.py", 
            "dataset_statistics.json", 
            "preprocessor_config.json", 
            "processor_config.json", 
            "test_huggingface.py",
            "bin_policy.json", 
            "gaussian_statistic.json", 
            "processing_spatialvla.py"
        ]
        for file in copy_files: 
            try:
                shutil.copy(model_name_or_path.parent / file, model_name_or_path)
            except:
                pass
    
        # import_keys = ["from model.vision", "from model.action_tokenizer", "from model"]
        # for key in import_keys: os.system(f"find {model_name_or_path} -name '*.py' -exec sed -i 's/{key}/from /' {{}} +")
        
        # os.system(f"find {model_name_or_path} -name '*.py' -exec sed -i 's/from ..constants/from .constants/' {{}} +")
    
    # start testing!
    if args.test:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        print(processor.statistics)

        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
        # model.language_model.config._attn_implementation = model.vision_tower.config._attn_implementation = "flash_attention_2"
        # print(f"🔥 language model {model.language_model.config._attn_implementation}, vision model: {model.vision_tower.config._attn_implementation}")
        # model.language_model.config._attn_implementation_internal = model.vision_tower.config._attn_implementation_internal = "flash_attention_2"
        # print(f"🔥 language model {model.language_model.config._attn_implementation}, vision model: {model.vision_tower.config._attn_implementation}")

        # model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
        # print(f"Model img_context_token_id = {model.img_context_token_id}")
        try: print(f"Model spatial_embed_tokens {model.spatial_embed_tokens.weight}")
        except: pass
        # print(f"language token {model.language_model.model.embed_tokens.weight}")
        # print(f"language lm_head {model.language_model.lm_head.weight}")

        image = Image.open("test/example3.png").convert("RGB")
        images = [image] * args.num_images
        prompt = "What action should the robot take to pick the cpu?"
        inputs = processor(images=images, text=prompt, unnorm_key="bridge_orig/1.0.0", return_tensors="pt")
        print(inputs)
        
        generation_outputs = model.predict_action(inputs)
        print(generation_outputs, processor.batch_decode(generation_outputs))

        actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
        print(actions)
    
    print("DONE!")
