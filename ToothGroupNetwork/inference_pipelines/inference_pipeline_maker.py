from models.modules.tsegnet import TSegNetModule
import torch

def make_inference_pipeline(model_name, ckpt_path_ls):
    if model_name=="tsegnet":
        from inference_pipelines.inference_pipeline_tsegnet import InferencePipeLine
        inference_config = {
            "model_info":{
                "model_parameter" :{
                    "input_feat": 6,
                    "stride": [1, 4, 4, 4, 4],
                    "nstride": [2, 2, 2, 2],
                    "nsample": [36, 24, 24, 24, 24],
                    "blocks": [2, 3, 4, 6, 3],
                    "block_num": 5,
                    "planes": [32, 64, 128, 256, 512],
                    "crop_sample_size": 3072,
                },
            },
            "run_tooth_segmentation_module": True
        }

        module = TSegNetModule(inference_config)
        module.load_state_dict(torch.load(ckpt_path_ls[0]))
        module.cuda()
        return InferencePipeLine(module)
        
    elif model_name=="tgnet":
        from inference_pipelines.inference_pipeline_tgn import InferencePipeLine
        
        # We define the 32-channel, 5-block architecture once.
        main_arch_params = {
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        }
        
        inference_config = {
            "fps_model_info":{
                "model_parameter": main_arch_params,
                "load_ckpt_path": ckpt_path_ls[0]
            },
            "boundary_model_info":{
                # THE FIX: We tell the boundary model to use the exact same 32-channel architecture 
                # so that it can successfully load the duplicated 32-channel checkpoint!
                "model_parameter": main_arch_params,
                "load_ckpt_path": ckpt_path_ls[1]
            },
            "boundary_sampling_info":{
                "bdl_ratio": 0.7,
                "num_of_bdl_points": 20000,
                "num_of_all_points": 24000,
            },
        }
        return InferencePipeLine(inference_config)
        
    elif model_name=="pointnet":
        from inference_pipelines.inference_pipeline_sem import InferencePipeLine
        from models.modules.pointnet import PointFirstModule
        module = PointFirstModule({})
        module.load_state_dict(torch.load(ckpt_path_ls[0]))
        module.cuda()
        return InferencePipeLine(module)
        
    elif model_name=="pointnetpp":
        from inference_pipelines.inference_pipeline_sem import InferencePipeLine
        from models.modules.pointnet_pp import PointPpFirstModule 
        module = PointPpFirstModule({})
        module.load_state_dict(torch.load(ckpt_path_ls[0]))
        module.cuda()
        return InferencePipeLine(module)
        
    elif model_name=="dgcnn":
        from inference_pipelines.inference_pipeline_sem import InferencePipeLine
        from models.modules.dgcnn import DGCnnModule
        module = DGCnnModule({})
        module.load_state_dict(torch.load(ckpt_path_ls[0]))
        module.cuda()
        return InferencePipeLine(module)
        
    elif model_name=="pointtransformer":
        inference_config = {
            "model_info":{
                "model_parameter" :{
                    "input_feat": 6,
                    "stride": [1, 4, 4, 4, 4],
                    "nstride": [2, 2, 2, 2],
                    "nsample": [36, 24, 24, 24, 24],
                    "blocks": [2, 3, 4, 6, 3],
                    "block_num": 5,
                    "planes": [32, 64, 128, 256, 512],
                    "crop_sample_size": 3072,
                },
            },
        }
        from inference_pipelines.inference_pipeline_sem import InferencePipeLine
        from models.modules.point_transformer import PointTransformerModule
        module = PointTransformerModule(inference_config["model_info"])
        module.load_state_dict(torch.load(ckpt_path_ls[0]))
        module.cuda()
        return InferencePipeLine(module)
    else:
        # Fixing the bad Python 2 exception so it throws a clean Python 3 error
        raise ValueError(f"Undefined model: {model_name}")