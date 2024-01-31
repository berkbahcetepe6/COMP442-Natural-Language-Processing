import torch
import torch.nn as nn
import transformers


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = taskmodels_dict

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        
        taskmodels_dict = {}
        shared_encoder = None
        
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(model_name, config=model_config_dict[task_name])

            if shared_encoder is None:
                shared_encoder = model.roberta

            else:
                model.roberta = shared_encoder
            taskmodels_dict[task_name] = model
            
        return cls(shared_encoder, taskmodels_dict)
            
    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)
