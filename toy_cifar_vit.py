from enum import KEEP
from datasets import load_dataset
import lightning as L
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import ViTModel, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTLayer, ViTEmbeddings
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from torch import nn
import torch
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms
from torch.utils.data import DataLoader
from quant_linear import (
    create_quantized_copy_of_model,
    QuantizationMode,
)


class MoEViTLayer(ViTLayer):
    def __init__(self, config, num_experts=4):
        super().__init__(config)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([ViTLayer(config) for _ in range(num_experts)])
        self.gate = nn.Linear(config.hidden_size, num_experts)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        gate_scores = self.gate(hidden_states)
        gate_softmax = nn.functional.softmax(gate_scores, dim=-1)

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states, head_mask, output_attentions)
            expert_outputs.append(expert_output[0])
        
        expert_outputs = torch.stack(expert_outputs, dim=1)

        expert_outputs = expert_outputs.transpose(1, 2)
        outputs = (gate_softmax.unsqueeze(-1) * expert_outputs).sum(dim=1, keepdim=False)
        return (outputs,) + expert_output[1:]


class MoEViTModel(ViTModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.encoder = nn.ModuleList([MoEViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embeddings(pixel_values)
        hidden_states = embedding_output

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)


class MoEViTForImageClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = MoEViTModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, head_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class MoEViTImageClassifier(L.LightningModule):
    def __init__(self, config: ViTConfig, lr=1e-3):
        super().__init__()
        self.model = MoEViTForImageClassification(config)
        self.config = config
        self.lr = lr

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        argmax = output.logits.argmax(dim=1)
        accuracy = (argmax == batch["labels"]).float().mean()
        self.log_dict(
            {
                "tl": loss.item(),
                "ta": accuracy.item(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
            loss = output.loss
            argmax = output.logits.argmax(dim=1)
            accuracy = (argmax == batch["labels"]).float().mean()
            self.log_dict(
                {
                    "vl": loss.item(),
                    "va": accuracy.item(),
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    


config = ViTConfig(
    hidden_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    intermediate_size=256,
    hidden_act="gelu",
    image_size=32,
    patch_size=4,
    num_labels=100,
    num_channels=3,
)


class ViTImageClassifier(L.LightningModule):
    def __init__(self, config: ViTConfig, lr=1e-3):
        super().__init__()
        self.model = ViTForImageClassification(config)
        self.config = config
        self.lr = lr

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        argmax = output.logits.argmax(dim=1)
        accuracy = (argmax == batch["labels"]).float().mean()
        self.log_dict(
            {
                "tl": loss.item(),
                "ta": accuracy.item(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
            loss = output.loss
            argmax = output.logits.argmax(dim=1)
            accuracy = (argmax == batch["labels"]).float().mean()

        self.log_dict(
            {
                "vl": loss.item(),
                "va": accuracy.item(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


dataset = load_dataset("cifar100")

image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

processed_dataset = dataset.map(
    lambda x: {"pixel_values": image_transforms(x["img"]), "labels": x["fine_label"]}
)
processed_dataset = processed_dataset.remove_columns(["fine_label", "img"])
processed_dataset.set_format("torch", columns=["pixel_values", "labels"])


train_dataloader = DataLoader(processed_dataset["train"], batch_size=128)
eval_dataloader = DataLoader(processed_dataset["test"], batch_size=128)

normal_model = ViTImageClassifier(config)
normal_moe_model = MoEViTImageClassifier(config)

one_bit_quantized_model = create_quantized_copy_of_model(
    normal_model, quantization_mode=QuantizationMode.one_bit
)
two_bit_quantized_model = create_quantized_copy_of_model(
    normal_model, quantization_mode=QuantizationMode.two_bit
)
one_bit_moe_quantized_model = create_quantized_copy_of_model(
    normal_moe_model, quantization_mode=QuantizationMode.one_bit
)


choice = input("Enter 1,2,3,4:")
if int(choice) == 1:
    normal_logger = WandbLogger(project="BitNet", name="normal_cifar100")
    normal_trainer = L.Trainer(
        max_epochs=10,
        logger=normal_logger,
    )
    normal_trainer.fit(
        normal_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )
if int(choice) == 2:
    one_bit_logger = WandbLogger(project="BitNet", name="one_bit_cifar100")
    one_bit_trainer = L.Trainer(
        max_epochs=10,
        logger=one_bit_logger,
    )
    one_bit_quantized_model.lr = 1e-4
    one_bit_trainer.fit(
        one_bit_quantized_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )

if int(choice) == 3:
    two_bit_logger = WandbLogger(project="BitNet", name="two_bit_cifar100")
    two_bit_trainer = L.Trainer(
        max_epochs=10,
        logger=two_bit_logger,
    )
    two_bit_quantized_model.lr = 1e-4
    two_bit_trainer.fit(
        two_bit_quantized_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )
if int(choice) == 4:
    one_bit_moe_logger = WandbLogger(project="BitNet", name="one_bit_moe_cifar100")
    one_bit_moe_trainer = L.Trainer(
        max_epochs=10,
        logger=one_bit_moe_logger,
    )
    one_bit_moe_quantized_model.lr = 1e-4
    one_bit_moe_trainer.fit(
        one_bit_moe_quantized_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )