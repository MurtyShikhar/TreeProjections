import torch
import torch.nn
from typing import Dict, Tuple
from models.encoder_decoder import add_eos
from models.transformer_enc_dec import TransformerResult
from ..model_interface import ModelInterface
import layers

from ..encoder_decoder import EncoderDecoderResult


class TransformerDecOnlyInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        l = l.sum() / mask.sum()
        return l

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], train_eos: bool = True
    ) -> EncoderDecoderResult:
        out_len = data["out_len"].long()
        out_with_eos = add_eos(
            data["out"].transpose(0, 1), data["out_len"], self.model.decoder_sos_eos
        ).transpose(0, 1)
        out_len += 1

        res = self.model(
            data["in"],
            data["in_len"].long(),
            out_with_eos,
            out_len,
            teacher_forcing=self.model.training,
            max_len=out_len.max().item(),
        )

        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(
            out_with_eos.shape[1], out_len if train_eos else (out_len - 1)
        ).transpose(0, 1)

        loss = self.loss(res, out_with_eos.transpose(0, 1), len_mask)
        return EncoderDecoderResult(res.data, res.length, loss)


class TransformerEncDecInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        l = l.sum() / mask.sum()
        return l

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], train_eos: bool = True
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long()
        out_len = data["out_len"].long()
        in_with_eos = add_eos(data["in"], data["in_len"], self.model.encoder_eos)
        out_with_eos = add_eos(data["out"], data["out_len"], self.model.decoder_sos_eos)
        in_len += 1
        out_len += 1

        res = self.model(
            in_with_eos.transpose(0, 1),
            in_len,
            out_with_eos.transpose(0, 1),
            out_len,
            teacher_forcing=self.model.training,
            max_len=out_len.max().item(),
        )

        if self.model.mode == "mlm":
            ### take the in_len mask
            mask_logits = res[:, 1:-1, :]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            len_mask = ~self.model.generate_len_mask(
                data["labels"].shape[0], data["in_len"]
            )
            loss = layers.cross_entropy(
                mask_logits, data["labels"].transpose(0, 1), reduction="none"
            )
            loss = loss * len_mask
            return EncoderDecoderResult(
                mask_logits, None, (loss.sum() / len_mask.sum())
            )
        else:
            res.data = res.data.transpose(0, 1)
            len_mask = ~self.model.generate_len_mask(
                out_with_eos.shape[0], out_len if train_eos else (out_len - 1)
            ).transpose(0, 1)

            loss = self.loss(res, out_with_eos, len_mask)
            return EncoderDecoderResult(res.data, res.length, loss)
