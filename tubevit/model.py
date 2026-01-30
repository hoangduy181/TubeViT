from functools import partial
from typing import Any, Callable, List, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from typing_extensions import OrderedDict

from tubevit.positional_encoding import get_3d_sincos_pos_embed


class Encoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.
    Code from torch.
    Move pos_embedding to TubeViT
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: Tensor):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        return self.ln(self.layers(self.dropout(x)))


class SparseTubesTokenizer(nn.Module):
    """
    Sparse Tubes Tokenizer with optional Space-to-Depth inspired tube embedding.
    
    Original behavior:
    - Creates tube tokens from video input using 3D convolutions
    - Each tube token has shape (T, H, W, d) where d = hidden_dim
    
    With Space-to-Depth (depth_to_space_factor = k):
    - Reduces channel dimension by factor k: (T, H, W, d) -> (T, H, W, d/k)
    - Reduces stride by factor k to get k times more tokens
    - Concatenates k neighboring tokens to recover original embedding size d
    - Increases effective receptive field while keeping same number of parameters
    """
    def __init__(
        self, 
        hidden_dim, 
        kernel_sizes, 
        strides, 
        offsets,
        depth_to_space_factor: int = 1,
        apply_on: str = "both",  # "temporal", "spatial", or "both"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.offsets = offsets
        
        # Space-to-Depth parameters
        self.depth_to_space_factor = depth_to_space_factor
        if apply_on not in ["temporal", "spatial", "both"]:
            raise ValueError(f"apply_on must be 'temporal', 'spatial', or 'both', got {apply_on}")
        self.apply_on = apply_on
        
        # Determine which axes to apply space-to-depth
        self.apply_temporal = apply_on in ["temporal", "both"]
        self.apply_spatial = apply_on in ["spatial", "both"]
        
        # Calculate effective hidden dim (reduced by factor k)
        # We always group k neighboring tokens to recover d channels
        # So: effective_hidden_dim * k = hidden_dim
        if self.depth_to_space_factor > 1:
            self.effective_hidden_dim = hidden_dim // depth_to_space_factor
            self.tokens_per_group = depth_to_space_factor
            
            # Ensure hidden_dim is divisible by depth_to_space_factor
            if hidden_dim % depth_to_space_factor != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by "
                    f"depth_to_space_factor ({depth_to_space_factor})"
                )
        else:
            self.effective_hidden_dim = hidden_dim
            self.tokens_per_group = 1

        self.conv_proj_weight = nn.Parameter(
            torch.empty(
                (self.effective_hidden_dim, 3, *self.kernel_sizes[0])
            ).normal_(),
            requires_grad=True,
        ) # (out_channels, in_channels, kT, kH, kW) 
        # initialize first tube weight with the first kernel size   

        self.register_parameter("conv_proj_weight", self.conv_proj_weight)

        self.conv_proj_bias = nn.Parameter(
            torch.zeros(len(self.kernel_sizes), self.effective_hidden_dim) # (num_tubes, effective_hidden_dim)
            , requires_grad=True)
        self.register_parameter("conv_proj_bias", self.conv_proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with optional Space-to-Depth tube embedding.
        
        Args:
            x: Input tensor of shape (n, c, t, h, w) - (batch, channels, time, height, width)
            
        Returns:
            Token tensor of shape (n, N, hidden_dim) where N is total number of tokens
            and hidden_dim is the original hidden dimension (recovered via concatenation)
            
        Tensor Shape Flow:
            Input: (n, c, t, h, w)
            After conv3d: (n, effective_hidden_dim, t_out, h_out, w_out)
            With space-to-depth:
              - If apply_on='temporal': group k tokens along T -> (n, hidden_dim, t_out/k, h_out, w_out)
              - If apply_on='spatial': group k tokens along H,W -> (n, hidden_dim, t_out, h_out/k, w_out/k)
              - If apply_on='both': group k^2 tokens -> (n, hidden_dim, t_out/k, h_out/k, w_out/k)
            Flatten: (n, hidden_dim, num_tokens)
            Permute: (n, num_tokens, hidden_dim)
        """
        n, c, t, h, w = x.shape  # CTHW
        # (batch, channels, time, height, width)

        tubes = []
        for i in range(len(self.kernel_sizes)): # loop on different tube types
            if i == 0:
                weight = self.conv_proj_weight
            else:
                weight = F.interpolate(self.conv_proj_weight, self.kernel_sizes[i], mode="trilinear")
                # interpolate the first tube weight to the current kernel size
                # scale receptive field mà không scale params.

            # Calculate modified stride based on depth_to_space_factor
            original_stride = self.strides[i]
            if self.depth_to_space_factor > 1:
                # Reduce stride by factor k to get k times more tokens along selected axes
                modified_stride = tuple(
                    max(1, s // self.depth_to_space_factor) if apply_axis else s
                    for s, apply_axis in zip(
                        original_stride,
                        [self.apply_temporal, self.apply_spatial, self.apply_spatial]
                    )
                )
            else:
                modified_stride = original_stride

            tube = F.conv3d(
                x[:, :, self.offsets[i][0] :, self.offsets[i][1] :, self.offsets[i][2] :], # process offset
                weight,
                bias=self.conv_proj_bias[i],
                stride=modified_stride,
            )
            # tube shape: (n, effective_hidden_dim, t_out, h_out, w_out)
            # where t_out, h_out, w_out are increased by factor k if space-to-depth is applied
            
            # Apply Space-to-Depth: concatenate k neighboring tokens to recover hidden_dim
            if self.depth_to_space_factor > 1:
                k = self.depth_to_space_factor
                n_batch, d_eff, t_out, h_out, w_out = tube.shape
                
                if self.apply_on == "temporal":
                    # Group k consecutive tokens along temporal dimension
                    # Input: (n, d_eff, t_out, h_out, w_out) where t_out = k * t_original
                    # Output: (n, hidden_dim, t_out/k, h_out, w_out)
                    assert t_out % k == 0, f"t_out ({t_out}) must be divisible by k ({k})"
                    tube = tube.view(n_batch, d_eff, t_out // k, k, h_out, w_out)
                    tube = tube.permute(0, 1, 3, 2, 4, 5).contiguous()  # (n, d_eff, k, t_out/k, h_out, w_out)
                    tube = tube.view(n_batch, d_eff * k, t_out // k, h_out, w_out)  # (n, hidden_dim, t_out/k, h_out, w_out)
                    
                elif self.apply_on == "spatial":
                    # Group k tokens along spatial dimensions
                    # We'll group along H dimension (k consecutive tokens)
                    # Input: (n, d_eff, t_out, h_out, w_out) where h_out = k * h_original
                    # Output: (n, hidden_dim, t_out, h_out/k, w_out)
                    assert h_out % k == 0, f"h_out ({h_out}) must be divisible by k ({k})"
                    tube = tube.view(n_batch, d_eff, t_out, h_out // k, k, w_out)
                    tube = tube.permute(0, 1, 4, 2, 3, 5).contiguous()  # (n, d_eff, k, t_out, h_out/k, w_out)
                    tube = tube.view(n_batch, d_eff * k, t_out, h_out // k, w_out)  # (n, hidden_dim, t_out, h_out/k, w_out)
                    
                else:  # apply_on == "both"
                    # Group k tokens along temporal dimension
                    # Note: For "both", we reduce stride along all axes (T, H, W) by k,
                    # but we group k tokens along temporal dimension to recover hidden_dim.
                    # This increases spatial coverage while maintaining temporal grouping.
                    # Input: (n, d_eff, t_out, h_out, w_out) where t_out = k * t_original
                    # Output: (n, hidden_dim, t_out/k, h_out, w_out)
                    # Note: h_out and w_out are also increased by k, providing better spatial coverage
                    assert t_out % k == 0, f"t_out ({t_out}) must be divisible by k ({k})"
                    tube = tube.view(n_batch, d_eff, t_out // k, k, h_out, w_out)
                    tube = tube.permute(0, 1, 3, 2, 4, 5).contiguous()  # (n, d_eff, k, t_out/k, h_out, w_out)
                    tube = tube.view(n_batch, d_eff * k, t_out // k, h_out, w_out)  # (n, hidden_dim, t_out/k, h_out, w_out)
                    
            # Reshape to (n, hidden_dim, num_tokens) for concatenation
            tube = tube.reshape((tube.shape[0], self.hidden_dim, -1))
            # Shape: (n, hidden_dim, num_tokens)

            tubes.append(tube)

        x = torch.cat(tubes, dim=-1)
        x = x.permute(0, 2, 1).contiguous() # (n, hidden_dim, N) -> (n, N, hidden_dim)

        return x 


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = nn.functional.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x


class TubeViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        video_shape: Union[List[int], np.ndarray],  # CTHW
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        representation_size=None,
        depth_to_space_factor: int = 1,
        apply_on: str = "both",  # "temporal", "spatial", or "both"
    ):
        super(TubeViT, self).__init__()
        self.video_shape = np.array(video_shape)  # (C, T, H, W)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim #(768) as ViT
        
        # Store space-to-depth parameters for positional embedding calculation
        self.depth_to_space_factor = depth_to_space_factor
        self.apply_on = apply_on
        self.apply_temporal = apply_on in ["temporal", "both"]
        self.apply_spatial = apply_on in ["spatial", "both"]

        # according to the paper, we have 4 type of tubes
        self.kernel_sizes = ( # kT, kH, kW
            (8, 8, 8), 
            (16, 4, 4), 
            (4, 12, 12),
            (1, 16, 16),
        )

        self.strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
        )

        self.offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )
        self.sparse_tubes_tokenizer = SparseTubesTokenizer(
            self.hidden_dim,
            self.kernel_sizes,
            self.strides,
            self.offsets,
            depth_to_space_factor=depth_to_space_factor,
            apply_on=apply_on,
        )

        self.pos_embedding = self._generate_position_embedding()

        self.pos_embedding = torch.nn.Parameter(self.pos_embedding, requires_grad=False)
        self.register_parameter("pos_embedding", self.pos_embedding)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        self.register_parameter("class_token", self.class_token)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.attention_pooling = SelfAttentionPooling(self.hidden_dim)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(self.hidden_dim, self.num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, self.num_classes)

        self.heads = nn.Sequential(heads_layers)

    def forward(self, x):
        x = self.sparse_tubes_tokenizer(x) # (n, N, hidden_dim)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding

        x = self.encoder(x)

        # Attention pooling
        x = self.attention_pooling(x)
        # Dense layer
        x = self.heads(x)

        return x

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        """
        Calculate output shape after conv3d, accounting for space-to-depth grouping.
        
        With space-to-depth:
        - We use modified stride (reduced by factor k)
        - This creates more tokens initially
        - Then we group k tokens, so final shape matches original stride calculation
        - But we need to account for the grouping in positional embedding
        
        Returns:
            Final token shape after grouping: (nT, nH, nW)
        """
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        
        if self.depth_to_space_factor > 1:
            # Calculate shape with modified stride (reduced by factor k)
            modified_stride = np.array([
                max(1, stride[0] // self.depth_to_space_factor) if self.apply_temporal else stride[0],
                max(1, stride[1] // self.depth_to_space_factor) if self.apply_spatial else stride[1],
                max(1, stride[2] // self.depth_to_space_factor) if self.apply_spatial else stride[2],
            ])
            
            # Shape after conv3d with modified stride
            shape_after_conv = np.floor(
                ((self.video_shape[[1, 2, 3]] - offset - kernel_size) / modified_stride) + 1
            ).astype(int)
            
            # After grouping k tokens, the final shape depends on apply_on
            k = self.depth_to_space_factor
            if self.apply_on == "temporal":
                # Group k tokens along temporal -> divide T by k
                # H and W remain unchanged
                final_shape = np.array([
                    shape_after_conv[0] // k,
                    shape_after_conv[1],
                    shape_after_conv[2],
                ])
            elif self.apply_on == "spatial":
                # Group k tokens along spatial (H) -> divide H by k
                # T and W remain unchanged
                final_shape = np.array([
                    shape_after_conv[0],
                    shape_after_conv[1] // k,
                    shape_after_conv[2],
                ])
            else:  # both
                # Group k tokens along temporal -> divide T by k
                # H and W remain increased (not grouped, so they stay at shape_after_conv)
                # This provides better spatial coverage
                final_shape = np.array([
                    shape_after_conv[0] // k,
                    shape_after_conv[1],  # Increased by k, not grouped
                    shape_after_conv[2],  # Increased by k, not grouped
                ])
            
            return final_shape
        else:
            # Original behavior: no space-to-depth
            output = np.floor(
                ((self.video_shape[[1, 2, 3]] - offset - kernel_size) / stride) + 1
            ).astype(int) #THW
            return output # (nT, nH, nW) number of tokens in each dimension

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        """
        Generate positional embeddings accounting for space-to-depth grouping.
        
        The positional embeddings need to match the final token positions after grouping.
        We use the final tube_shape (after grouping) but need to adjust the stride
        information passed to get_3d_sincos_pos_embed to reflect the effective stride
        after grouping.
        """
        position_embedding = [torch.zeros(1, self.hidden_dim)] # (1, hidden_dim)

        for i in range(len(self.kernel_sizes)): # loop on different tube types
            # Calculate final tube shape after grouping
            tube_shape = self._calc_conv_shape(
                self.kernel_sizes[i], 
                self.strides[i],
                self.offsets[i]
            )
            
            # For positional embedding with space-to-depth:
            # - We group k tokens, so final tube_shape is reduced (divided by k along grouped axis)
            # - After grouping along an axis, we keep every k-th token, so spacing = original stride
            # - For axes that are NOT grouped but have reduced stride, spacing = modified stride
            #
            # Example with temporal grouping (k=2):
            # - Modified stride: s/2, tokens at: 0, s/2, s, 3s/2, 2s, ...
            # - After grouping: keep tokens at 0, s, 2s, ... (every k-th)
            # - Final spacing along T: s (original stride)
            #
            # Example with "both" (k=2):
            # - Modified stride: (s_t/2, s_h/2, s_w/2)
            # - After grouping along T: keep every 2nd token along T
            # - Final spacing: (s_t, s_h/2, s_w/2) - T uses original, H/W use modified
            
            if self.depth_to_space_factor > 1:
                # Calculate effective stride for positional embedding
                # For grouped axes: use original stride (spacing after grouping)
                # For non-grouped axes with reduced stride: use modified stride
                k = self.depth_to_space_factor
                effective_stride = []
                for axis_idx, (orig_s, apply_reduction) in enumerate([
                    (self.strides[i][0], self.apply_temporal),
                    (self.strides[i][1], self.apply_spatial),
                    (self.strides[i][2], self.apply_spatial),
                ]):
                    if self.apply_on == "temporal" and axis_idx == 0:
                        # Temporal axis: grouped, so use original stride
                        effective_stride.append(orig_s)
                    elif self.apply_on == "spatial" and axis_idx > 0:
                        # Spatial axis: grouped, so use original stride
                        effective_stride.append(orig_s)
                    elif self.apply_on == "both":
                        if axis_idx == 0:
                            # Temporal: grouped, use original stride
                            effective_stride.append(orig_s)
                        else:
                            # Spatial: not grouped but stride reduced, use modified stride
                            effective_stride.append(max(1, orig_s // k))
                    else:
                        # No reduction on this axis
                        effective_stride.append(orig_s)
                
                effective_stride = tuple(effective_stride)
            else:
                effective_stride = self.strides[i]
            
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tuple(tube_shape),  # Final shape after grouping
                kernel_size=self.kernel_sizes[i],
                stride=effective_stride,  # Effective stride after grouping
                offset=self.offsets[i],
            ) # Returns (num_tokens_i, hidden_dim)
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous() # (sum(num_tokens_i), hidden_dim)
        # ensures contiguous memory for efficient computation
        return position_embedding


class TubeViTLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        video_shape,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        lr: float = 3e-4,
        weight_decay: float = 0,
        weight_path: str = None,
        max_epochs: int = None,
        label_smoothing: float = 0.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        depth_to_space_factor: int = 1,
        apply_on: str = "both",  # "temporal", "spatial", or "both"
        # Additional training parameters to save
        batch_size: int = None,
        frames_per_clip: int = None,
        video_size: tuple = None,
        num_workers: int = None,
        seed: int = None,
        **kwargs,
    ):
        # Save all hyperparameters including training config
        # This automatically generates hparams.yaml in the logs folder
        self.save_hyperparameters(ignore=['weight_path'])  # Exclude weight_path from hparams
        super().__init__()
        self.num_classes = num_classes
        self.model = TubeViT(
            num_classes=num_classes,
            video_shape=video_shape,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            depth_to_space_factor=depth_to_space_factor,
            apply_on=apply_on,
        )

        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.example_input_array = Tensor(1, *video_shape)

        if weight_path is not None: 
            self.model.load_state_dict(torch.load(weight_path), strict=False)

        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        # # Enable gradient checkpointing to save memory [SAVE MEMORY BUT SLOW AT BACKWARD PASS]
        # if hasattr(self.model, 'encoder'):
        #     self.model.encoder.gradient_checkpointing = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)
        self.log("train_f1", f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = self(x)
            loss = self.loss_func(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "val_acc",
            accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            sync_dist=True,
        )
        
        # Periodically clear GPU cache during validation to prevent OOM
        if batch_idx % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)
        # Force GPU memory cleanup after training epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        y_pred = torch.softmax(y_hat, dim=-1)

        return {"y": y, "y_pred": torch.argmax(y_pred, dim=-1), "y_prob": y_pred}
