from .MotionPrior import * 
from .VectorQuantizer import VectorQuantizer
from .GumbelVectorQuantizer import GumbelVectorQuantizer
from torch import nn
from ..BlockFactory import preprocessor_from_cfg
from omegaconf import open_dict
from gdl.utils.other import class_from_str
import sys
from gdl.models.temporal.PositionalEncodings import positional_encoding_from_cfg
from gdl.models.temporal.TransformerMasking import biased_mask_from_cfg
from munch import Munch, munchify
from omegaconf import OmegaConf

class L2lVqVae(MotionPrior):

    def __init__(self, cfg) -> None:
        # motion_encoder = motion_decoder_from_cfg(cfg)
        # motion_decoder = motion_decoder_from_cfg(cfg)
        # motion_codebook = motion_codebook_from_cfg(cfg)
        self.cfg = cfg
        input_dim = self.get_input_sequence_dim()
        
        encoder_class = class_from_str(cfg.model.sequence_encoder.type, sys.modules[__name__])
        decoder_class = class_from_str(cfg.model.sequence_decoder.type, sys.modules[__name__])

        with open_dict(cfg.model.sequence_encoder):
            cfg.model.sequence_encoder.input_dim = input_dim

        with open_dict(cfg.model.sizes):
            assert cfg.learning.batching.sequence_length_train % (2 **cfg.model.sizes.quant_factor) == 0, \
                "Sequence length must be divisible by quantization factor"
            cfg.model.sizes.quant_sequence_length = cfg.learning.batching.sequence_length_train // (2 **cfg.model.sizes.quant_factor)
            cfg.model.sizes.sequence_length = cfg.learning.batching.sequence_length_train 
            if 'quantizer' in cfg.model.keys():
                cfg.model.sizes.bottleneck_feature_dim = cfg.model.quantizer.vector_dim 
                if encoder_class == L2lEncoderWithClassificationHead:
                    cfg.model.sizes.num_classes = cfg.model.quantizer.codebook_size

        # motion_encoder = L2lEncoder(cfg.model.sequence_encoder, cfg.model.sizes)
        motion_encoder = encoder_class(cfg.model.sequence_encoder, cfg.model.sizes)
        
        # motion_decoder = L2lDecoder(cfg.model.sequence_decoder, cfg.model.sizes, motion_encoder.get_input_dim())
        motion_decoder = decoder_class(cfg.model.sequence_decoder, cfg.model.sizes, motion_encoder.get_input_dim())
        # motion_quantizer = VectorQuantizer(cfg.model.quantizer)
        if cfg.model.get('quantizer', None) is not None:
            motion_quantizer = class_from_str(cfg.model.quantizer.type, sys.modules[__name__])(cfg.model.quantizer)
        else:
            motion_quantizer = None
        if cfg.model.get('preprocessor', None) is not None:
            preprocessor = preprocessor_from_cfg(cfg.model.preprocessor)
        else:
            preprocessor = None
        super().__init__(cfg, motion_encoder, motion_decoder, motion_quantizer, preprocessor, preprocessor)

    def get_bottleneck_dim(self):
        return self.motion_encoder.bottleneck_dim()

    def get_codebook_size(self): 
        if self.motion_quantizer is None:
            return None
        else:
            return self.motion_quantizer.codebook_size

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'L2lVqVae':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = L2lVqVae(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = L2lVqVae.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model


def create_squasher(input_dim, hideen_dim, quant_factor):
    layers = [nn.Sequential(
            nn.Conv1d(input_dim, hideen_dim,5,stride=2,padding=2,
                        padding_mode='replicate'),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(hideen_dim))]
    for _ in range(1, quant_factor):
        layers += [nn.Sequential(
                    nn.Conv1d(hideen_dim,hideen_dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(hideen_dim),
                    nn.MaxPool1d(2)
                    )]
    squasher = nn.Sequential(*layers)
    return squasher


class L2lEncoder(MotionEncoder): 
    """
    Inspired by by the encoder from Learning to Listen.
    """

    def __init__(self, cfg, sizes):
        super().__init__()
        self.config = cfg
        # size=self.config['transformer_config']['in_dim']
        self.sizes = sizes
        size = self.config.input_dim
        # dim=self.config['transformer_config']['hidden_size']
        dim = self.config.feature_dim
        # layers = [nn.Sequential(
        #             nn.Conv1d(size,dim,5,stride=2,padding=2,
        #                         padding_mode='replicate'),
        #             nn.LeakyReLU(0.2, True),
        #             nn.BatchNorm1d(dim))]
        # for _ in range(1, sizes.quant_factor):
        #     layers += [nn.Sequential(
        #                 nn.Conv1d(dim,dim,5,stride=1,padding=2,
        #                             padding_mode='replicate'),
        #                 nn.LeakyReLU(0.2, True),
        #                 nn.BatchNorm1d(dim),
        #                 nn.MaxPool1d(2)
        #                 )]
        # self.squasher = nn.Sequential(*layers) 
        self.squasher = create_squasher(size, dim, sizes.quant_factor)
        # the purpose of the squasher is to reduce the FPS of the input sequence

        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, 
                    nhead=cfg.nhead, 
                    dim_feedforward=cfg.intermediate_size, 
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # self.encoder_transformer = Transformer(
        #     in_size=self.config['transformer_config']['hidden_size'],
        #     hidden_size=self.config['transformer_config']['hidden_size'],
        #     num_hidden_layers=\
        #             self.config['transformer_config']['num_hidden_layers'],
        #     num_attention_heads=\
        #             self.config['transformer_config']['num_attention_heads'],
        #     intermediate_size=\
        #             self.config['transformer_config']['intermediate_size'])
        
        pos_enc_cfg = munchify(OmegaConf.to_container(cfg.positional_encoding))
        pos_enc_cfg.max_seq_len = sizes.quant_sequence_length
        self.encoder_pos_embedding = positional_encoding_from_cfg(pos_enc_cfg, self.config.feature_dim)

        attention_mask_cfg = cfg.get('temporal_bias', None)
        if attention_mask_cfg is not None:
            attention_mask_cfg = munchify(OmegaConf.to_container(attention_mask_cfg))
            attention_mask_cfg.nhead = cfg.nhead 
            self.attention_mask = biased_mask_from_cfg(attention_mask_cfg)
        else:
            self.attention_mask = None

        self.encoder_linear_embedding = nn.Linear(
            self.config.feature_dim,
            self.config.feature_dim
        )

    def get_input_dim(self):
        return self.config.input_dim
        # return input_dim

    def forward(self, batch, input_key="input_sequence", output_key="encoded_features", **kwargs):
        # ## downsample into path-wise length seq before passing into transformer
        # dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = batch[input_key]
        # input is batch first: B, T, D but the convolution expects B, D, T
        # so we need to permute back and forth
        inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1)
        
        encoded_features = self.encoder_linear_embedding(inputs)
        
        # add positional encoding
        if self.encoder_pos_embedding is not None:
            encoded_features = self.encoder_pos_embedding(encoded_features)
        
        # add attention mask (if any)
        B, T = encoded_features.shape[:2]
        mask = None
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=encoded_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
    
        encoded_features = self.encoder_transformer(encoded_features, mask=mask)
        batch[output_key] = encoded_features
        return batch

    def bottleneck_dim(self):
        return self.config.feature_dim
    
    def latent_temporal_factor(self): 
        return 2 ** self.sizes.quant_factor

    def quant_factor(self): 
        return self.sizes.quant_factor


class L2lEncoderWithClassificationHead(L2lEncoder): 

    def __init__(self, cfg, sizes) -> None:
        super().__init__(cfg, sizes) 
        self.classification_head = nn.Linear(self.config.feature_dim, sizes.num_classes)

    def forward(self, batch, input_key="input_sequence", output_key="encoded_features", **kwargs):
        batch = super().forward(batch, input_key, output_key, **kwargs)
        batch[output_key] = self.classification_head(batch[output_key])
        return batch


class L2lEncoderWithGaussianHead(L2lEncoder): 

    def __init__(self, cfg, sizes) -> None:
        super().__init__(cfg, sizes) 
        self.mean = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        self.logvar = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.logvar.weight.device)
        self.N.scale = self.N.scale.to(self.logvar.weight.device)

    def to(self, *args, device=None, **kwargs):
        super().to(*args, device=None, **kwargs)
        self.N.loc.to(device)
        self.N.scale.to(device)
        self.mean = self.mean.to(device)
        self.logvar = self.logvar.to(device)
        return self

    def forward(self, batch, input_key="input_sequence", output_key="encoded_features", **kwargs):
        if self.N.loc.device != self.logvar.weight.device:
            self.N.loc = self.N.loc.to(self.logvar.weight.device)
            self.N.scale = self.N.scale.to(self.logvar.weight.device)
        batch = super().forward(batch, input_key, output_key, **kwargs)
        encoded_feature = batch[output_key]
        B,T = encoded_feature.shape[:2]
        encoded_feature = encoded_feature.reshape(B*T, -1)
        mean = self.mean(encoded_feature)
        logvar = self.logvar(encoded_feature)
        std = torch.exp(0.5*logvar)
        # eps = self.N.sample(std.size())
        z = mean + std * self.N.sample(mean.shape)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
        batch["kl_divergence"] = kld_loss
        z = z.reshape(B,T,-1)
        batch[output_key] = z
        batch[output_key + "_mean"] = mean.reshape(B,T,-1)
        batch[output_key + "_logvar"] = logvar.reshape(B,T,-1)
        batch[output_key + "_std"] = std.reshape(B,T,-1)
        
        return batch


class L2lDecoder(MotionEncoder): 

    def __init__(self, cfg, sizes, out_dim): 
        super().__init__()
        is_audio = False
        self.cfg = cfg
        size = self.cfg.feature_dim
        dim = self.cfg.feature_dim
        self.expander = nn.ModuleList()
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        # padding_mode='replicate' # crashes, only zeros padding mode allowed
                                        padding_mode='zeros' 
                                        ),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim)))
        num_layers = sizes.quant_factor + 2 \
            if is_audio else sizes.quant_factor
        # TODO: check if we need to keep the sequence length fixed
        seq_len = sizes.sequence_length*4 \
            if is_audio else sizes.sequence_length
        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(0.2, True),
                                nn.BatchNorm1d(dim),
                                ))
        decoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, 
                    nhead=cfg.nhead, 
                    dim_feedforward=cfg.intermediate_size, 
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.decoder_transformer = torch.nn.TransformerEncoder(decoder_layer, num_layers=cfg.num_layers)

        
        pos_enc_cfg = munchify(OmegaConf.to_container(cfg.positional_encoding))
        pos_enc_cfg.max_seq_len = seq_len
        self.decoder_pos_embedding = positional_encoding_from_cfg(pos_enc_cfg, self.cfg.feature_dim)

        attention_mask_cfg = cfg.get('temporal_bias', None)
        if attention_mask_cfg is not None:
            attention_mask_cfg = munchify(OmegaConf.to_container(attention_mask_cfg))
            attention_mask_cfg.nhead = cfg.nhead 
            self.attention_mask = biased_mask_from_cfg(attention_mask_cfg)
        else:
            self.attention_mask = None


        self.decoder_linear_embedding = nn.Linear(
            self.cfg.feature_dim,
            self.cfg.feature_dim
            )

        conv_smooth_layer = cfg.get('conv_smooth_layer', 'l2l_default')
        
        if conv_smooth_layer == 'l2l_default':
            self.cross_smooth_layer = nn.Conv1d(
                    cfg.feature_dim,
                    out_dim, 5, 
                    padding=2
                )
        elif conv_smooth_layer is False:
            self.cross_smooth_layer = None
        else:
            raise ValueError(f'conv_smooth_layer value {conv_smooth_layer} not supported')

        if cfg.get('post_transformer_proj', None):
            lin_out_dim = cfg.feature_dim if self.cross_smooth_layer is not None else out_dim
            self.post_transformer_linear = nn.Linear(
                cfg.feature_dim,
                lin_out_dim,
            )
            if cfg.post_transformer_proj.init == "zeros":
                torch.nn.init.zeros_(self.post_transformer_linear.weight)
                torch.nn.init.zeros_(self.post_transformer_linear.bias)
        else:
            self.post_transformer_linear = None

        if cfg.get('post_conv_proj', None):
            self.post_conv_proj = nn.Linear(
                out_dim,
                out_dim,
            )
            if cfg.post_conv_proj.init == "zeros":
                torch.nn.init.zeros_(self.post_conv_proj.weight)
                torch.nn.init.zeros_(self.post_conv_proj.bias)
        else:
            self.post_conv_proj = None

        # initialize the last layer of the decoder to zero 
        if cfg.get('last_layer_init', None) == "zeros":
            torch.nn.init.zeros_(self.decoder_transformer.layers[-1].linear2.weight)
            torch.nn.init.zeros_(self.decoder_transformer.layers[-1].linear2.bias)

    def forward(self, batch, input_key="quantized_features", output_key="decoded_sequence", **kwargs):
        # dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        ## upsample to the original length of the sequence before passing into transformer
        inputs = batch[input_key]
        for i, module in enumerate(self.expander):
            # input is batch first: B, T, D but the convolution expects B, D, T
            # so we need to permute back and forth
            inputs = module(inputs.permute(0,2,1)).permute(0,2,1)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=1)
        
        decoded_features = self.decoder_linear_embedding(inputs)
        
        # add positional encoding
        if self.decoder_pos_embedding is not None:
            decoded_features = self.decoder_pos_embedding(decoded_features)
        
        # add attention mask bias (if any)
        B,T = decoded_features.shape[:2]
        mask = None
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=decoded_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)

        decoded_features = self.decoder_transformer(decoded_features, mask=mask)
        decoded_reconstruction = decoded_features
        if self.post_transformer_linear is not None:
            decoded_reconstruction = self.post_transformer_linear(decoded_reconstruction)
        if self.cross_smooth_layer is not None:
            decoded_reconstruction = self.cross_smooth_layer(decoded_reconstruction.permute(0,2,1)).permute(0,2,1)
        if self.post_conv_proj is not None:
            decoded_reconstruction = self.post_conv_proj(decoded_reconstruction)
        batch[output_key] = decoded_reconstruction
        return batch


class CodeTalkerEncoder(MotionEncoder): 
    """
    Inspired by by the encoder from CodeTalker
    """

    def __init__(self, cfg, sizes):
        super().__init__()
        self.config = cfg
        self.sizes = sizes
        size = self.config.input_dim
        dim = self.config.feature_dim

        self.lin1 = nn.Linear(size, dim)
        self.lrelu1 = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='zeros')
        self.lrelu2 = nn.LeakyReLU(0.2, True)
        self.norm = nn.InstanceNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)

                
        pos_enc_cfg = munchify(OmegaConf.to_container(cfg.positional_encoding))
        pos_enc_cfg.max_seq_len = sizes.quant_sequence_length
        self.encoder_pos_embedding = positional_encoding_from_cfg(pos_enc_cfg, self.config.feature_dim)

        attention_mask_cfg = cfg.get('temporal_bias_type', None)
        if attention_mask_cfg is not None:
            self.attention_mask = biased_mask_from_cfg(munchify(OmegaConf.to_container(attention_mask_cfg)))
        else:
            self.attention_mask = None

        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, # default 1024
                    nhead=cfg.nhead, # default 8
                    dim_feedforward=cfg.intermediate_size, # default 1536
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, 
            num_layers=cfg.num_layers # 6 by default
        )

        self.lin3 = nn.Linear(dim, sizes.bottleneck_feature_dim)

    def get_input_dim(self):
        return self.config.input_dim

    def forward(self, batch, input_key="input_sequence", output_key="encoded_features", **kwargs):
        inputs = batch[input_key] 
        B, T = inputs.shape[:2]
        inputs = inputs.reshape(B * T, -1)
        encoded_features = self.lin1(inputs)
        encoded_features = self.lrelu1(encoded_features)
        encoded_features = encoded_features.reshape(B, T, -1)
        encoded_features = self.conv1(encoded_features.permute(0,2,1)).permute(0,2,1)
        encoded_features = encoded_features.reshape(B * T, -1)
        encoded_features = self.lrelu2(encoded_features)
        encoded_features = self.norm(encoded_features)
        encoded_features = self.lin2(encoded_features)

        encoded_features = encoded_features.reshape(B, T, -1)
        
        # add positional encoding
        if self.encoder_pos_embedding is not None:
            encoded_features = self.encoder_pos_embedding(encoded_features)
        
        # add attention mask bias (if any)
        mask = None
        B, T = encoded_features.shape[:2]
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=encoded_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
    
        encoded_features = self.encoder_transformer(encoded_features, mask=mask)
        encoded_features = encoded_features.reshape(B * T, -1)
        encoded_features = self.lin3(encoded_features)
        encoded_features = encoded_features.reshape(B, T, -1)
        batch[output_key] = encoded_features
        return batch


class CodeTalkerDecoder(MotionDecoder): 
    
    def __init__(self, cfg, sizes, out_dim): 
        super().__init__()
        self.cfg = cfg
        size = self.cfg.feature_dim
        dim = self.cfg.feature_dim
        is_audio = False

        self.lin1 = nn.Linear(sizes.bottleneck_feature_dim, dim)
        self.conv1 = nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='zeros') 
        self.lrelu1 = nn.LeakyReLU(0.2, True) 
        self.norm = nn.InstanceNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        seq_len = sizes.sequence_length*4 if is_audio else sizes.sequence_length

        attention_mask_cfg = cfg.get('temporal_bias_type', None)
        if attention_mask_cfg is not None:
            self.attention_mask = biased_mask_from_cfg(munchify(OmegaConf.to_container(attention_mask_cfg)))
        else:
            self.attention_mask = None
                
        pos_enc_cfg = munchify(OmegaConf.to_container(cfg.positional_encoding))
        pos_enc_cfg.max_seq_len = seq_len
        self.decoder_pos_embedding = positional_encoding_from_cfg(pos_enc_cfg, self.cfg.feature_dim)

        decoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, # default 1024
                    nhead=cfg.nhead, # default 8
                    dim_feedforward=cfg.intermediate_size, # default 1536
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.decoder_transformer = torch.nn.TransformerEncoder(decoder_layer, 
            num_layers=cfg.num_layers # 6 by default
        )

        self.lin3 = nn.Linear(dim, out_dim)

    def forward(self, batch, input_key="quantized_features", output_key="decoded_sequence", **kwargs):
        inputs = batch[input_key]
        B, T = inputs.shape[:2]
        inputs = inputs.reshape(B * T, -1)
        decoded_features = self.lin1(inputs)
        decoded_features = decoded_features.reshape(B, T, -1)
        decoded_features = self.conv1(decoded_features.permute(0,2,1)).permute(0,2,1)
        decoded_features = decoded_features.reshape(B * T, -1)
        decoded_features = self.lrelu1(decoded_features)
        decoded_features = self.norm(decoded_features)
        decoded_features = self.lin2(decoded_features)

        decoded_features = decoded_features.reshape(B, T, -1)

        # add positional encoding (if any)
        if self.decoder_pos_embedding is not None:
            decoded_features = self.decoder_pos_embedding(decoded_features)
        
        # add attention mask (if any)
        mask = None
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=decoded_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
    
    
        decoded_features = self.decoder_transformer(decoded_features, mask=mask)
        decoded_features = decoded_features.reshape(B * T, -1)

        decoded_features = self.lin3(decoded_features)
        decoded_features = decoded_features.reshape(B, T, -1)
        batch[output_key] = decoded_features
        return batch


