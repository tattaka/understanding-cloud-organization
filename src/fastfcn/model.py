import sys
sys.path.append('../')

from .decoder import FastFCNDecoder, FastFCNImproveDecoder
from base import EncoderDecoder
from encoders import get_encoder

        
class FastFCN(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            classes=1,
            activation='sigmoid',
            mid_channel = 128, 
            tta=False, 
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = FastFCNDecoder(
            encoder_channels=encoder.out_shapes,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            mid_channel = mid_channel,
        )

        super().__init__(encoder, decoder, activation, tta)

        self.name = 'fastfcn-{}'.format(encoder_name)

class FastFCNImprove(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            classes=1,
            activation='sigmoid',
            mid_channel = 128, 
            tta=False, 
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = FastFCNImproveDecoder(
            encoder_channels=encoder.out_shapes,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            mid_channel = mid_channel,
        )

        super().__init__(encoder, decoder, activation, tta)

        self.name = 'fastfcn_improve-{}'.format(encoder_name)