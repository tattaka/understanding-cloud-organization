import sys
sys.path.append('../')

from .decoder import RefineNetPoolingImproveDecoder, RefineNetDecoder
from base import EncoderDecoder
from encoders import get_encoder

class RefineNetPoolingImprove(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            input_shape=(3, 384),
            classes=1,
            activation='sigmoid',
            features = 256,
            tta=False 
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder =RefineNetPoolingImproveDecoder(
             input_shape,
             encoder_channels=encoder.out_shapes,
             num_classes=classes,
             features=features
        )

        super().__init__(encoder, decoder, activation, tta)

        self.name = 'refinenet-pooling-improve-{}'.format(encoder_name)
        
class RefineNet(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            input_shape=(3, 384),
            classes=1,
            activation='sigmoid',
            features = 256,
            tta=False 
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder =RefineNetDecoder(
             input_shape,
             encoder_channels=encoder.out_shapes,
             num_classes=classes,
             features=features
        )

        super().__init__(encoder, decoder, activation, tta)

        self.name = 'refinenet-{}'.format(encoder_name)