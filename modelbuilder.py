import torch
from torch import nn
import collections
from torch.nn import functional as F

class BlockPart(nn.Module):
    def __init__(self, part_number):
        super().__init__()
        self.part_number = part_number
        self.forward = self.no_relationships

    def no_relationships(self, x):
        out = self.layers(x)
        return out

    def has_relationships(self, x):
        self.out = self.layers(x)
        return self.out

    def store_results(self):
        self.forward = self.has_relationships
        

class SegentBlockPart(BlockPart):
    def __init__(self, part_number, in_channels, out_channels, kernel, stride, padding):
        super().__init__(part_number)
        self.layers = nn.Sequential(
            *[
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel,stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        )

class SegentEncoderBlock(nn.Module):
    def __init__(self, block_number, layer_count, in_channels, out_channels, kernel=3, stride=1, padding=1, block_part=None):
        super().__init__()
        self.block_number = block_number
        if block_part is None:
            block_part = SegentBlockPart

        layers = []
        layers.append(block_part(0, in_channels=in_channels,out_channels=out_channels,kernel=kernel,stride=stride, padding=padding))
        for layer in range(1, layer_count):
            layers.append(block_part(layer, in_channels=out_channels,out_channels=out_channels,kernel=kernel,stride=stride, padding=padding))

        self.layers = layers
        self.layers_executor = nn.Sequential(*layers)

    def forward(self, x):
        layers_out = self.layers_executor(x)
        out, self.indices = F.max_pool2d(layers_out, kernel_size=2, stride=2, return_indices=True)
        return out


class SegnetDecoderBlock(nn.Module):
    def __init__(self, block_number, layer_count, in_channels, out_channels, kernel=3, stride=1, padding=1, block_part=None):
        super().__init__()
        self.block_number = block_number
        if block_part is None:
            block_part = SegentBlockPart

        layers = []
        for layer in range(layer_count - 1):
            layers.append(block_part(layer, in_channels=in_channels,out_channels=in_channels,kernel=kernel,stride=stride, padding=padding))
        layers.append(block_part(len(layers), in_channels=in_channels,out_channels=out_channels,kernel=kernel,stride=stride, padding=padding))

        self.layers = layers
        self.layers_executor = nn.Sequential(*layers)

    def forward(self, x, indices):
        out = self.layers_executor(F.max_unpool2d(x, indices, kernel_size=2, stride=2))
        return out



class EncoderDecoderBlockPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward = self.no_deeper_blocks
        self.deeper_blocks = []

    def set_blocks(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def no_deeper_blocks(self, x):
        en_out = self.encoder(x)
        de_out = self.decoder(en_out, self.encoder.indices)
        return de_out

    def has_deeper_blocks(self, x):
        en_out = self.encoder(x)
        sub_blk_out = self.sub_network(en_out)
        de_out = self.decoder(sub_blk_out, self.encoder.indices)
        return de_out

    def add_deeper_block(self, block):
        self.deeper_blocks.append(block)
        self.sub_network = nn.Sequential(*self.deeper_blocks)

        self.forward = self.has_deeper_blocks


def SegnetEncoderDecoderBlockPairBuilder(block_number, layer_count, in_channels, out_channels, kernel=3, stride=1, padding=1, block_part=None):
        if block_part is None:
            block_part = SegentBlockPart

        encoder = SegentEncoderBlock(block_number, layer_count, in_channels=in_channels, out_channels=out_channels, kernel=kernel, stride=stride, padding=padding, block_part=block_part)
        decoder = SegnetDecoderBlock(block_number, layer_count, in_channels=out_channels, out_channels=in_channels, kernel=kernel, stride=stride, padding=padding, block_part=block_part)
        
        pair = EncoderDecoderBlockPair()
        pair.set_blocks(encoder, decoder)

        return pair

def get_default_model(in_channels, classes):
    block_infos = [
        (3, 64, 128),
        (3, 128, 256),
        (3, 256, 512),
        (3, 512, 512),
    ]

    encoder = SegentEncoderBlock(0, 2, in_channels, block_infos[0][1])
    decoder = SegnetDecoderBlock(0, 2, block_infos[0][1], classes)
    first_pair = EncoderDecoderBlockPair()
    first_pair.set_blocks(encoder, decoder)

    last_pair = first_pair

    for index, (layers, in_chs, out_chs) in enumerate(block_infos):
        current_pair = SegnetEncoderDecoderBlockPairBuilder(index + 1, layers, in_chs, out_chs)
        last_pair.add_deeper_block(current_pair)
        last_pair = current_pair
    
    parts = []
    parts.append(("network", first_pair))
    parts.append(("output", nn.Softmax2d()))

    model = nn.Sequential(collections.OrderedDict(parts))
    return model