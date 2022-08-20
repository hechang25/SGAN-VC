
import os
import os.path as osp

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class PartUpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=(1, 2), mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='in', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            #reshape input
            out = out.unsqueeze(1)
            out = self.norm(out)
            out = out.view(out.size(0),out.size(2))
        if self.activation:
            out = self.activation(out)
        return out


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        # self.upsample = UpSample(upsample)
        self.upsample = PartUpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))



class Generator(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=48*8,
                 repeat_num=4, res_num=2, multimlp=True,
                 in_style_dim=2048, content_loss=True, mel_max=0, mel_min=0
                 ):
        super().__init__()

        self.in_style_dim = in_style_dim
        self.multimlp = multimlp
        self.content_loss = content_loss
        self.mel_max, self.mel_min = mel_max, mel_min
        
        self.toneShift = ToneShift()

        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode1 = nn.ModuleList()
        self.decode2 = nn.ModuleList()
        self.decode3 = nn.ModuleList()
        self.decode4 = nn.ModuleList()
        self.to_out1 = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, dim_in, 1, 1, 0),
        )
        self.to_out2 = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, dim_in, 1, 1, 0),
        )
        self.to_out3 = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, dim_in, 1, 1, 0),
        )
        self.to_out4 = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, dim_in, 1, 1, 0),
        )
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 3, 1, 1))

        # down21
        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=_downtype))

            if repeat_num == 2 and lid == 0:
                _downtype = 'half'
            elif repeat_num == 2:
                _downtype = "none"

            self.decode1.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=_downtype))  # stack-like
            self.decode2.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=_downtype))  # stack-like
            self.decode3.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=_downtype))  # stack-like
            self.decode4.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(res_num):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))

        # bottleneck blocks (decoder)
        for _ in range(res_num):
            self.decode1.insert(0, AdainResBlk(dim_out, dim_out, style_dim))
            self.decode2.insert(0, AdainResBlk(dim_out, dim_out, style_dim))
            self.decode3.insert(0, AdainResBlk(dim_out, dim_out, style_dim))
            self.decode4.insert(0, AdainResBlk(dim_out, dim_out, style_dim))


        if self.multimlp:
            self.mlp_w1 = MLP(self.in_style_dim, style_dim, 512, 3, norm="none", activ="lrelu")
            self.mlp_w2 = MLP(self.in_style_dim, style_dim, 512, 3, norm="none", activ="lrelu")
            self.mlp_w3 = MLP(self.in_style_dim, style_dim, 512, 3, norm="none", activ="lrelu")
            self.mlp_w4 = MLP(self.in_style_dim, style_dim, 512, 3, norm="none", activ="lrelu")

        self.dropout = nn.Dropout(p=0.2)

        self.conv1x1 = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 1, 1, 0),
        )
        
        
        self.conv1x1_reduction = nn.Sequential(
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 1, 1, 0),
        )
        
    def encode_forward(self, x):
        enc_x = self.stem(x)
        for item, block in enumerate(self.encode):
            enc_x = block(enc_x)
            if item == 0:
                enc_recover_size = F.interpolate(enc_x, scale_factor=2, mode='nearest')
        return enc_x, enc_recover_size


    def decode_forward(self, enc_x, s):
        if self.multimlp:

            s1 = self.mlp_w1(s[:, :self.in_style_dim])
            s2 = self.mlp_w2(s[:, self.in_style_dim: self.in_style_dim * 2])
            s3 = self.mlp_w3(s[:, self.in_style_dim * 2: self.in_style_dim * 3])
            s4 = self.mlp_w4(s[:, self.in_style_dim * 3:])
            s = torch.cat((s1, s2, s3, s4), 1)

        # decode1
        x1 = enc_x.clone()
        for block in self.decode1:
            x1 = block(x1, s1)

        # decode2
        x2 = enc_x.clone()
        for block in self.decode2:
            x2 = block(x2, s2)
            
        # decode3
        x3 = enc_x.clone()
        for block in self.decode3:
            x3 = block(x3, s3)

        # decode4
        for block in self.decode4:
            enc_x = block(enc_x, s4)

        x1 = self.to_out1(x1)
        x2 = self.to_out2(x2)
        x3 = self.to_out3(x3)
        enc_x = self.to_out4(enc_x)
        x = torch.cat((x1, x2, x3, enc_x), dim=2)
        return x

    def forward(self, x, s):
        enc_x, enc_recover_size = self.encode_forward(x)
        
        enc_x = self.toneShift(enc_x)

        if self.content_loss:
            encoder_feature = enc_x

        # add dropout
        enc_x = self.dropout(enc_x)
        enc_x = self.conv1x1(enc_x)

        x = self.decode_forward(enc_x, s)
        # add shortcut
        enc_recover_size = self.conv1x1_reduction(enc_recover_size)
        x = x + enc_recover_size

        if self.content_loss:
            return torch.clamp(self.to_out(x), self.mel_min, self.mel_max), encoder_feature    # vctk
        else:
            return torch.clamp(self.to_out(x), self.mel_min, self.mel_max)   # vctk


class StyleEncoder(nn.Module):
    def __init__(self, class_num, norm=False, stride=2, droprate=0.5, pool='avg'):
        super(StyleEncoder, self).__init__()
        model_ft = models.resnet50(pretrained=True)

        weight = model_ft.conv1.weight.clone()

        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        model_ft.conv1.weight.data = weight.sum(dim=(1), keepdim=True) / 3

        self.part = 4
        if pool == 'max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = model_ft

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv2.stride = (1,1)  # resnet50

        # resnet50
        self.classifier_total = ClassBlock(2048, class_num, 0.5)
        self.classifier_1 = ClassBlock(2048, class_num, 0.75)
        self.classifier_2 = ClassBlock(2048, class_num, 0.75)
        self.classifier_3 = ClassBlock(2048, class_num, 0.75)
        self.classifier_4 = ClassBlock(2048, class_num, 0.75)

    def get_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)

        f1 = f[:, :, 0:1, :].view(f.size(0), f.size(1))
        f2 = f[:, :, 1:2, :].view(f.size(0), f.size(1))
        f3 = f[:, :, 2:3, :].view(f.size(0), f.size(1))
        f4 = f[:, :, 3:, :].view(f.size(0), f.size(1))

        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
       
        # add global feature         
        f1 = torch.cat((x, f1), dim=1)
        f2 = torch.cat((x, f2), dim=1)
        f3 = torch.cat((x, f3), dim=1)
        f4 = torch.cat((x, f4), dim=1)

        f = torch.cat((f1, f2, f3, f4), dim=1)

        return f, x

    def forward(self, x):
        f, x_total = self.get_features(x)

        x_total = self.classifier_total(x_total)
        return f, x_total



class Discriminator(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()

        # real/fake discriminator
        self.dis = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        self.num_domains = num_domains

    def forward(self, x, y):
        return self.dis(x, y)

    def classifier(self, x):
        return self.cls.get_feature(x)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        self.main = nn.Sequential(*blocks)
        self.conv1x1 = nn.Conv2d(dim_out, num_domains, 1, 1, 0)


    def get_feature(self, x):
        # print("dis...........")
        out = self.main(x)  # (bs, 256)
        out = self.conv1x1(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out


    def forward(self, x, y):
        out = self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        #num_bottleneck = input_dim # We remove the input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck, affine=True)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


def build_model(args):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim,
                          w_hpf=args.w_hpf, repeat_num=args.gen_n_repeat, res_num=args.gen_n_res,
                          multimlp=args.multimlp, in_style_dim=args.in_style_dim, content_loss=args.content_loss,
                          mel_max=args.mel_max, mel_min=args.mel_min)
    
    style_encoder = StyleEncoder(args.classes_num, stride=args.stride, norm=args.norm, pool=args.pool)
    id_model_path = os.path.join("id_models", args.data_name + "_max_acc1.pth")
    style_encoder.load_state_dict(torch.load(id_model_path))
    style_encoder.eval()
    
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.dis_n_repeat)

    generator_ema = copy.deepcopy(generator)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 style_encoder=style_encoder,
                 discriminator=discriminator)

    for key in nets:
        if nets[key] is not None:
            total = sum([param.nelement() for param in nets[key].parameters()])
            print(key, "Number of parameter: %.2fM" % (total / 1e6))

    nets_ema = Munch(generator=generator_ema,
                     style_encoder=style_encoder_ema)
    return nets, nets_ema
    

class ToneShift(nn.Module):
    def __init__(self):
        super(ToneShift, self).__init__()

        self.conv_shift = nn.Sequential(
                             nn.Conv2d(256, 32, kernel_size=5, stride=(2,1), padding=2), 
                             nn.InstanceNorm2d(32, affine=True),
                             nn.LeakyReLU(0.2),
                             nn.Conv2d(32, 32, kernel_size=5, stride=(2,1), padding=2), 
                             nn.InstanceNorm2d(32, affine=True),
                             nn.LeakyReLU(0.2),
                             nn.Conv2d(32, 32, kernel_size=5, stride=(2,1), padding=2), 
                             nn.InstanceNorm2d(32, affine=True),
                             nn.LeakyReLU(0.2),
                             nn.Conv2d(32, 32, kernel_size=5, stride=(2,1), padding=2), 
                             nn.InstanceNorm2d(32, affine=True),
                             nn.LeakyReLU(0.2),
                             nn.Conv2d(32, 32, kernel_size=5, stride=(2,1), padding=2), 
                             nn.InstanceNorm2d(32, affine=True),
                             nn.LeakyReLU(0.2),
                             nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), 
                             nn.LeakyReLU(0.2),
                             nn.Tanh()
                             )
        
    def forward(self, x_org):
        B, _, H, W = x_org.shape
        
        vertical_shift = self.conv_shift(x_org).view(x_org.size(0), x_org.size(-1)) * 0.3
        
        grid = torch.zeros(B, H, W, 2).cuda()
        
        y_axis = torch.arange(H).unsqueeze(1).repeat(1, W).cuda()
        x_axis = torch.arange(W).unsqueeze(0).repeat(H, 1).cuda()
        
        x_axis = ((x_axis - 56) / (56)).cuda()
        grid[:, :, :, 0] = x_axis
        
        vertical_shift = vertical_shift.unsqueeze(1).repeat(1,H,1)
        y_axis = y_axis.unsqueeze(0).repeat(B,1,1)

        y_axis = (((y_axis + vertical_shift * 10)  - 10) / 10)
        grid[:, :, :, 1] = y_axis
        
        grid = torch.clamp(grid, -1, 1)
        x = F.grid_sample(x_org, grid, mode='bilinear', padding_mode="border", align_corners=True).cuda()
        
        return x


if __name__ == '__main__':

    net = ToneShift()
    x = torch.randn(1, 1, 80, 224)
    
    output = net(x)
    print(x.shape)
