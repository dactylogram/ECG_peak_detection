import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

n_channel = 2
down_ratio = 2**5
feature_shape = 2048
atrous_rate = [1,3,6,9]

def padding_helper(inputs, n_kernel, n_dilation):
    n_kernel_valid = n_kernel + (n_kernel-1)*(n_dilation-1)
    pad_total = n_kernel_valid - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end))
    return padded_inputs

class Conv1D(nn.Module):
    def __init__(self, n_in, n_out, n_kernel=3, n_stride=1, n_dilation=1):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=n_kernel, stride=n_stride, padding=0, dilation=n_dilation, groups=1, bias=False)

    def forward(self, x):
        x=padding_helper(x, self.conv1.kernel_size[0], self.conv1.dilation[0])
        x=self.conv1(x)
        return x

class CBR(nn.Module):
    def __init__(self, n_in, n_out, n_kernel=3, n_stride=1, n_dilation=1, activation=True):
        super(CBR, self).__init__()
        self.C = Conv1D(n_in, n_out, n_kernel, n_stride, n_dilation)
        self.B = nn.BatchNorm1d(n_out)
        self.R = nn.ELU(inplace=True)
        self.activation = activation

    def forward(self, x):
        x=self.C(x)
        x=self.B(x)
        if self.activation == True:
            x=self.R(x)
        return x

class SepConv1D(nn.Module):
    def __init__(self, n_in, n_out, n_kernel=3, n_stride=1, n_dilation=1):
        super(SepConv1D, self).__init__()
        self.depthwise = nn.Conv1d(n_in, n_in, kernel_size=n_kernel, stride=n_stride, padding=0, dilation=n_dilation, groups=n_in, bias=False)
        self.B = nn.BatchNorm1d(n_in)
        self.R = nn.ELU(inplace=True)
        self.pointwise = nn.Conv1d(n_in, n_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

    def forward(self, x):
        x=padding_helper(x, self.depthwise.kernel_size[0], self.depthwise.dilation[0])
        x=self.depthwise(x)
        x=self.B(x)
        x=self.R(x)
        x=self.pointwise(x)
        return x

class SBR(nn.Module):
    def __init__(self, n_in, n_out, n_kernel=3, n_stride=1, n_dilation=1, activation=True, return_skip=False):
        super(SBR, self).__init__()
        if n_kernel != 1:
            self.S = SepConv1D(n_in, n_out, n_kernel, n_stride, n_dilation)
        else:
            self.S = Conv1D(n_in, n_out, n_kernel, n_stride, n_dilation)
        self.B = nn.BatchNorm1d(n_out)
        self.R = nn.ELU(inplace=True)
        self.activation = activation
        self.return_skip = return_skip

    def forward(self, x):
        x=self.S(x)
        x1=self.B(x)
        if self.activation == True:
            x=self.R(x1)
        if self.return_skip == True:
            return x, x1
        return x

def reshape_tensor(x, seq_len):
    # x.shape = (batch_size, n_channel, seq_len * N)
    N = x.shape[-1] // seq_len
    n_batch = x.shape[0]
    n_channel = x.shape[1]

    x = x.view(n_batch, n_channel, seq_len, N)
    x = x.permute(0, 3, 1, 2)
    x = x.reshape(n_batch*N, n_channel, seq_len)
    return x, N

def restore_tensor(x, N):
    # x.shape = (batch_size * N, n_channel, seq_len)
    n_batch = x.shape[0] // N
    n_channel = x.shape[1]
    seq_len = x.shape[2]

    x = x.view(n_batch, N, n_channel, seq_len)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(n_batch, n_channel, seq_len*N)
    return x

class AvgPooling(nn.Module):
    def __init__(self, n_in, n_out):
        super(AvgPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=1, bias=False)
        self.B = nn.BatchNorm1d(n_out)
        self.R = nn.ELU(inplace=True)
        self.seq_len = feature_shape // down_ratio

    def forward(self, x):
        if x.shape[-1] != self.seq_len:
            restore = True
            x, N = reshape_tensor(x, self.seq_len)
        else:
            restore = False
        # x_shape = (batch_size * N, n_channel, seq_len)
        x=self.avgpool(x)
        # x_shape = (batch_size * N, n_channel, 1)
        x=self.conv1(x)
        x=self.B(x)
        x=self.R(x)
        if restore == True:
            x = restore_tensor(x, N)
        # x_shape = (batch_size, n_channel, 1 * N)
        x=F.interpolate(x, scale_factor=self.seq_len, mode='linear', align_corners=True)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, n_kernel=3, n_stride=1, n_dilation=1, residual=True, return_skip=False):
        super(ResidualBlock, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_kernel = n_kernel
        self.n_stride = n_stride
        self.n_dilation = n_dilation
        self.residual = residual
        self.return_skip = return_skip

        self.relu = nn.ELU(inplace=True)
        self.sbr_1 = SBR(self.n_in, self.n_out, self.n_kernel, 1, self.n_dilation, return_skip=return_skip)
        self.sbr_2 = SBR(self.n_out, self.n_out, self.n_kernel, 1, self.n_dilation)
        self.sbr_3 = SBR(self.n_out, self.n_out, self.n_kernel, self.n_stride, self.n_dilation, activation=False)
        if self.n_in != self.n_out or self.n_stride != 1:
            self.conv1 = Conv1D(self.n_in, self.n_out, 1, self.n_stride)

    def forward(self, x):
        if self.n_in != self.n_out or self.n_stride != 1:
            residual = self.conv1(x)
        else:
            residual = x
        
        x = self.relu(x) # pre-excitation

        # 3 sequential layers of Seperable Conv 1D
        if self.return_skip == True:
            x, skip = self.sbr_1(x)
        else:
            x = self.sbr_1(x)
        x = self.sbr_2(x)
        x = self.sbr_3(x)

        if self.residual == True:
            x += residual
        else:
            x = self.relu(x)
        
        if self.return_skip == True:
            return x, skip
        return x

class Encoder(nn.Module):
    def __init__(self, n_channel=n_channel, atrous_rate=atrous_rate):
        super(Encoder, self).__init__()
        self.n_channel = n_channel
        self.atrous_rate = atrous_rate

        self.cbr_e1 = CBR(self.n_channel, 16, n_stride=2)
        self.cbr_e3 = CBR(16, 16, activation=False)
        self.resblock_e1 = ResidualBlock(16, 32, n_stride=2)
        self.resblock_e2 = ResidualBlock(32, 64, n_stride=2, return_skip=True)
        self.resblock_e3 = ResidualBlock(64, 128, n_stride=2)
        self.resblock_e4 = ResidualBlock(128, 256, n_stride=2)
        self.resblock_e5 = ResidualBlock(256, 256, residual=False)

        self.aspp_1 = CBR(256, 64, n_kernel=1, n_dilation=self.atrous_rate[0])
        self.aspp_2 = CBR(256, 64, n_kernel=3, n_dilation=self.atrous_rate[1])
        self.aspp_3 = CBR(256, 64, n_kernel=3, n_dilation=self.atrous_rate[2])
        self.aspp_4 = CBR(256, 64, n_kernel=3, n_dilation=self.atrous_rate[3])
        self.aspp_5 = AvgPooling(256, 64)

        self.aspp_out = CBR(64*5, 64, n_kernel=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.cbr_e1(x)
        x = self.cbr_e3(x)

        x = self.resblock_e1(x)
        x, skip = self.resblock_e2(x)
        x = self.resblock_e3(x)
        x = self.resblock_e4(x)
        x = self.resblock_e5(x)

        b1 = self.aspp_1(x)
        b2 = self.aspp_2(x)
        b3 = self.aspp_3(x)
        b4 = self.aspp_4(x)
        b5 = self.aspp_5(x)

        aspp = torch.cat([b1, b2, b3, b4, b5], dim=1)
        aspp = self.aspp_out(aspp)
        aspp = self.dropout(aspp)

        return aspp, skip

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.skip_conv = CBR(64, 12, n_kernel=1) # skip-connection
        self.cat_conv_1 = CBR(64+12, 64)
        self.cat_conv_2 = CBR(64, 16)
        self.classifier = Conv1D(16, 1, n_kernel=1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=8, align_corners=True, mode='linear')
        skip = self.skip_conv(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.cat_conv_1(x)
        x = self.cat_conv_2(x)
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=4, align_corners=True, mode='linear')
        return x

class Sep_conv_detector(nn.Module):
    def __init__(self, n_channel=n_channel, atrous_rate=atrous_rate):
        super(Sep_conv_detector, self).__init__()
        self.enc = Encoder(n_channel, atrous_rate)
        self.dec = Decoder()
    
    def forward(self, x):
        mid, skip = self.enc(x)
        y = self.dec(mid, skip)
        return y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice
